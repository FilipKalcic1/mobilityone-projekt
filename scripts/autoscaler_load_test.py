"""
Autoscaler Load & Stress Test Suite

TESTS:
1. Gradual Ramp-up: 100 req/s -> +100 every 5 min until limit
2. Burst/Spike Test: 5000 messages in 10 seconds
3. Cooldown Test: Monitor instance shutdown after traffic stops

KPI TRACKING:
- Pod Startup Latency: Time for new worker to become active
- Scaling Threshold: CPU/Queue depth when scaling triggers
- Memory Usage: Track for leaks during soak test

USAGE:
    python scripts/autoscaler_load_test.py --test gradual
    python scripts/autoscaler_load_test.py --test burst
    python scripts/autoscaler_load_test.py --test cooldown
    python scripts/autoscaler_load_test.py --test soak --duration 28800
    python scripts/autoscaler_load_test.py --test all

REQUIREMENTS:
    pip install redis asyncio aiohttp
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

import redis

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_NAME = os.getenv("STREAM_NAME", "whatsapp_stream_inbound")
API_URL = os.getenv("API_URL", "http://localhost:8000")


@dataclass
class TestMetrics:
    """KPI Metrics collector."""
    test_name: str
    start_time: float = field(default_factory=time.time)

    # Pod Startup Latency
    scaling_events: List[Dict] = field(default_factory=list)
    startup_latencies: List[float] = field(default_factory=list)

    # Scaling thresholds
    scale_up_queue_depths: List[int] = field(default_factory=list)
    scale_down_queue_depths: List[int] = field(default_factory=list)

    # Memory tracking
    memory_samples: List[Dict] = field(default_factory=list)

    # Messages
    messages_sent: int = 0
    messages_processed: int = 0
    errors: int = 0

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "duration_seconds": time.time() - self.start_time,
            "scaling_events": len(self.scaling_events),
            "avg_startup_latency": sum(self.startup_latencies) / len(self.startup_latencies) if self.startup_latencies else 0,
            "max_startup_latency": max(self.startup_latencies) if self.startup_latencies else 0,
            "messages_sent": self.messages_sent,
            "messages_processed": self.messages_processed,
            "errors": self.errors,
            "memory_samples": len(self.memory_samples),
        }


def log(level: str, event: str, data: Optional[Dict] = None):
    """Structured JSON logging."""
    print(json.dumps({
        "ts": datetime.now().isoformat(),
        "level": level,
        "event": event,
        **(data or {})
    }), flush=True)


def get_worker_count() -> int:
    """Get current number of worker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "label=com.docker.compose.service=worker", "-q"],
            capture_output=True, text=True, timeout=10
        )
        containers = [l for l in result.stdout.strip().split('\n') if l]
        return len(containers)
    except Exception as e:
        log("error", "docker_count_failed", {"error": str(e)})
        return 0


def get_container_memory(container_pattern: str = "mobility_worker") -> Dict[str, int]:
    """Get memory usage of worker containers in MB."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
            capture_output=True, text=True, timeout=30
        )
        memory = {}
        for line in result.stdout.strip().split('\n'):
            if container_pattern in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    name = parts[0]
                    mem_str = parts[1].split('/')[0].strip()
                    # Parse memory (e.g., "256MiB" -> 256)
                    if 'GiB' in mem_str:
                        mem_mb = float(mem_str.replace('GiB', '')) * 1024
                    elif 'MiB' in mem_str:
                        mem_mb = float(mem_str.replace('MiB', ''))
                    elif 'KiB' in mem_str:
                        mem_mb = float(mem_str.replace('KiB', '')) / 1024
                    else:
                        mem_mb = 0
                    memory[name] = int(mem_mb)
        return memory
    except Exception as e:
        log("error", "memory_check_failed", {"error": str(e)})
        return {}


class LoadTestRunner:
    """Main test runner for autoscaler tests."""

    def __init__(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.metrics: Optional[TestMetrics] = None

    def push_messages(self, count: int, batch_size: int = 100) -> int:
        """Push test messages to Redis stream."""
        sent = 0
        for i in range(0, count, batch_size):
            batch = min(batch_size, count - i)
            pipe = self.redis.pipeline()
            for j in range(batch):
                msg_id = f"test_{int(time.time()*1000)}_{i+j}"
                pipe.xadd(STREAM_NAME, {
                    "type": "load_test",
                    "message_id": msg_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps({"test": True, "index": i+j})
                })
            try:
                pipe.execute()
                sent += batch
            except Exception as e:
                log("error", "push_failed", {"error": str(e), "batch": i})
                self.metrics.errors += batch if self.metrics else 0
        return sent

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        try:
            return self.redis.xlen(STREAM_NAME)
        except Exception:
            return 0

    def monitor_scaling(self, timeout: int = 300) -> Dict:
        """Monitor for scaling events."""
        start = time.time()
        initial_workers = get_worker_count()
        log("info", "monitoring_start", {"workers": initial_workers})

        last_count = initial_workers
        scale_detected_at = None

        while time.time() - start < timeout:
            current = get_worker_count()
            queue = self.get_queue_depth()

            if current != last_count:
                latency = time.time() - (scale_detected_at or start)
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "from_workers": last_count,
                    "to_workers": current,
                    "queue_depth": queue,
                    "latency_seconds": latency
                }

                if self.metrics:
                    self.metrics.scaling_events.append(event)
                    self.metrics.startup_latencies.append(latency)
                    if current > last_count:
                        self.metrics.scale_up_queue_depths.append(queue)
                    else:
                        self.metrics.scale_down_queue_depths.append(queue)

                log("info", "scaling_detected", event)
                last_count = current
                scale_detected_at = time.time()

            time.sleep(5)

        return {
            "initial_workers": initial_workers,
            "final_workers": get_worker_count(),
            "events": len(self.metrics.scaling_events) if self.metrics else 0
        }

    # ---
    # TEST 1: Gradual Ramp-up
    # ---
    def test_gradual_rampup(self,
                            initial_rate: int = 100,
                            increment: int = 100,
                            interval_seconds: int = 300,
                            max_rate: int = 1000):
        """
        Gradual Load Ramp-up Test

        Starts at initial_rate messages/second, increases by increment every
        interval_seconds until max_rate is reached or system fails.
        """
        self.metrics = TestMetrics(test_name="gradual_rampup")
        log("info", "test_start", {
            "test": "gradual_rampup",
            "initial_rate": initial_rate,
            "increment": increment,
            "interval": interval_seconds
        })

        current_rate = initial_rate

        try:
            while current_rate <= max_rate:
                log("info", "rampup_phase", {"rate": current_rate})

                # Send messages at current rate for interval duration
                phase_start = time.time()
                messages_this_phase = 0

                while time.time() - phase_start < interval_seconds:
                    # Send batch of messages
                    batch_size = min(current_rate, 500)
                    sent = self.push_messages(batch_size)
                    messages_this_phase += sent
                    self.metrics.messages_sent += sent

                    # Track memory
                    mem = get_container_memory()
                    if mem:
                        self.metrics.memory_samples.append({
                            "timestamp": time.time(),
                            "workers": mem,
                            "queue_depth": self.get_queue_depth()
                        })

                    # Sleep to maintain rate
                    elapsed = time.time() - phase_start
                    expected = messages_this_phase / current_rate
                    if expected > elapsed:
                        time.sleep(expected - elapsed)

                    # Check for system stress
                    queue = self.get_queue_depth()
                    workers = get_worker_count()
                    log("debug", "phase_status", {
                        "rate": current_rate,
                        "queue": queue,
                        "workers": workers,
                        "sent": messages_this_phase
                    })

                # Increment rate
                current_rate += increment
                log("info", "rate_increased", {"new_rate": current_rate})

        except KeyboardInterrupt:
            log("info", "test_interrupted")

        # Final report
        self._report_results()

    # ---
    # TEST 2: Burst/Spike Test
    # ---
    def test_burst_spike(self,
                         total_messages: int = 5000,
                         duration_seconds: int = 10):
        """
        Burst/Spike Test

        Sends total_messages in duration_seconds to test if autoscaler
        reacts quickly enough before system overloads.
        """
        self.metrics = TestMetrics(test_name="burst_spike")
        log("info", "test_start", {
            "test": "burst_spike",
            "messages": total_messages,
            "duration": duration_seconds
        })

        initial_workers = get_worker_count()
        initial_queue = self.get_queue_depth()

        log("info", "initial_state", {
            "workers": initial_workers,
            "queue": initial_queue
        })

        # Send burst
        burst_start = time.time()
        batch_size = total_messages // 10  # 10 batches

        for i in range(10):
            sent = self.push_messages(batch_size)
            self.metrics.messages_sent += sent

            # Small delay to spread over duration
            elapsed = time.time() - burst_start
            target = (i + 1) * (duration_seconds / 10)
            if target > elapsed:
                time.sleep(target - elapsed)

        burst_end = time.time()
        burst_duration = burst_end - burst_start

        log("info", "burst_complete", {
            "messages": self.metrics.messages_sent,
            "actual_duration": burst_duration,
            "rate": self.metrics.messages_sent / burst_duration
        })

        # Monitor scaling response
        log("info", "monitoring_scaling_response")

        max_queue = 0
        scale_detected = False
        scale_time = None

        for _ in range(60):  # Monitor for 60 seconds
            queue = self.get_queue_depth()
            workers = get_worker_count()
            max_queue = max(max_queue, queue)

            if workers > initial_workers and not scale_detected:
                scale_detected = True
                scale_time = time.time() - burst_end
                log("info", "scale_up_detected", {
                    "time_after_burst": scale_time,
                    "workers": workers,
                    "queue": queue
                })

            self.metrics.memory_samples.append({
                "timestamp": time.time(),
                "queue": queue,
                "workers": workers
            })

            time.sleep(1)

        # Results
        result = {
            "burst_duration": burst_duration,
            "messages_sent": self.metrics.messages_sent,
            "max_queue_depth": max_queue,
            "scale_detected": scale_detected,
            "scale_reaction_time": scale_time,
            "final_workers": get_worker_count()
        }

        # KPI Check: Reaction time < 30s is good
        if scale_time and scale_time < 30:
            log("info", "KPI_PASS", {"metric": "scale_reaction_time", "value": scale_time, "threshold": 30})
        elif scale_time:
            log("warning", "KPI_FAIL", {"metric": "scale_reaction_time", "value": scale_time, "threshold": 30})

        self._report_results()
        return result

    # ---
    # TEST 3: Cooldown Test
    # ---
    def test_cooldown(self,
                      warmup_messages: int = 2000,
                      cooldown_timeout: int = 600):
        """
        Cooldown Test

        Sends messages to trigger scale-up, then stops and monitors
        if instances scale down within expected time (5-10 min).
        """
        self.metrics = TestMetrics(test_name="cooldown")
        log("info", "test_start", {
            "test": "cooldown",
            "warmup_messages": warmup_messages,
            "timeout": cooldown_timeout
        })

        # Phase 1: Warmup - trigger scale up
        log("info", "warmup_phase")
        initial_workers = get_worker_count()

        sent = self.push_messages(warmup_messages)
        self.metrics.messages_sent = sent

        # Wait for scale up
        log("info", "waiting_for_scale_up")
        scale_up_time = None
        max_workers = initial_workers

        for i in range(120):  # Wait up to 2 minutes
            workers = get_worker_count()
            if workers > initial_workers:
                scale_up_time = i
                max_workers = workers
                log("info", "scale_up_detected", {
                    "workers": workers,
                    "time_seconds": i
                })
                break
            time.sleep(1)

        if not scale_up_time:
            log("warning", "no_scale_up_detected")
            return {"error": "Scale up not triggered"}

        # Wait for queue to drain
        log("info", "waiting_for_queue_drain")
        while self.get_queue_depth() > 0:
            time.sleep(5)
            log("debug", "queue_draining", {"depth": self.get_queue_depth()})

        # Phase 2: Cooldown - monitor scale down
        log("info", "cooldown_phase_started", {"workers": get_worker_count()})
        cooldown_start = time.time()
        scale_down_time = None

        while time.time() - cooldown_start < cooldown_timeout:
            workers = get_worker_count()
            elapsed = time.time() - cooldown_start

            self.metrics.memory_samples.append({
                "timestamp": time.time(),
                "workers": workers,
                "queue": self.get_queue_depth()
            })

            if workers < max_workers and not scale_down_time:
                scale_down_time = elapsed
                log("info", "scale_down_detected", {
                    "time_seconds": scale_down_time,
                    "workers": workers
                })

            if workers <= initial_workers:
                log("info", "cooldown_complete", {
                    "total_time": elapsed,
                    "final_workers": workers
                })
                break

            log("debug", "cooldown_status", {
                "elapsed": int(elapsed),
                "workers": workers
            })
            time.sleep(10)

        result = {
            "initial_workers": initial_workers,
            "max_workers": max_workers,
            "scale_up_time": scale_up_time,
            "scale_down_time": scale_down_time,
            "final_workers": get_worker_count()
        }

        # KPI Check: Scale down within 5-10 minutes
        if scale_down_time:
            if scale_down_time <= 300:
                log("info", "KPI_PASS", {"metric": "cooldown_time", "value": scale_down_time, "threshold": "300s (5min)"})
            elif scale_down_time <= 600:
                log("info", "KPI_OK", {"metric": "cooldown_time", "value": scale_down_time, "threshold": "600s (10min)"})
            else:
                log("warning", "KPI_FAIL", {"metric": "cooldown_time", "value": scale_down_time, "threshold": "600s (10min)"})

        self._report_results()
        return result

    # ---
    # TEST 4: Soak Test (Memory Leak Detection)
    # ---
    def test_soak(self,
                  duration_hours: float = 8.0,
                  rate: int = 50):
        """
        Soak Test - 8 hour stable load

        Maintains steady load and monitors for memory leaks.
        If RAM continuously grows, there's a leak.
        """
        duration_seconds = int(duration_hours * 3600)
        self.metrics = TestMetrics(test_name="soak")

        log("info", "test_start", {
            "test": "soak",
            "duration_hours": duration_hours,
            "rate": rate
        })

        start = time.time()
        sample_interval = 300  # Sample memory every 5 minutes
        last_sample = 0

        memory_baseline = get_container_memory()
        log("info", "memory_baseline", {"memory": memory_baseline})

        try:
            while time.time() - start < duration_seconds:
                # Send messages at steady rate
                sent = self.push_messages(rate)
                self.metrics.messages_sent += sent

                # Sample memory periodically
                if time.time() - last_sample > sample_interval:
                    mem = get_container_memory()
                    queue = self.get_queue_depth()
                    workers = get_worker_count()

                    sample = {
                        "timestamp": time.time(),
                        "elapsed_hours": (time.time() - start) / 3600,
                        "memory": mem,
                        "queue": queue,
                        "workers": workers
                    }
                    self.metrics.memory_samples.append(sample)

                    log("info", "soak_sample", sample)
                    last_sample = time.time()

                    # Check for memory growth
                    if len(self.metrics.memory_samples) > 2:
                        self._check_memory_trend()

                time.sleep(1)

        except KeyboardInterrupt:
            log("info", "soak_interrupted")

        self._report_results()
        self._analyze_memory_leak()

    def _check_memory_trend(self):
        """Check if memory is consistently growing."""
        samples = self.metrics.memory_samples[-6:]  # Last 30 minutes
        if len(samples) < 3:
            return

        # Calculate memory trend
        total_growth = 0
        for container in samples[0].get("memory", {}):
            first = samples[0]["memory"].get(container, 0)
            last = samples[-1]["memory"].get(container, 0)
            if first > 0:
                growth = ((last - first) / first) * 100
                total_growth += growth

        if total_growth > 20:  # >20% growth is concerning
            log("warning", "memory_growth_detected", {"growth_percent": total_growth})

    def _analyze_memory_leak(self):
        """Final memory leak analysis."""
        if len(self.metrics.memory_samples) < 3:
            return

        first = self.metrics.memory_samples[0]
        last = self.metrics.memory_samples[-1]

        log("info", "memory_analysis", {
            "start": first.get("memory", {}),
            "end": last.get("memory", {}),
            "duration_hours": last["elapsed_hours"]
        })

    def _report_results(self):
        """Generate final test report."""
        if not self.metrics:
            return

        report = self.metrics.to_dict()

        print("\n" + "=" * 60)
        print(f"TEST REPORT: {self.metrics.test_name}")
        print("=" * 60)
        print(json.dumps(report, indent=2))
        print("=" * 60)

        # Save to file
        report_file = f"test_report_{self.metrics.test_name}_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "report": report,
                "scaling_events": self.metrics.scaling_events,
                "memory_samples": self.metrics.memory_samples
            }, f, indent=2)
        log("info", "report_saved", {"file": report_file})

    def cleanup(self):
        """Clean up test messages from stream."""
        try:
            # Trim stream to remove test messages
            self.redis.xtrim(STREAM_NAME, maxlen=0, approximate=False)
            log("info", "cleanup_complete")
        except Exception as e:
            log("error", "cleanup_failed", {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Autoscaler Load & Stress Test")
    parser.add_argument("--test", choices=["gradual", "burst", "cooldown", "soak", "all"],
                        default="all", help="Test type to run")
    parser.add_argument("--duration", type=int, default=28800,
                        help="Soak test duration in seconds (default: 8 hours)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up test messages after test")

    args = parser.parse_args()

    runner = LoadTestRunner()

    try:
        if args.test == "gradual" or args.test == "all":
            runner.test_gradual_rampup()

        if args.test == "burst" or args.test == "all":
            runner.test_burst_spike()

        if args.test == "cooldown" or args.test == "all":
            runner.test_cooldown()

        if args.test == "soak":
            runner.test_soak(duration_hours=args.duration / 3600)

    finally:
        if args.cleanup:
            runner.cleanup()


if __name__ == "__main__":
    main()
