"""
KPI Metrics Exporter for MobilityOne Autoscaler

METRICS EXPOSED:
1. pod_startup_latency_seconds - Time for new worker to become active
2. hpa_scaling_threshold_queue_depth - Queue depth when scaling triggered
3. worker_memory_usage_mb - Memory usage per worker
4. autoscaler_scale_events_total - Total scale up/down events
5. queue_depth_current - Current Redis stream queue depth
6. worker_count_current - Current number of workers

USAGE:
    # Run as standalone metrics server
    python scripts/kpi_metrics.py

    # Or import and use in autoscaler
    from scripts.kpi_metrics import MetricsCollector

GRAFANA DASHBOARD:
    Import the JSON from scripts/grafana_dashboard.json

PROMETHEUS CONFIG:
    Add to prometheus.yml:
    - job_name: 'autoscaler-metrics'
      static_configs:
        - targets: ['autoscaler:9100']
"""

import os
import time
import threading
import subprocess
from typing import Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field

try:
    from prometheus_client import (
        Gauge, Counter, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Run: pip install prometheus-client")


# ---
# METRICS DEFINITIONS
# ---

if PROMETHEUS_AVAILABLE:
    # Create custom registry
    REGISTRY = CollectorRegistry()

    # 1. Pod Startup Latency (KPI: <30s is good)
    POD_STARTUP_LATENCY = Histogram(
        'pod_startup_latency_seconds',
        'Time for new worker pod to become active',
        buckets=[5, 10, 15, 20, 25, 30, 45, 60, 90, 120],
        registry=REGISTRY
    )

    # 2. HPA Scaling Threshold
    SCALING_THRESHOLD_QUEUE = Gauge(
        'hpa_scaling_threshold_queue_depth',
        'Queue depth when scaling was triggered',
        ['direction'],  # up or down
        registry=REGISTRY
    )

    # 3. Worker Memory Usage
    WORKER_MEMORY_MB = Gauge(
        'worker_memory_usage_mb',
        'Memory usage per worker container in MB',
        ['worker_id'],
        registry=REGISTRY
    )

    # 4. Scale Events Counter
    SCALE_EVENTS = Counter(
        'autoscaler_scale_events_total',
        'Total number of scaling events',
        ['direction'],  # up or down
        registry=REGISTRY
    )

    # 5. Current Queue Depth
    QUEUE_DEPTH = Gauge(
        'queue_depth_current',
        'Current Redis stream queue depth',
        registry=REGISTRY
    )

    # 6. Current Worker Count
    WORKER_COUNT = Gauge(
        'worker_count_current',
        'Current number of active workers',
        registry=REGISTRY
    )

    # 7. Memory Leak Indicator (Soak Test)
    MEMORY_TREND = Gauge(
        'memory_trend_percent',
        'Memory growth percentage over last hour',
        registry=REGISTRY
    )

    # 8. Scaling Reaction Time
    SCALING_REACTION_TIME = Summary(
        'scaling_reaction_time_seconds',
        'Time between threshold breach and scale action',
        registry=REGISTRY
    )

    # 9. CPU Usage per Worker
    WORKER_CPU_PERCENT = Gauge(
        'worker_cpu_usage_percent',
        'CPU usage percentage per worker',
        ['worker_id'],
        registry=REGISTRY
    )


@dataclass
class MetricsCollector:
    """Collect and expose metrics for KPI tracking."""

    # Tracking for startup latency
    scale_requested_at: Optional[float] = None
    previous_worker_count: int = 0

    # Memory tracking for leak detection
    memory_history: list = field(default_factory=list)

    def record_scale_event(self, direction: str, queue_depth: int):
        """Record a scaling event."""
        if not PROMETHEUS_AVAILABLE:
            return

        SCALE_EVENTS.labels(direction=direction).inc()
        SCALING_THRESHOLD_QUEUE.labels(direction=direction).set(queue_depth)
        self.scale_requested_at = time.time()

    def record_startup_complete(self, new_worker_count: int):
        """Record when new workers are ready."""
        if not PROMETHEUS_AVAILABLE or not self.scale_requested_at:
            return

        if new_worker_count > self.previous_worker_count:
            latency = time.time() - self.scale_requested_at
            POD_STARTUP_LATENCY.observe(latency)
            SCALING_REACTION_TIME.observe(latency)

        self.previous_worker_count = new_worker_count
        self.scale_requested_at = None

    def update_queue_depth(self, depth: int):
        """Update current queue depth."""
        if PROMETHEUS_AVAILABLE:
            QUEUE_DEPTH.set(depth)

    def update_worker_count(self, count: int):
        """Update current worker count."""
        if PROMETHEUS_AVAILABLE:
            WORKER_COUNT.set(count)

    def update_worker_stats(self, stats: Dict[str, Dict]):
        """
        Update per-worker stats.

        stats format: {
            "worker_1": {"memory_mb": 256, "cpu_percent": 45},
            "worker_2": {"memory_mb": 280, "cpu_percent": 52}
        }
        """
        if not PROMETHEUS_AVAILABLE:
            return

        for worker_id, data in stats.items():
            WORKER_MEMORY_MB.labels(worker_id=worker_id).set(data.get("memory_mb", 0))
            WORKER_CPU_PERCENT.labels(worker_id=worker_id).set(data.get("cpu_percent", 0))

        # Track memory for leak detection
        total_mem = sum(d.get("memory_mb", 0) for d in stats.values())
        self.memory_history.append({
            "timestamp": time.time(),
            "total_mb": total_mem
        })

        # Keep last hour of data
        one_hour_ago = time.time() - 3600
        self.memory_history = [m for m in self.memory_history if m["timestamp"] > one_hour_ago]

        # Calculate trend
        if len(self.memory_history) >= 2:
            first = self.memory_history[0]["total_mb"]
            last = self.memory_history[-1]["total_mb"]
            if first > 0:
                trend = ((last - first) / first) * 100
                MEMORY_TREND.set(trend)


def get_container_stats() -> Dict[str, Dict]:
    """Get memory and CPU stats for worker containers."""
    stats = {}
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"],
            capture_output=True, text=True, timeout=30
        )

        for line in result.stdout.strip().split('\n'):
            if 'worker' in line.lower():
                parts = line.split('\t')
                if len(parts) >= 3:
                    name = parts[0]

                    # Parse memory
                    mem_str = parts[1].split('/')[0].strip()
                    if 'GiB' in mem_str:
                        mem_mb = float(mem_str.replace('GiB', '')) * 1024
                    elif 'MiB' in mem_str:
                        mem_mb = float(mem_str.replace('MiB', ''))
                    else:
                        mem_mb = 0

                    # Parse CPU
                    cpu_str = parts[2].replace('%', '').strip()
                    try:
                        cpu = float(cpu_str)
                    except ValueError:
                        cpu = 0

                    stats[name] = {"memory_mb": mem_mb, "cpu_percent": cpu}

    except Exception as e:
        print(f"Error getting container stats: {e}")

    return stats


# ---
# HTTP SERVER FOR PROMETHEUS SCRAPING
# ---

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    collector: Optional[MetricsCollector] = None

    def do_GET(self):
        if self.path == '/metrics':
            # Update stats before responding
            if self.collector:
                stats = get_container_stats()
                self.collector.update_worker_stats(stats)

            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


def start_metrics_server(port: int = 9100, collector: Optional[MetricsCollector] = None):
    """Start the metrics HTTP server."""
    MetricsHandler.collector = collector
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Metrics server started on port {port}")
    print(f"Prometheus scrape URL: http://localhost:{port}/metrics")

    return server


# ---
# GRAFANA DASHBOARD CONFIG
# ---

GRAFANA_DASHBOARD = {
    "title": "MobilityOne Autoscaler KPIs",
    "panels": [
        {
            "title": "Pod Startup Latency (Target: <30s)",
            "type": "gauge",
            "targets": [{
                "expr": "histogram_quantile(0.95, pod_startup_latency_seconds_bucket)"
            }],
            "thresholds": {"steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 20},
                {"color": "red", "value": 30}
            ]}
        },
        {
            "title": "Worker Count",
            "type": "stat",
            "targets": [{"expr": "worker_count_current"}]
        },
        {
            "title": "Queue Depth",
            "type": "timeseries",
            "targets": [{"expr": "queue_depth_current"}]
        },
        {
            "title": "Memory Usage per Worker",
            "type": "timeseries",
            "targets": [{"expr": "worker_memory_usage_mb"}]
        },
        {
            "title": "Memory Trend (Leak Detection)",
            "type": "gauge",
            "targets": [{"expr": "memory_trend_percent"}],
            "thresholds": {"steps": [
                {"color": "green", "value": -10},
                {"color": "yellow", "value": 10},
                {"color": "red", "value": 20}
            ]}
        },
        {
            "title": "Scale Events",
            "type": "timeseries",
            "targets": [
                {"expr": "rate(autoscaler_scale_events_total[5m])"}
            ]
        }
    ]
}


# ---
# MAIN
# ---

if __name__ == "__main__":
    import json

    if not PROMETHEUS_AVAILABLE:
        print("Please install prometheus-client: pip install prometheus-client")
        exit(1)

    print("=" * 60)
    print("KPI METRICS SERVER")
    print("=" * 60)
    print("\nMetrics exposed:")
    print("  - pod_startup_latency_seconds")
    print("  - hpa_scaling_threshold_queue_depth")
    print("  - worker_memory_usage_mb")
    print("  - autoscaler_scale_events_total")
    print("  - queue_depth_current")
    print("  - worker_count_current")
    print("  - memory_trend_percent")
    print("\nKPI Thresholds (2026 Standards):")
    print("  - Pod Startup: <30s")
    print("  - Memory Trend: <20% growth/hour")
    print("=" * 60)

    # Save Grafana dashboard
    dashboard_path = "scripts/grafana_autoscaler_dashboard.json"
    with open(dashboard_path, 'w') as f:
        json.dump(GRAFANA_DASHBOARD, f, indent=2)
    print(f"\nGrafana dashboard saved to: {dashboard_path}")

    collector = MetricsCollector()
    server = start_metrics_server(port=9100, collector=collector)

    # Keep running
    try:
        while True:
            time.sleep(10)
            stats = get_container_stats()
            collector.update_worker_stats(stats)
    except KeyboardInterrupt:
        print("\nShutting down...")
