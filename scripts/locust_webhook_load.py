"""
Locust Load Test for MobilityOne WhatsApp Webhook

Simulates Infobip webhook traffic to find the throttle point at 0.5 CPU.

Strategy:
  - Ramp from 1 to 50 users over 2 minutes
  - Each user sends 1 webhook/second (realistic Infobip rate)
  - Monitor until CPU hits 0.45 (90% of 0.5 limit)
  - Record: p99 latency, error rate, queue depth at throttle point

Usage:
    pip install locust
    locust -f scripts/locust_webhook_load.py --host http://localhost:8000

    # Headless mode (CI):
    locust -f scripts/locust_webhook_load.py \
        --host http://localhost:8000 \
        --headless -u 50 -r 5 --run-time 3m \
        --csv results/load_test
"""

import json
import random
import string
import time
from locust import HttpUser, task, between


def _random_phone() -> str:
    """Generate a random Croatian phone number."""
    return f"38599{''.join(random.choices(string.digits, k=7))}"


def _random_message() -> str:
    """Pick a realistic user message."""
    messages = [
        "kolika mi je kilometraža",
        "kad mi ističe registracija",
        "trebam rezervirati vozilo",
        "prijavi štetu na vozilu",
        "koji su mi troškovi",
        "daj mi info o vozilu",
        "bok",
        "hvala",
        "trebam pomoć",
        "koliko km ima moj auto",
        "kad je sljedeći servis",
        "pošalji mi podatke o leasingu",
    ]
    return random.choice(messages)


class InfobipWebhookUser(HttpUser):
    """Simulates Infobip sending WhatsApp webhook payloads."""

    wait_time = between(0.8, 1.2)  # ~1 req/sec per user

    @task
    def send_webhook(self):
        """POST a realistic Infobip webhook payload."""
        sender = _random_phone()
        message_id = f"ABGGFlA5Fpa-{random.randint(100000, 999999)}"

        payload = {
            "results": [
                {
                    "from": sender,
                    "to": "385991234567",
                    "integrationType": "WHATSAPP",
                    "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
                    "messageId": message_id,
                    "message": {
                        "type": "TEXT",
                        "text": _random_message(),
                    },
                    "contact": {"name": "Test User"},
                    "price": {"pricePerMessage": 0, "currency": "EUR"},
                }
            ],
            "messageCount": 1,
            "pendingMessageCount": 0,
        }

        with self.client.post(
            "/webhook/whatsapp",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
