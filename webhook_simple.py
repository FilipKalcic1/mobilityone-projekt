"""
Simple webhook endpoint for WhatsApp messages (Infobip).
Receives messages and pushes to Redis queue for worker processing.

- Handles ALL known Infobip payload formats
- Case-insensitive type matching
- Non-text message forwarding (image, location, voice → user gets response)
- Webhook-level DLQ for Redis failures
- Diagnostic endpoint for remote debugging
- Request ID tracking for log correlation

Security:
- Webhook signature validation (HMAC-SHA256) when VERIFY_WHATSAPP_SIGNATURE=True
- Verify token from environment variable WHATSAPP_VERIFY_TOKEN
"""

import asyncio
import hmac
import hashlib
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException
import redis.asyncio as aioredis
import logging

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ---
# DIAGNOSTIC RING BUFFER - Last N webhook events for remote debugging
# ---
_MAX_DIAG_ENTRIES = 50
_diag_buffer: deque = deque(maxlen=_MAX_DIAG_ENTRIES)
_stats = {
    "total_received": 0,
    "total_pushed": 0,
    "total_no_text": 0,
    "total_no_sender": 0,
    "total_no_results": 0,
    "total_redis_errors": 0,
    "total_parse_errors": 0,
    "last_success_at": None,
    "last_error_at": None,
    "last_error": None,
    "started_at": datetime.now(timezone.utc).isoformat(),
}


def _diag_log(event: str, data: dict = None):
    """Log to diagnostic ring buffer for remote debugging."""
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "event": event,
        **(data or {})
    }
    _diag_buffer.append(entry)


# ---
# SIGNATURE VALIDATION
# ---

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify Infobip webhook signature using HMAC-SHA256.

    Args:
        payload: Raw request body bytes
        signature: Signature from X-Hub-Signature-256 header
        secret: INFOBIP_SECRET_KEY from environment

    Returns:
        True if signature is valid, False otherwise
    """
    if not secret:
        logger.warning("INFOBIP_SECRET_KEY not configured, skipping signature validation")
        return True

    if not signature:
        logger.warning("No signature header in webhook request")
        return False

    # Infobip uses sha256=<hex_digest> format
    if signature.startswith("sha256="):
        signature = signature[7:]

    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected.lower(), signature.lower())


# ---
# TEXT EXTRACTION - handles ALL known Infobip formats
# ---

def extract_text_and_type(result: dict) -> tuple:
    """
    Extract text and message type from Infobip webhook result.

    Handles ALL known formats:
    1. {"message": {"type": "TEXT", "text": "..."}}          - Standard Infobip
    2. {"message": {"type": "text", "text": "..."}}          - Lowercase variant
    3. {"content": [{"type": "TEXT", "text": "..."}]}        - Content as list
    4. {"content": {"type": "TEXT", "text": "..."}}          - Content as dict
    5. {"text": "..."}                                        - Direct text field
    6. {"message": {"text": "..."}}                           - Message without type
    7. {"body": "..."}                                        - Body field variant

    Returns:
        (text: str, msg_type: str) - text is empty string if no text found
    """
    text = ""
    msg_type = "UNKNOWN"

    # Format 1 & 2: message object (most common Infobip format)
    message_obj = result.get("message")
    if message_obj and isinstance(message_obj, dict):
        msg_type = message_obj.get("type", "UNKNOWN")

        # Case-insensitive type check
        if msg_type.upper() == "TEXT":
            text = message_obj.get("text", "")
            if text:
                return text.strip(), "TEXT"

        # Format 6: message object without type field but with text
        if not text and "text" in message_obj:
            text = message_obj.get("text", "")
            if text:
                return text.strip(), msg_type or "TEXT"

        # Non-text message - return type for handling
        return "", msg_type

    # Format 3 & 4: content field
    content = result.get("content")
    if content:
        if isinstance(content, dict):
            content_type = content.get("type", "")
            if content_type.upper() == "TEXT":
                text = content.get("text", "")
                if text:
                    return text.strip(), "TEXT"
            return "", content_type or "UNKNOWN"

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type.upper() == "TEXT":
                        text = item.get("text", "")
                        if text:
                            return text.strip(), "TEXT"
            # Return type of first item if no text found
            if content and isinstance(content[0], dict):
                return "", content[0].get("type", "UNKNOWN")

    # Format 5: direct text field on result
    direct_text = result.get("text")
    if direct_text and isinstance(direct_text, str):
        return direct_text.strip(), "TEXT"

    # Format 7: body field
    body = result.get("body")
    if body and isinstance(body, str):
        return body.strip(), "TEXT"

    return "", msg_type


# ---
# REDIS CLIENT
# ---

router = APIRouter()

_redis_client = None
_redis_lock = asyncio.Lock()


async def get_redis():
    """Get async Redis client (thread-safe lazy initialization with pool config)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        _redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_keepalive=True,
            health_check_interval=30
        )
        return _redis_client


# ---
# MAIN WEBHOOK HANDLER
# ---

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Receive WhatsApp webhook messages and push to Redis STREAM.

    Flow:
    1. Validate webhook signature (if enabled)
    2. Parse JSON body
    3. Extract message data (sender, text, message_id) from ALL formats
    4. VALIDATE sender is present
    5. Handle non-text messages (forward type info to Redis)
    6. Push to Redis STREAM: "whatsapp_stream_inbound"
    7. Worker picks up from stream via consumer group
    """
    _stats["total_received"] += 1

    try:
        # Get raw body for signature validation
        raw_body = await request.body()

        # Validate signature if enabled
        if settings.VERIFY_WHATSAPP_SIGNATURE:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not verify_webhook_signature(raw_body, signature, settings.INFOBIP_SECRET_KEY):
                logger.warning(
                    f"Invalid webhook signature from {request.client.host}"
                )
                _diag_log("signature_failed", {"ip": request.client.host})
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse JSON body
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError) as e:
            _stats["total_parse_errors"] += 1
            logger.error(f"Invalid JSON in webhook body: {e}")
            _diag_log("json_parse_error", {"error": str(e), "body_preview": raw_body[:200].decode(errors='replace')})
            return {"status": "ok", "error": "invalid_json"}

        logger.info(f"Received WhatsApp webhook: {json.dumps(body, ensure_ascii=False)[:500]}")
        _diag_log("received", {"keys": list(body.keys()), "result_count": len(body.get("results", []))})

        # Extract message details from Infobip format
        results = body.get("results", [])
        if not results:
            _stats["total_no_results"] += 1
            logger.warning(f"No results in webhook body. Keys: {list(body.keys())}")
            _diag_log("no_results", {"body_keys": list(body.keys())})
            return {"status": "ok", "note": "no_results"}

        pushed = 0
        for result in results:
            # Infobip uses "from", legacy/test may use "sender"
            sender = result.get("from") or result.get("sender", "")
            message_id = result.get("messageId", "")

            # CRITICAL: Validate sender is present
            if not sender:
                _stats["total_no_sender"] += 1
                logger.error(
                    "MISSING SENDER in webhook! "
                    f"message_id={message_id}, keys={list(result.keys())}"
                )
                _diag_log("no_sender", {"keys": list(result.keys()), "message_id": message_id})
                continue

            # Extract text using robust multi-format parser
            text, msg_type = extract_text_and_type(result)

            if not text:
                _stats["total_no_text"] += 1
                logger.info(
                    f"Non-text message from {sender[-4:]}...: type={msg_type}"
                )
                _diag_log("non_text", {"sender": sender[-4:], "type": msg_type})

                # Forward non-text messages to Redis so worker can respond
                # "We only support text messages" is better than silence
                stream_data = {
                    "sender": sender,
                    "text": f"[NON_TEXT:{msg_type}]",
                    "message_id": message_id,
                    "original_type": msg_type
                }

                try:
                    redis = await get_redis()
                    await redis.xadd("whatsapp_stream_inbound", stream_data)
                    pushed += 1
                    logger.info(f"Non-text message forwarded: {sender[-4:]}... type={msg_type}")
                except Exception as redis_err:
                    _stats["total_redis_errors"] += 1
                    logger.error(f"Redis push failed for non-text: {redis_err}")
                    _diag_log("redis_error", {"error": str(redis_err)})
                continue

            # Push text message to Redis STREAM
            stream_data = {
                "sender": sender,
                "text": text,
                "message_id": message_id
            }

            try:
                redis = await get_redis()
                await redis.xadd("whatsapp_stream_inbound", stream_data)
                pushed += 1
                _stats["total_pushed"] += 1
                _stats["last_success_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Message pushed to stream: {sender[-4:]}... - {text[:50]}")
                _diag_log("pushed", {"sender": sender[-4:], "text_preview": text[:30]})

            except Exception as redis_err:
                # WEBHOOK-LEVEL DLQ: Redis failed, log full payload for recovery
                _stats["total_redis_errors"] += 1
                _stats["last_error_at"] = datetime.now(timezone.utc).isoformat()
                _stats["last_error"] = f"Redis push failed: {redis_err}"

                logger.error(
                    f"REDIS PUSH FAILED - MESSAGE WILL BE LOST! "
                    f"sender={sender}, text={text[:100]}, message_id={message_id}, "
                    f"error={redis_err}",
                    exc_info=True
                )
                _diag_log("redis_push_failed", {
                    "sender": sender[-4:],
                    "message_id": message_id,
                    "error": str(redis_err)
                })

                # DLQ: Write to stderr in structured format for log aggregation recovery
                dlq_entry = json.dumps({
                    "dlq": "webhook",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "sender": sender,
                    "text": text,
                    "message_id": message_id,
                    "error": str(redis_err)
                })
                sys.stderr.write(f"DLQ_WEBHOOK: {dlq_entry}\n")
                sys.stderr.flush()

        return {"status": "ok", "pushed": pushed}

    except HTTPException:
        raise
    except Exception as e:
        # CRITICAL: Always return 200 to WhatsApp/Infobip!
        # Returning 500 causes retry storms that cascade into duplicate messages.
        _stats["last_error_at"] = datetime.now(timezone.utc).isoformat()
        _stats["last_error"] = str(e)
        logger.error(f"Webhook processing error (returning 200 to prevent retries): {e}", exc_info=True)
        _diag_log("exception", {"error": str(e)})
        return {"status": "ok", "error": "processing_failed"}


# ---
# VERIFICATION & HEALTH CHECK
# ---

@router.get("/whatsapp")
async def whatsapp_webhook_verify(request: Request):
    """
    Webhook verification / health check endpoint.

    - Simple GET (no params): returns 200 OK (for Infobip URL validation)
    - With hub.mode params: Meta-style verification (if WHATSAPP_VERIFY_TOKEN set)
    """
    mode = request.query_params.get("hub.mode")

    # No hub.mode param = simple health check (Infobip just pings the URL)
    if not mode:
        return {"status": "ok", "webhook": "active"}

    # Meta-style verification (hub.mode + hub.verify_token + hub.challenge)
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    expected_token = settings.WHATSAPP_VERIFY_TOKEN

    if not expected_token:
        logger.warning("WHATSAPP_VERIFY_TOKEN not configured, skipping verification")
        return {"status": "ok"}

    if mode == "subscribe" and token == expected_token:
        logger.info("WhatsApp webhook verified successfully")
        return int(challenge)

    logger.warning(f"Webhook verification failed: mode={mode}")
    raise HTTPException(status_code=403, detail="Verification failed")


# ---
# DIAGNOSTIC ENDPOINT - for remote debugging without SSH
# ---

@router.get("/whatsapp/debug")
async def webhook_debug(request: Request):
    """
    Diagnostic endpoint for remote debugging.

    SECURED: Requires ?token=ADMIN_TOKEN_1 query parameter.
    Returns 404 if token is missing or invalid (404 instead of 401
    to avoid revealing that the endpoint exists).

    Shows:
    - Webhook statistics (received, pushed, errors)
    - Last N events from ring buffer
    - Redis connection status
    - Stream info (pending messages, consumer groups)
    """
    # Auth: require admin token via query param
    # Reads ADMIN_TOKEN_1..4 from env (same as admin_api.py)
    token = request.query_params.get("token", "")
    expected_tokens = set()
    for i in range(1, 5):
        env_token = os.environ.get(f"ADMIN_TOKEN_{i}")
        if env_token:
            expected_tokens.add(env_token)

    if not expected_tokens or token not in expected_tokens:
        raise HTTPException(status_code=404, detail="Not found")

    diag = {
        "stats": dict(_stats),
        "recent_events": list(_diag_buffer),
        "redis": {"status": "unknown"},
        "stream": {},
        "config": {
            "verify_signature": settings.VERIFY_WHATSAPP_SIGNATURE,
            "has_secret_key": bool(settings.INFOBIP_SECRET_KEY),
            "has_api_key": bool(settings.INFOBIP_API_KEY),
            "sender_number": settings.INFOBIP_SENDER_NUMBER,
            "redis_url_masked": settings.REDIS_URL.split("@")[-1] if "@" in settings.REDIS_URL else settings.REDIS_URL,
        }
    }

    # Check Redis connection
    try:
        redis = await get_redis()
        await redis.ping()
        diag["redis"]["status"] = "connected"

        # Get stream info
        try:
            stream_info = await redis.xinfo_stream("whatsapp_stream_inbound")
            diag["stream"] = {
                "length": stream_info.get("length", 0),
                "first_entry": stream_info.get("first-entry"),
                "last_entry": stream_info.get("last-entry"),
            }
        except Exception as e:
            diag["stream"] = {"error": str(e)}

        # Get consumer group info
        try:
            groups = await redis.xinfo_groups("whatsapp_stream_inbound")
            diag["consumer_groups"] = [
                {
                    "name": g.get("name"),
                    "consumers": g.get("consumers"),
                    "pending": g.get("pending"),
                    "last_delivered": g.get("last-delivered-id"),
                }
                for g in groups
            ]
        except Exception as e:
            diag["consumer_groups"] = {"error": str(e)}

    except Exception as e:
        diag["redis"]["status"] = f"error: {e}"

    return diag
