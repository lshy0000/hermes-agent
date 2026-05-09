"""
Hermes Clawith Platform Adapter

Drop-in adapter for the Hermes gateway to communicate with Clawith
via the Longlink WebSocket protocol.

Installation:
    1. Copy this file to ~/.hermes/hermes-agent/gateway/platforms/clawith.py
    2. Add CLAWITH = "clawith" to Platform enum in gateway/config.py
    3. Add the platform handler in gateway/run.py (see patches/)
    4. pip install websockets httpx
    5. Configure in ~/.hermes/config.yaml and restart gateway

Configuration in config.yaml:
    platforms:
      clawith:
        enabled: true
        extra:
          host: "172.17.144.1"
          api_key: "oc-xxx-xxx"
          longlink_port: 38438      # optional, default 38438
          directory_port: 3008     # optional, default 3008
          user_id: "agent0323"     # optional, default agent0323
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import websockets
    import websockets.client
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# -- Protocol constants -----------------------------------------------------

LONGLINK_HEARTBEAT_INTERVAL_S = 10       # Send heartbeat every 10s
LONGLINK_RECONNECT_BACKOFF = [30, 120, 3600]
DEFAULT_LONGLINK_PORT = 38438
DEFAULT_DIRECTORY_PORT = 3008
DEFAULT_USER_ID = "agent0323"
MAX_MESSAGE_LENGTH = 20000


# ---------------------------------------------------------------------------

def check_clawith_requirements() -> bool:
    """Check if Clawith dependencies are available and configured."""
    if not WEBSOCKETS_AVAILABLE or not HTTPX_AVAILABLE:
        return False
    return True


class ClawithAdapter(BasePlatformAdapter):
    """Clawith chatbot adapter using Longlink (WebSocket).

    Maintains a persistent WebSocket connection to Clawith's longlink endpoint.
    Incoming ``gateway.task`` events are dispatched to the Hermes message handler.
    Replies are sent via ``clawith.user_dm`` frames followed by ``report`` frames.

    Protocol summary:
        - Connect: ws://<host>:<port>/ws?apiKey=<key>&userId=agent0323
        - Heartbeat: send {"type":"heartbeat"} every 10s
        - Ping/Pong: reply to "ping" with "pong"
        - Inbound: {"type":"event","payload":{"source":"gateway.task",...}}
        - Outbound: {"type":"clawith.user_dm","target_user_id":"...","content":"..."}
                   {"type":"report","message_id":"...","result":"..."}
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.CLAWITH)

        extra = config.extra or {}
        raw_host = extra.get("host", "") or ""
        # Normalize host: strip http:// or ws:// prefix
        self._host: str = raw_host
        for prefix in ("http://", "https://", "ws://", "wss://"):
            if self._host.startswith(prefix):
                self._host = self._host[len(prefix):]
                break

        self._api_key: str = extra.get("api_key", "") or ""
        self._longlink_port: int = int(extra.get("longlink_port", DEFAULT_LONGLINK_PORT))
        self._directory_port: int = int(extra.get("directory_port", DEFAULT_DIRECTORY_PORT))
        self._user_id: str = extra.get("user_id", DEFAULT_USER_ID)

        self._ws = None
        self._ws_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Message deduplication
        self._dedup = MessageDeduplicator(max_size=1000)

        # Map conversation_id -> target_user_id for reply routing
        self._conversations: Dict[str, str] = {}
        # Map user_id -> conversation_id
        self._user_conversations: Dict[str, str] = {}
        # Map event_id (task id) -> user_id for report routing
        self._pending_tasks: Dict[str, str] = {}

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Connect to Clawith Longlink WebSocket."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("[%s] websockets not installed. Run: pip install websockets", self.name)
            return False
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._host or not self._api_key:
            logger.warning("[%s] host and api_key required in config", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(timeout=30.0)
            self._ws_task = asyncio.create_task(self._run_longlink())
            self._mark_connected()
            logger.info("[%s] Connecting to Clawith Longlink ws://%s:%d/ws",
                        self.name, self._host, self._longlink_port)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _run_longlink(self) -> None:
        """Run the WebSocket client with auto-reconnection."""
        backoff_idx = 0
        logger.info("[%s] _run_longlink loop STARTING, _running=%s", self.name, self._running)
        while self._running:
            ws_url = self._build_ws_url()
            logger.info("[%s] Connecting to %s", self.name, self._scrub_url(ws_url))
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    backoff_idx = 0  # Reset backoff on successful connect
                    logger.info("[%s] WebSocket connected", self.name)
                    # Start heartbeat
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
                    # Process messages
                    await self._process_messages(ws)
                    logger.warning("[%s] WebSocket connection closed", self.name)
            except asyncio.CancelledError:
                logger.info("[%s] _run_longlink CancelledError, exiting", self.name)
                return
            except websockets.ConnectionClosed as e:
                logger.warning("[%s] Connection closed: code=%s reason=%s",
                               self.name, e.code, e.reason)
            except Exception as e:
                if not self._running:
                    logger.info("[%s] _run_longlink exception but _running=False, exiting",
                                self.name)
                    return
                logger.warning("[%s] Connection error: %s (type=%s)",
                               self.name, e, type(e).__name__)

            # Cleanup
            self._ws = None
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            if not self._running:
                logger.info("[%s] _run_longlink: _running=False, exiting", self.name)
                return

            # Reconnect with backoff
            delay = LONGLINK_RECONNECT_BACKOFF[
                min(backoff_idx, len(LONGLINK_RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1
        logger.info("[%s] _run_longlink loop EXITED, _running=%s",
                     self.name, self._running)

    async def _process_messages(self, ws) -> None:
        """Process incoming WebSocket messages."""
        async for raw_message in ws:
            try:
                await self._handle_raw_message(raw_message)
            except Exception:
                logger.exception("[%s] Error processing message", self.name)

    async def _heartbeat_loop(self, ws) -> None:
        """Send periodic heartbeat frames."""
        try:
            while self._running:
                await asyncio.sleep(LONGLINK_HEARTBEAT_INTERVAL_S)
                if self._ws is not ws:
                    break
                # websockets 16: ws.state is a State enum; websockets < 14: ws.closed
                try:
                    from websockets import State
                    is_closed = ws.state == State.CLOSED
                except (ImportError, AttributeError):
                    is_closed = getattr(ws, 'closed', True)
                if is_closed:
                    break
                await ws.send(json.dumps({
                    "type": "heartbeat",
                    "id": str(uuid.uuid4()),
                    "ts": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
                }))
                logger.debug("[%s] Heartbeat sent", self.name)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.warning("[%s] Heartbeat loop error: %s", self.name, e)

    async def disconnect(self) -> None:
        """Disconnect from Clawith."""
        self._running = False
        self._mark_disconnected()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._ws = None
        self._conversations.clear()
        self._user_conversations.clear()
        self._pending_tasks.clear()
        self._dedup.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound message processing -----------------------------------------

    async def _handle_raw_message(self, raw: str) -> None:
        """Parse and dispatch an incoming WebSocket message."""
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[%s] Received non-JSON message: %s", self.name, raw[:200])
            return

        frame_type = frame.get("type", "")
        frame_id = frame.get("id", "")
        payload = frame.get("payload", {})

        # -- Control frames ------------------------------------------------
        if frame_type == "ping":
            pong_id = frame_id or str(uuid.uuid4())
            await self._ws.send(json.dumps({
                "type": "pong",
                "id": pong_id,
                "ts": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            }))
            logger.debug("[%s] Responded to ping with pong", self.name)
            return

        if frame_type == "heartbeat":
            logger.debug("[%s] Received heartbeat (ignored)", self.name)
            return

        if frame_type == "session.ready":
            logger.info("[%s] session.ready received", self.name)
            return

        if frame_type in ("pong", "ack"):
            logger.debug("[%s] Received %s frame (ignored)", self.name, frame_type)
            return

        # -- Event frames --------------------------------------------------
        if frame_type != "event":
            logger.warning("[%s] Unknown frame type: %s", self.name, frame_type)
            return

        source = payload.get("source", "") if isinstance(payload, dict) else ""

        # -- session.ready (can come as event type) ------------------------
        if source == "session.ready":
            logger.info("[%s] session.ready received (as event)", self.name)
            return

        # -- gateway.task: inbound user message ----------------------------
        if source == "gateway.task":
            await self._handle_gateway_task(frame_id, payload)
            return

        # -- user_dm_ok: outbound message confirmed ------------------------
        if source == "clawith.user_dm_ok":
            conv_id = payload.get("conversation_id", "")
            target_id = payload.get("target_user_id", "")
            if conv_id:
                if target_id:
                    self._user_conversations[str(target_id).lower()] = str(conv_id)
                self._conversations[str(conv_id).lower()] = (
                    str(target_id) if target_id else ""
                )
                logger.info("[%s] user_dm_ok: conversation_id=%s", self.name, conv_id)
            return

        # -- user_dm_failed: outbound message failed -----------------------
        if source == "clawith.user_dm_failed":
            msg = str(payload.get("message", "unknown"))
            logger.warning("[%s] user_dm_failed: %s", self.name, msg)
            return

        # -- peer_message_ok / peer_message_failed -------------------------
        if source in ("clawith.peer_message_ok", "clawith.peer_message_failed"):
            logger.debug("[%s] Received %s (peer messages not used)", self.name, source)
            return

        logger.warning("[%s] Unhandled event source: %s", self.name, source)

    async def _handle_gateway_task(self, event_id: str,
                                    payload: Dict[str, Any]) -> None:
        """Process a gateway.task event (inbound user message)."""
        if not event_id:
            logger.warning("[%s] gateway.task missing event id, skipping", self.name)
            return

        # Log payload structure for debugging
        logger.info("[%s] gateway.task payload_keys=%s",
                     self.name, list(payload.keys()))
        # Log message_data structure
        msg_data = payload.get("message")
        if isinstance(msg_data, dict):
            logger.info("[%s] message_data keys=%s relationships=%s",
                        self.name, list(msg_data.keys()),
                        "present" if payload.get("relationships") is not None else "absent")
            # Log all potential user_id fields
            for k in ("user_id", "userId", "sender_id", "senderId", "from_user_id", "author_id"):
                if k in msg_data:
                    logger.info("[%s]   %s = %r", self.name, k, str(msg_data[k])[:50])

        # Deduplicate
        if self._dedup.is_duplicate(event_id):
            logger.debug("[%s] Duplicate task %s, skipping", self.name, event_id)
            return

        message_data = payload.get("message", {})
        if isinstance(message_data, str):
            text = message_data
        elif isinstance(message_data, dict):
            text = (message_data.get("text", "")
                    or message_data.get("content", "")
                    or "")
            logger.info("[%s] message_data keys=%s",
                        self.name, list(message_data.keys()))
        else:
            text = str(message_data) if message_data else ""

        if not text:
            logger.debug("[%s] Empty gateway.task message, skipping", self.name)
            return

        # Extract user_id with robust logic (matching xcclawithplugin TS)
        user_id = self._extract_user_id_from_payload(payload)

        # Extract conversation info
        conversation_id = ""
        if isinstance(message_data, dict):
            conversation_id = (message_data.get("conversation_id", "")
                               or message_data.get("converter_id", "")
                               or "")
        if not conversation_id:
            conversation_id = (payload.get("conversation_id", "")
                               or payload.get("converter_id", "")
                               or "")

        if not conversation_id and user_id:
            conversation_id = self._user_conversations.get(user_id.lower(), "")

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Store mappings
        if user_id:
            self._conversations[conversation_id.lower()] = user_id
            self._user_conversations[user_id.lower()] = conversation_id

        # Track pending task for report
        self._pending_tasks[event_id] = user_id

        logger.info("[%s] Task from user=%s conv=%s: %s",
                     self.name, (user_id or "NONE")[:20],
                     conversation_id[:20], text[:80])

        # Build source for Hermes
        source = self.build_source(
            chat_id=user_id or conversation_id,
            chat_name=user_id,
            chat_type="dm",
            user_id=user_id,
            user_name=user_id,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=event_id,
            raw_message=payload,
            timestamp=datetime.now(tz=timezone.utc),
        )

        await self.handle_message(event)

    # -- Outbound messaging -------------------------------------------------

    def _is_ws_open(self) -> bool:
        """Check if WebSocket is open (works with websockets 14+ and 16+)."""
        if not self._ws:
            return False
        try:
            from websockets import State
            return self._ws.state == State.OPEN
        except (ImportError, AttributeError):
            return getattr(self._ws, 'closed', True) is False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a reply via Clawith Longlink.

        Sends ``clawith.user_dm`` to the user, then ``report`` to close the task.
        """
        metadata = metadata or {}

        if not self._is_ws_open():
            return SendResult(success=False,
                              error="Clawith WebSocket not connected")

        # Resolve target user id
        target_user_id = metadata.get("target_user_id") or chat_id
        if not target_user_id:
            for conv_id, uid in self._conversations.items():
                if uid:
                    target_user_id = uid
                    break

        if not target_user_id:
            return SendResult(success=False,
                              error="Cannot resolve target_user_id for reply")

        # Resolve conversation id
        conversation_id = (metadata.get("conversation_id")
                           or self._user_conversations.get(target_user_id.lower(), ""))
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self._user_conversations[target_user_id.lower()] = conversation_id
            self._conversations[conversation_id.lower()] = target_user_id

        try:
            # Send user_dm
            user_dm_frame = json.dumps({
                "type": "clawith.user_dm",
                "target_user_id": target_user_id,
                "content": content[:self.MAX_MESSAGE_LENGTH],
                "conversation_id": conversation_id,
            })
            await self._ws.send(user_dm_frame)
            logger.info("[%s] Sent user_dm to %s (conv=%s, len=%d)",
                        self.name, target_user_id[:20], conversation_id[:20],
                        len(content))

            # Send report for the pending task (if this is a reply to a task)
            if reply_to:
                report_frame = json.dumps({
                    "type": "report",
                    "message_id": reply_to,
                    "result": content[:200],
                    "requires_reply": False,
                })
                await self._ws.send(report_frame)
                self._pending_tasks.pop(reply_to, None)
                logger.info("[%s] Sent report for task %s", self.name, reply_to[:20])

            return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Clawith does not support typing indicators."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a Clawith conversation."""
        return {"name": chat_id, "type": "dm"}

    # -- Directory API ------------------------------------------------------

    async def search_directory(self, query: str = "", limit: int = 20) -> list:
        """Search Clawith directory for users and agents.

        Returns list of dicts with keys: kind, id, display_name, username, email.
        """
        if not self._http_client:
            return []

        url = (f"http://{self._host}:{self._directory_port}"
               f"/api/gateway/directory")
        params = {"limit": min(max(1, limit), 50)}
        if query:
            params["q"] = query

        try:
            resp = await self._http_client.get(url, params=params, headers={
                "X-Api-Key": self._api_key,
            }, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", [])
            logger.warning("[%s] Directory API returned %d: %s",
                           self.name, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("[%s] Directory API error: %s", self.name, e)

        return []

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _pick_string(obj: Dict[str, Any], keys: tuple) -> Optional[str]:
        """Pick the first non-empty string from a list of keys."""
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @staticmethod
    def _extract_user_id(message: Any) -> Optional[str]:
        """Extract user id from gateway.task payload.message (many possible shapes)."""
        if not message or not isinstance(message, dict):
            return None
        # Direct fields
        direct = ClawithAdapter._pick_string(message, (
            "user_id", "userId", "sender_id", "senderId",
            "from_user_id", "fromUserId", "sender_user_id", "senderUserId",
            "author_id", "authorId", "from_id", "fromId",
            "participant_id", "participantId", "creator_id", "creatorId",
        ))
        if direct:
            return direct
        # Nested objects
        for key in ("user", "sender", "author", "from", "participant", "contact"):
            nested = message.get(key)
            if nested and isinstance(nested, dict):
                uid = ClawithAdapter._pick_string(nested, ("id", "user_id", "userId"))
                if uid:
                    return uid
        # Meta/metadata fields
        for wrap_key in ("meta", "metadata", "attributes", "data"):
            wrap = message.get(wrap_key)
            if wrap and isinstance(wrap, dict):
                uid = ClawithAdapter._pick_string(wrap, (
                    "user_id", "userId", "sender_id", "senderId"))
                if uid:
                    return uid
        return None

    @staticmethod
    def _extract_user_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
        """Resolve user id from full gateway.task payload."""
        # Try message first
        from_msg = ClawithAdapter._extract_user_id(payload.get("message"))
        if from_msg:
            return from_msg
        # Try relationships array
        rel = payload.get("relationships")
        if isinstance(rel, list):
            for item in rel:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind", item.get("type", item.get("role", "")))).lower()
                if kind and "user" in kind:
                    uid = ClawithAdapter._pick_string(item, ("id", "user_id", "userId", "uuid"))
                    if uid:
                        return uid
                    nested_uid = ClawithAdapter._extract_user_id(item)
                    if nested_uid:
                        return nested_uid
            # Fallback: any relationship with an id
            for item in rel:
                if not isinstance(item, dict):
                    continue
                uid = ClawithAdapter._pick_string(item, ("id", "user_id", "userId", "uuid"))
                if uid:
                    return uid
                nested_uid = ClawithAdapter._extract_user_id(item)
                if nested_uid:
                    return nested_uid
        # Try relationships dict
        if isinstance(rel, dict):
            for key in ("user", "sender", "contact", "participant", "customer"):
                single = rel.get(key)
                if single and isinstance(single, dict):
                    uid = ClawithAdapter._pick_string(single, ("id", "user_id", "userId"))
                    if uid:
                        return uid
        # Top-level payload
        top = ClawithAdapter._pick_string(payload, (
            "user_id", "userId", "sender_user_id", "senderUserId",
            "from_user_id", "fromUserId",
        ))
        if top:
            return top
        return None

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with auth params."""
        return (f"ws://{self._host}:{self._longlink_port}/ws"
                f"?apiKey={self._api_key}&userId={self._user_id}")

    @staticmethod
    def _scrub_url(url: str) -> str:
        """Return a URL safe for logging (hide api key)."""
        if "?" in url:
            base, _ = url.split("?", 1)
            return f"{base}?apiKey=***&userId=..."
        return url
