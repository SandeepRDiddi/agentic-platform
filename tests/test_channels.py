import asyncio
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from agentic.channels import slack, teams
from agentic.common.errors import AuthError
from agentic.common.schemas import TaskStatus

SECRET = "test-signing-secret"


def sign(body: bytes, ts: int | None = None, secret: str = SECRET):
    ts = str(ts or int(time.time()))
    sig = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


def test_slack_signature_rules():
    body = b'{"a":1}'
    h = sign(body)
    slack.verify_signature(SECRET, body, h["x-slack-request-timestamp"], h["x-slack-signature"])
    with pytest.raises(AuthError):
        slack.verify_signature(SECRET, body + b"x", h["x-slack-request-timestamp"], h["x-slack-signature"])
    old = sign(body, ts=int(time.time()) - 3600)
    with pytest.raises(AuthError):
        slack.verify_signature(SECRET, body, old["x-slack-request-timestamp"], old["x-slack-signature"])
    with pytest.raises(AuthError):
        slack.verify_signature(None, body, h["x-slack-request-timestamp"], h["x-slack-signature"])
    with pytest.raises(AuthError):
        slack.verify_signature(SECRET, body, None, None)


async def test_slack_url_verification_and_bad_signature(bare_stack):
    s = bare_stack
    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()
    r = await s.channels_api.post("/slack/events", content=body, headers=sign(body))
    assert r.json() == {"challenge": "abc"}
    bad = await s.channels_api.post("/slack/events", content=body, headers=sign(body, secret="wrong"))
    assert bad.status_code == 401


def mention(text, event_id="Ev1", user="U1", ts="1700000000.000100"):
    return json.dumps({"type": "event_callback", "event_id": event_id,
                       "event": {"type": "app_mention", "user": user, "text": f"<@UBOT> {text}", "channel": "C1", "ts": ts}}).encode()


async def test_slack_mention_queues_task_dedupes_retries_and_replies(stack):
    s = stack
    body = mention("Create a ticket for the failing nightly ingestion job")
    assert (await s.channels_api.post("/slack/events", content=body, headers=sign(body))).status_code == 200
    await s.channels_api.post("/slack/events", content=body, headers=sign(body))  # Slack retry → ignored
    msgs = s.c.bus.messages(s.settings.topic_tasks)
    assert len(msgs) == 1
    task = await s.c.tasks.get(msgs[0].value["task_id"])
    assert task.user_id == "slack:U1" and task.query.startswith("Create a ticket") and "<@" not in task.query

    await s.worker.handle(msgs[0])
    (res,) = s.c.bus.messages(s.settings.topic_results)
    await s.channels.state.dispatcher.handle(res.value)
    (post,) = [r for r in s.ext_calls if "chat.postMessage" in str(r.url)]
    payload = json.loads(post.content)
    assert payload["channel"] == "C1" and payload["thread_ts"] == "1700000000.000100"
    buttons = [e["action_id"] for b in payload["blocks"] if b["type"] == "actions" for e in b["elements"]]
    assert any(b.startswith("approve:") for b in buttons) and any(b.startswith("deny:") for b in buttons)
    assert post.headers["authorization"] == "Bearer xoxb-test"

    # approver clicks the button in Slack (a different person from the requester)
    approval_id = [b for b in buttons if b.startswith("approve:")][0].split(":")[1]
    form = urlencode({"payload": json.dumps({"user": {"id": "U2"}, "actions": [{"action_id": f"approve:{approval_id}"}]})}).encode()
    r = await s.channels_api.post("/slack/interactive", content=form, headers=sign(form))
    assert r.status_code == 200 and "executed" in r.json()["text"]
    assert len(s.internal.state.tickets.rows) == 1
    # and the requester clicking approve is refused by four-eyes
    r2 = await s.run("Create a ticket for the broken dashboard", sub="slack:U1")
    form = urlencode({"payload": json.dumps({"user": {"id": "U1"}, "actions": [{"action_id": f"approve:{r2['pending_approvals'][0]['id']}"}]})}).encode()
    r = await s.channels_api.post("/slack/interactive", content=form, headers=sign(form))
    assert "Could not record decision" in r.json()["text"]


async def test_slack_ignores_bot_messages(bare_stack):
    s = bare_stack
    body = json.dumps({"type": "event_callback", "event_id": "E9", "event": {"type": "message", "bot_id": "B1", "text": "hi", "channel": "C", "ts": "1"}}).encode()
    assert (await s.channels_api.post("/slack/events", content=body, headers=sign(body))).status_code == 200
    assert s.c.bus.messages(s.settings.topic_tasks) == []


# ── Teams ─────────────────────────────────────────────────────────────────────────────────
def test_teams_service_url_allowlist():
    assert teams.trusted_service_url("https://smba.trafficmanager.net/emea/")
    assert teams.trusted_service_url("https://x.botframework.com")
    assert not teams.trusted_service_url("http://smba.trafficmanager.net")
    assert not teams.trusted_service_url("https://evil.example.com")
    assert not teams.trusted_service_url("https://botframework.com.evil.io")


class FakeJwks:
    def __init__(self, key):
        self.key = key

    def get_signing_key_from_jwt(self, token):
        class K:
            pass

        k = K()
        k.key = self.key.public_key()
        return k


def teams_token(key, aud="app-id", iss=teams.ISSUER, exp=3600):
    return jwt.encode({"aud": aud, "iss": iss, "exp": int(time.time()) + exp}, key, algorithm="RS256")


def test_teams_jwt_validation():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    v = teams.TeamsJwtVerifier("app-id", FakeJwks(key))
    assert v.verify("Bearer " + teams_token(key))["aud"] == "app-id"
    for bad in (teams_token(key, aud="other"), teams_token(key, iss="https://evil"), teams_token(key, exp=-10),
                teams_token(rsa.generate_private_key(public_exponent=65537, key_size=2048))):
        with pytest.raises(AuthError):
            v.verify("Bearer " + bad)
    with pytest.raises(AuthError):
        v.verify(None)


async def test_teams_message_flow(stack):
    import httpx

    from agentic.channels.app import create_app

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    app = create_app(stack.c, external=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"access_token": "t", "expires_in": 3600}))),
                     teams_verifier=teams.TeamsJwtVerifier("app-id", FakeJwks(key)))
    api = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://channels")
    act = {"type": "message", "id": "a1", "text": "<at>Bot</at> What is NNDM?", "serviceUrl": "https://smba.trafficmanager.net/emea/",
           "from": {"id": "29:x", "aadObjectId": "oid-1"}, "conversation": {"id": "19:conv"}}
    h = {"authorization": "Bearer " + teams_token(key)}
    assert (await api.post("/teams/messages", json=act, headers={"authorization": "Bearer junk"})).status_code == 401
    assert (await api.post("/teams/messages", json={**act, "serviceUrl": "https://evil.example.com"}, headers=h)).status_code == 403
    assert (await api.post("/teams/messages", json=act, headers=h)).status_code == 200
    (m,) = stack.c.bus.messages(stack.settings.topic_tasks)
    t = await stack.c.tasks.get(m.value["task_id"])
    assert t.user_id == "teams:oid-1" and t.query == "What is NNDM?" and t.reply_to["channel"] == "teams"


# ── Web ───────────────────────────────────────────────────────────────────────────────────
async def test_web_chat_sse_flow(stack):
    s = stack
    assert "Agent Platform" in (await s.channels_api.get("/")).text
    tok = (await s.channels_api.post("/dev/token", json={"sub": "web-user"})).json()["access_token"]
    h = {"authorization": f"Bearer {tok}"}
    assert (await s.channels_api.post("/api/chat", json={"query": "What is NNDM?"})).status_code == 401
    r = await s.channels_api.post("/api/chat", json={"query": "What is NNDM?", "conversation_id": "w1"}, headers=h)
    assert r.status_code == 202
    tid = r.json()["task_id"]
    runner = asyncio.create_task(s.worker.run())
    try:
        resp = await asyncio.wait_for(s.channels_api.get(f"/api/chat/{tid}/events", headers=h), 20)
    finally:
        s.worker.stop()
        await runner
    assert "event: result" in resp.text
    last = [ln for ln in resp.text.splitlines() if ln.startswith("data:")][-1]
    assert json.loads(last[5:])["status"] == TaskStatus.SUCCEEDED.value
    # another user cannot stream someone else's task
    other = (await s.channels_api.post("/dev/token", json={"sub": "mallory"})).json()["access_token"]
    denied = await s.channels_api.get(f"/api/chat/{tid}/events", headers={"authorization": f"Bearer {other}"})
    assert "event: error" in denied.text and "event: result" not in denied.text
