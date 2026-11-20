"""SQLite accounts + profile store (stdlib only). Local file, no telemetry."""

import hashlib
import logging
import os
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("THERAPIST_DB", "data/therapist.db"))

# The 11 onboarding fields (plan §5); stored as one row per user.
PROFILE_FIELDS = (
    "call_name",
    "age_bracket",
    "gender",
    "language_pref",
    "religion",
    "family_religion",
    "faith_importance",
    "faith_stance",
    "faith_consent",
    "supports",
    "concern",
)

MINOR_DECLINE = (
    "Thanks for being honest — this space is only for adults (18+), so I can't "
    "start counselling with you here. If you need someone to talk to right now, "
    "please call Childline at 1098 (24/7, free for children) or Tele MANAS at 14416."
)


class MinorError(ValueError):
    """Raised when a profile declares age_bracket <18 (holds the warm decline text)."""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
              pw_salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created_at TEXT NOT NULL);
           CREATE TABLE IF NOT EXISTS tokens
             (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT NOT NULL);
           CREATE TABLE IF NOT EXISTS profiles
             (user_id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}',
              updated_at TEXT NOT NULL);
           CREATE TABLE IF NOT EXISTS messages
             (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, session_id TEXT NOT NULL,
              role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);""")
    return con


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)


def create_user(username: str, password: str) -> int:
    if not username.strip() or not password:
        raise ValueError("username and password are required")
    salt = secrets.token_bytes(16)
    try:
        with _connect() as con:
            cur = con.execute(
                "INSERT INTO users (username, pw_salt, pw_hash, created_at) VALUES (?,?,?,?)",
                (username.strip(), salt, _hash(password, salt), _now()),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError as e:
        raise ValueError("username taken") from e


def verify_user(username: str, password: str) -> int | None:
    with _connect() as con:
        row = con.execute(
            "SELECT id, pw_salt, pw_hash FROM users WHERE username=?", (username.strip(),)
        ).fetchone()
    if row is None or _hash(password, row[1]) != row[2]:
        return None
    log.debug("login ok for user_id=%s", row[0])
    return row[0]


def issue_token(user_id: int) -> str:
    token = secrets.token_hex(32)
    with _connect() as con:
        con.execute(
            "INSERT INTO tokens (token, user_id, created_at) VALUES (?,?,?)",
            (token, user_id, _now()),
        )
    return token


def user_id_for_token(token: str) -> int | None:
    with _connect() as con:
        row = con.execute("SELECT user_id FROM tokens WHERE token=?", (token,)).fetchone()
    return row[0] if row else None


def revoke_token(token: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM tokens WHERE token=?", (token,))


def username_for(user_id: int) -> str | None:
    with _connect() as con:
        row = con.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    return row[0] if row else None


def add_message(user_id: int, session_id: str, role: str, content: str) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO messages (user_id, session_id, role, content, created_at)"
            " VALUES (?,?,?,?,?)",
            (user_id, session_id, role, content, _now()),
        )


def get_history(user_id: int, session_id: str) -> list:
    with _connect() as con:
        rows = con.execute(
            "SELECT role, content FROM messages WHERE user_id=? AND session_id=?" " ORDER BY id",
            (user_id, session_id),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]


def list_sessions(user_id: int) -> list:
    with _connect() as con:
        rows = con.execute(
            "SELECT id, session_id, content, created_at FROM messages"
            " WHERE user_id=? ORDER BY id",
            (user_id,),
        ).fetchall()
    sessions: dict = {}
    latest: dict = {}
    for mid, sid, content, created in rows:
        sessions.setdefault(sid, content[:80])
        latest[sid] = (created, mid)
    return sorted(
        (
            {"id": sid, "preview": sessions[sid], "updated_at": updated}
            for sid, (updated, _) in latest.items()
        ),
        key=lambda s: (s["updated_at"], latest[s["id"]][1]),
        reverse=True,
    )


def get_profile(user_id: int) -> dict:
    import json

    with _connect() as con:
        row = con.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return json.loads(row[0]) if row else {}


def put_profile(user_id: int, fields: dict) -> dict:
    import json

    unknown = set(fields) - set(PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    if fields.get("age_bracket") == "<18":
        raise MinorError(MINOR_DECLINE)
    merged = get_profile(user_id) | {k: v for k, v in fields.items() if v is not None}
    with _connect() as con:
        con.execute(
            "INSERT INTO profiles (user_id, data, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
            (user_id, json.dumps(merged), _now()),
        )
    return merged
