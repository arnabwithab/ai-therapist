"""F024 db tests — schema, scrypt round-trip, tokens, profile validation."""

import sqlite3

import pytest

from backend.core import db


@pytest.fixture()
def mem(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    return db.DB_PATH


def tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_schema_created_on_first_use(mem):
    db.create_user("a", "pw123456")
    assert {"users", "tokens", "profiles", "messages"} <= tables(mem)


def test_password_round_trip(mem):
    uid = db.create_user("ara", "s3cret-pw")
    assert db.verify_user("ara", "s3cret-pw") == uid
    assert db.verify_user("ara", "wrong-pw") is None
    assert db.verify_user("nobody", "s3cret-pw") is None


def test_passwords_salted(mem):
    db.create_user("u1", "same-pw")
    db.create_user("u2", "same-pw")
    con = sqlite3.connect(mem)
    try:
        hashes = [r[0] for r in con.execute("SELECT pw_hash FROM users")]
    finally:
        con.close()
    assert hashes[0] != hashes[1]


def test_duplicate_and_blank_rejected(mem):
    db.create_user("ara", "s3cret-pw")
    with pytest.raises(ValueError):
        db.create_user("ara", "other-pw")
    with pytest.raises(ValueError):
        db.create_user("", "s3cret-pw")
    with pytest.raises(ValueError):
        db.create_user("bob", "")


def test_token_issue_revoke(mem):
    uid = db.create_user("ara", "s3cret-pw")
    tok = db.issue_token(uid)
    assert db.user_id_for_token(tok) == uid
    assert db.user_id_for_token("bogus") is None
    db.revoke_token(tok)
    assert db.user_id_for_token(tok) is None


def test_profile_round_trip(mem):
    uid = db.create_user("ara", "s3cret-pw")
    assert db.get_profile(uid) == {}
    saved = db.put_profile(uid, {"call_name": "Ara", "religion": "Hindu", "concern": "sleep"})
    assert saved == {"call_name": "Ara", "religion": "Hindu", "concern": "sleep"}
    assert db.get_profile(uid) == saved
    # partial update merges
    assert db.put_profile(uid, {"concern": "exams"})["call_name"] == "Ara"


def test_profile_isolated_between_users(mem):
    a = db.create_user("a", "pw-a-long")
    b = db.create_user("b", "pw-b-long")
    db.put_profile(a, {"call_name": "A"})
    assert db.get_profile(b) == {}


def test_profile_rejects_unknown_field(mem):
    uid = db.create_user("ara", "s3cret-pw")
    with pytest.raises(ValueError):
        db.put_profile(uid, {"chakras": "open"})


def test_profile_minor_raises_with_helplines(mem):
    uid = db.create_user("kid", "pw-kid-long")
    with pytest.raises(db.MinorError) as e:
        db.put_profile(uid, {"age_bracket": "<18"})
    assert "1098" in str(e.value) and "14416" in str(e.value)
    assert db.get_profile(uid) == {}
