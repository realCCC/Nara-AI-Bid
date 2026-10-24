"""SQLite 저장소: 업체(로그인·프로필)와 작업 이력. 표준 sqlite3만 사용."""
import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from backend import config

SCHEMA = Path(__file__).with_name("schema.sql")
# 컬럼으로 분리하는 시드 필드. 나머지는 profile_json
COLUMN_FIELDS = {"업체ID": "company_id", "회사명": "name", "중소기업여부": "is_sme", "소재지": "region"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.SQLITE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))


# ---------- 비밀번호 (PoC: 데모용 업체 식별. 실제 인증 아님 — 4번 문서 4-2) ----------
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt_hex)), stored)


# ---------- 업체 ----------
def upsert_company(seed: dict, db_path=None) -> None:
    """시드 JSON 1건 → companies (업체 ID 기준 upsert)."""
    profile = {k: v for k, v in seed.items() if k not in COLUMN_FIELDS and k != "비밀번호"}
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO companies (company_id, name, is_sme, region, password_hash, profile_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(company_id) DO UPDATE SET name=excluded.name, is_sme=excluded.is_sme, region=excluded.region,
                 password_hash=excluded.password_hash, profile_json=excluded.profile_json, updated_at=excluded.updated_at""",
            (seed["업체ID"], seed["회사명"], int(bool(seed["중소기업여부"])), seed["소재지"],
             hash_password(str(seed["비밀번호"])), json.dumps(profile, ensure_ascii=False), _now()),
        )


def count_companies(db_path=None) -> int:
    with closing(connect(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]


def _row_to_profile(row: sqlite3.Row) -> dict:
    return {"업체ID": row["company_id"], "회사명": row["name"], "중소기업여부": bool(row["is_sme"]),
            "소재지": row["region"], **json.loads(row["profile_json"])}


def load_company_profile(company_id: str, db_path=None) -> dict | None:
    """업체 프로필 (비밀번호 해시 제외). 추천 로직은 이 dict만 받는다."""
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM companies WHERE company_id = ?", (company_id,)).fetchone()
    return _row_to_profile(row) if row else None


def login(company_id: str, password: str, db_path=None) -> dict:
    """{'ok': True, 'profile': {...}} 또는 {'ok': False, 'error': 메시지}"""
    with closing(connect(db_path)) as conn:
        if conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 0:
            return {"ok": False, "error": "업체 데이터 미적재 (scripts/seed_companies.py 실행)"}
        row = conn.execute("SELECT * FROM companies WHERE company_id = ?", (company_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "존재하지 않는 업체 ID입니다."}
    if not verify_password(password, row["password_hash"]):
        return {"ok": False, "error": "비밀번호가 일치하지 않습니다."}
    return {"ok": True, "profile": _row_to_profile(row)}


# ---------- 작업 이력 ----------
def save_run(thread_id: str, company_id: str, status: str, *, recommendations=None, selection=None,
             proposal=None, db_path=None) -> None:
    """노드 끝에서 호출. thread_id 기준 upsert라 재실행해도 중복 없음."""
    now = _now()
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO runs (thread_id, company_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at""",
            (thread_id, company_id, status, now, now),
        )
        if recommendations is not None:
            conn.execute("DELETE FROM recommendations WHERE thread_id = ?", (thread_id,))
            conn.executemany(
                "INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(thread_id, r["rank"], r["bidNtceNo"], r.get("bidNtceNm"), r.get("soft_score"),
                  json.dumps(r.get("reasons", []), ensure_ascii=False), r.get("source", "rule")) for r in recommendations],
            )
        if selection is not None:
            conn.execute("INSERT OR REPLACE INTO selections VALUES (?, ?, ?)", (thread_id, selection, now))
        if proposal is not None:
            conn.execute(
                "INSERT OR REPLACE INTO proposals VALUES (?, ?, ?, ?, ?)",
                (thread_id, proposal["bidNtceNo"], json.dumps(proposal.get("criteria"), ensure_ascii=False),
                 proposal.get("html"), now),
            )


def list_runs(company_id: str, db_path=None) -> list[dict]:
    """로그인 업체의 작업만 최근순. 선택 공고명(없으면 1순위 공고명)을 함께."""
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            """SELECT r.thread_id, r.status, r.updated_at, s.bidNtceNo AS selected,
                      COALESCE((SELECT bidNtceNm FROM recommendations x WHERE x.thread_id = r.thread_id AND x.bidNtceNo = s.bidNtceNo),
                               (SELECT bidNtceNm FROM recommendations x WHERE x.thread_id = r.thread_id AND x.rank = 1)) AS title
               FROM runs r LEFT JOIN selections s ON s.thread_id = r.thread_id
               WHERE r.company_id = ? ORDER BY r.updated_at DESC, r.created_at DESC""",
            (company_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(thread_id: str, db_path=None) -> dict | None:
    with closing(connect(db_path)) as conn:
        run = conn.execute("SELECT * FROM runs WHERE thread_id = ?", (thread_id,)).fetchone()
        if not run:
            return None
        recs = conn.execute("SELECT * FROM recommendations WHERE thread_id = ? ORDER BY rank", (thread_id,)).fetchall()
        sel = conn.execute("SELECT * FROM selections WHERE thread_id = ?", (thread_id,)).fetchone()
        prop = conn.execute("SELECT * FROM proposals WHERE thread_id = ?", (thread_id,)).fetchone()
    return {
        "run": dict(run),
        "recommendations": [{**dict(r), "reasons": json.loads(r["reasons_json"] or "[]")} for r in recs],
        "selection": dict(sel) if sel else None,
        "proposal": dict(prop) if prop else None,
    }
