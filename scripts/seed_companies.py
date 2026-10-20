"""data/companies/*.json → SQLite companies 테이블 (업체 ID 기준 upsert, 재실행해도 중복 없음).

uv run python scripts/seed_companies.py [--dir data/companies] [--db data/app.db]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app import config  # noqa: E402
from app.db import repository as repo  # noqa: E402
from app.schemas import CompanySeed  # noqa: E402


def seed(companies_dir: Path, db_path: Path | None = None) -> tuple[int, list[str]]:
    repo.init_db(db_path)
    ok, failed = 0, []
    for f in sorted(companies_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            CompanySeed.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as e:
            failed.append(f"{f.name}: {e}")
            continue
        repo.upsert_company(data, db_path)
        ok += 1
    return ok, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=config.COMPANIES_DIR)
    p.add_argument("--db", type=Path, default=None)
    args = p.parse_args()
    ok, failed = seed(args.dir, args.db)
    print(f"companies {ok}건 적재 → {args.db or config.SQLITE_PATH}")
    for msg in failed:
        print("적재 제외:", msg)
    sys.exit(1 if failed else 0)
