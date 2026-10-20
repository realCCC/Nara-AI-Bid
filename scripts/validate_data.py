"""데이터 검증: 업체 시드 스키마 + 업체별 하드 필터 통과 건수 (10건 미만이면 경고).

uv run python scripts/validate_data.py [--list-region-limited]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app import config  # noqa: E402
from app.schemas import CompanySeed  # noqa: E402
from app.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from app.tools.rules import recommend_by_rules  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list-region-limited", action="store_true", help="지역 제한 공고(PDF 보유) 목록 출력")
    args = p.parse_args()
    tenders = load_tenders()

    if args.list_region_limited:
        mapped = load_region_limits()
        for t in tenders:
            if t["pdfs"] and t.get("rgnLmtBidLocplcJdgmBssNm"):
                print(f"{t['bidNtceNo']}\t{'매핑됨' if t['bidNtceNo'] in mapped else '미매핑'}\t{t['bidNtceNm']}")
        sys.exit(0)

    errors = []
    for f in sorted(config.COMPANIES_DIR.glob("*.json")):
        try:
            CompanySeed.model_validate(json.loads(f.read_text(encoding="utf-8")))
        except (ValidationError, json.JSONDecodeError) as e:
            errors.append(f"{f.name}: {e}")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"데이터 스키마 통과 (업체 {len(list(config.COMPANIES_DIR.glob('*.json')))}건, 공고 {len(tenders)}건)")

    region_limits = load_region_limits()
    low = 0
    for f in sorted(config.COMPANIES_DIR.glob("*.json")):
        c = load_company_file(f)
        r = recommend_by_rules(tenders, c, region_limits)
        warn = " ← 10건 미만: 업종·금액 범위 조정 필요" if r["funnel"]["hard_pass"] < 10 else ""
        low += bool(warn)
        print(f"{c['업체ID']} 통과 {r['funnel']['hard_pass']:>3}건  제외 {r['rule_counts']}{warn}")
    sys.exit(1 if low else 0)
