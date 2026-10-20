"""최소 실행 경로: 업체 1개 + 공고 → 하드 필터 → 소프트 점수 → 상위 10건 (LLM·서버·로그인·DB 불필요).

uv run python scripts/run_demo.py                       # data/sample/ 사용
uv run python scripts/run_demo.py --tenders data/raw/tenders_202605.json --company data/companies/company_01.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from app.tools.rules import recommend_by_rules  # noqa: E402

SAMPLE = ROOT / "data" / "sample"


def run(tenders_path: Path, company_path: Path, region_path: Path) -> dict:
    company = load_company_file(company_path)
    result = recommend_by_rules(load_tenders(tenders_path), company, load_region_limits(region_path))
    return {"company_id": company["업체ID"], "funnel": result["funnel"], "rule_counts": result["rule_counts"],
            "top": result["top"], "excluded": result["excluded"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tenders", type=Path, default=SAMPLE / "tenders_sample.json")
    p.add_argument("--company", type=Path, default=SAMPLE / "company.json")
    p.add_argument("--regions", type=Path, default=SAMPLE / "region_limits_sample.csv")
    args = p.parse_args()
    out = run(args.tenders, args.company, args.regions)
    path = ROOT / "outputs" / "recommendations" / f"{out['company_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"top": [(r["rank"], r["soft_score"], r["bidNtceNm"]) for r in out["top"]],
                      "excluded": [(e["bidNtceNo"], e["rules"]) for e in out["excluded"]]}, ensure_ascii=False, indent=1))
    print(f"→ {path} (상위 {len(out['top'])}건 + 제외 {len(out['excluded'])}건)")
