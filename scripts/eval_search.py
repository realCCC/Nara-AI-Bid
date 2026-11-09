"""검색·리랭크 개선 비교: 같은 검색 문장(data/query_cache.json 고정)으로 변형별 상위 10건을 LLM 판정(0/1/2)으로 채점.

변형
  before : 리랭크에 검색 문장 전체 1개 (개선 전)
  clause : 검색 문장을 절로 나눠 절별 리랭크 점수 중 최댓값 (개선 1)
  clause30: clause + 리랭크 구간 15 → 30 (RRF 16위 밖 검색 누락 구제 시도)
판정·pooling·캐시는 eval_body.py와 같다 (data/eval/body_judgments.json을 함께 쓴다).

uv run python scripts/eval_search.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.graphs.recommend import rule_stage, search_stage  # noqa: E402
from backend.tools import search  # noqa: E402
from backend.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from scripts.eval_body import OUT, judge, ndcg  # noqa: E402

CLAUSES = search.query_clauses
VARIANTS = {"before": (lambda q: [q], 15), "clause": (CLAUSES, 15), "clause30": (CLAUSES, 30)}


def top10(stage: dict, by_no: dict, company: dict, variant: str) -> list[str]:
    clauses, top = VARIANTS[variant]
    search.query_clauses = clauses
    try:
        s = search_stage(stage["candidates"], by_no, company)
        docs = {c["bidNtceNo"]: search.tender_doc(by_no[c["bidNtceNo"]]) for c in stage["candidates"]}
        r = search.rerank(s, docs, top=top) if docs else s
        return r["order"][:config.TOP_K]
    finally:
        search.query_clauses = CLAUSES


def main():
    if not config.llm_enabled():
        sys.exit("LLM 키 필요 (.env)")
    tenders, regions = load_tenders(), load_region_limits()
    by_no = {t["bidNtceNo"]: t for t in tenders}
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    rows = []
    for path in sorted(config.COMPANIES_DIR.glob("*.json")):
        company = load_company_file(path)
        cid = company["업체ID"]
        stage = rule_stage(tenders, company, regions)
        tops = {v: top10(stage, by_no, company, v) for v in VARIANTS}
        pool_ids = list(dict.fromkeys(i for t in tops.values() for i in t))
        known = saved.setdefault(cid, {})
        if todo := [no for no in pool_ids if no not in known]:
            known.update(judge(company, [by_no[no] for no in todo]))
            OUT.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        pool = [known.get(no, 0) for no in pool_ids]
        row = {"company": cid}
        for v, t in tops.items():
            gs = [known.get(no, 0) for no in t]
            row[v] = {"p@10": sum(g >= 1 for g in gs) / 10, "strict@10": sum(g == 2 for g in gs) / 10,
                      "ndcg": round(ndcg(gs, pool), 3)}
        rows.append(row)
        print(cid, " | ".join(f"{v} {row[v]}" for v in VARIANTS), flush=True)
    summary = {v: {m: round(sum(r[v][m] for r in rows) / len(rows), 3) for m in ("p@10", "strict@10", "ndcg")}
               for v in VARIANTS}
    for v, s in summary.items():
        print(f"평균 {v:<9} {s}")
    path = ROOT / "outputs" / "eval" / "search_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
