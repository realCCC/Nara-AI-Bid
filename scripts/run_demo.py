"""최소 실행 경로: 업체 1개 + 공고 → 하드 필터 → 소프트 점수 → 상위 10건 (LLM·서버·로그인·DB 불필요).

uv run python scripts/run_demo.py                       # data/sample/ 사용
uv run python scripts/run_demo.py --tenders data/raw/tenders_202605.json --company data/companies/company_01.json
uv run python scripts/run_demo.py --pipeline --answers priority=S2,region=bonus   # 추천 Worker(필터→검색→리랭크) + 인터뷰
uv run python scripts/run_demo.py --pipeline --questions                          # 업체별 인터뷰 문항(LLM 생성 또는 기본) 보기
uv run python scripts/run_demo.py --pipeline --answers "q0=o1,q3=o0|o2"            # multi 문항은 | 로 구분
uv run python scripts/run_demo.py --pipeline --answers "note=시설 공사는 빼 주세요"  # 자유 입력
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from backend.graphs.recommend import run_pipeline  # noqa: E402
from backend.tools.interview import build_questions, parse_note, to_prefs  # noqa: E402
from backend.tools.rules import recommend_by_rules  # noqa: E402

SAMPLE = ROOT / "data" / "sample"


def run(tenders_path: Path, company_path: Path, region_path: Path) -> dict:
    company = load_company_file(company_path)
    result = recommend_by_rules(load_tenders(tenders_path), company, load_region_limits(region_path))
    return {"company_id": company["업체ID"], "funnel": result["funnel"], "rule_counts": result["rule_counts"],
            "top": result["top"], "excluded": result["excluded"]}


def run_worker(tenders_path: Path, company_path: Path, region_path: Path, answers: dict | None = None) -> dict:
    company = load_company_file(company_path)
    prefs = None
    if answers is not None:
        prefs = to_prefs(answers, questions=build_questions(company), note_effect=parse_note(answers.get("note"), company))
    out = run_pipeline(load_tenders(tenders_path), company, load_region_limits(region_path), prefs)
    return {"company_id": company["업체ID"], "prefs": prefs, **out}


def print_worker(out: dict):
    s = out["search"]
    print(f"검색 경로: {s['path']} ({s['reason'] or '-'}) · 리랭크: {s['rerank']} ({s['rerank_reason'] or '-'})")
    print(f"검색 문장[{s['query_source']}]: {s['query']}")
    sem = s.get("semantic") or {}
    print(f"S2 의미 매칭: {'BGE-M3 ' + ', '.join(sem.get('keywords', [])) if sem.get('used') else '문자열 일치만 (' + str(sem.get('reason')) + ')'}")
    if out["prefs"]:
        print(f"인터뷰: {out['prefs']['answers']} → 가중치 {out['prefs']['weights']}")
    print(f"퍼널: {out['filter_log']['funnel']} · 제외: {out['filter_log']['rule_counts']}")
    for r in out["top"]:
        sc = r["search"]
        print(f"{r['rank']:>2}. {r['soft_score']:>5} | rule#{sc['rule_rank']} bm25={sc['bm25']} dense={sc['dense']} "
              f"rrf={sc['rrf']} rerank={sc['rerank']} | {r['bidNtceNm'][:40]}")
        for it in r["score_detail"]["items"]:
            mark = " [인터뷰 반영]" if it["interview"] else ""
            w = "-" if it["weight"] is None else f"{it['weight']:.3f}"
            print(f"      {it['item']:<6} {it['score'] if it['score'] is not None else '-':>5} × {w} = {it['points']:>5}{mark}  {it['reason']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tenders", type=Path, default=SAMPLE / "tenders_sample.json")
    p.add_argument("--company", type=Path, default=SAMPLE / "company.json")
    p.add_argument("--regions", type=Path, default=SAMPLE / "region_limits_sample.csv")
    p.add_argument("--pipeline", action="store_true", help="추천 Worker(필터→검색→리랭크) 실행")
    p.add_argument("--answers", default=None, help="인터뷰 답변 예: budget=100_300,priority=S2,region=bonus,schedule=7")
    p.add_argument("--questions", action="store_true", help="업체별 인터뷰 문항 출력")
    args = p.parse_args()
    if args.questions:
        for q in build_questions(load_company_file(args.company)):
            print(f"[{q['id']}] ({q['source']}·{q['type']}) {q['title']}: " + " / ".join(
                f"{o['value']}={o['label']}" + (f" [{o['hint']}]" if o.get("hint") else "") for o in q.get("options", [])))
        sys.exit(0)
    if args.pipeline:
        answers = None
        if args.answers:
            answers = {k: (v.split("|") if "|" in v else v)
                       for k, v in (kv.split("=", 1) for kv in args.answers.split(","))}
        print_worker(run_worker(args.tenders, args.company, args.regions, answers))
        sys.exit(0)
    out = run(args.tenders, args.company, args.regions)
    path = ROOT / "outputs" / "recommendations" / f"{out['company_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"top": [(r["rank"], r["soft_score"], r["bidNtceNm"]) for r in out["top"]],
                      "excluded": [(e["bidNtceNo"], e["rules"]) for e in out["excluded"]]}, ensure_ascii=False, indent=1))
    print(f"→ {path} (상위 {len(out['top'])}건 + 제외 {len(out['excluded'])}건)")
