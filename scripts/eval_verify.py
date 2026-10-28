"""검증 에이전트 평가: 정답셋(추천/적격-비추천/부적격 라벨)으로 검증 전후 순위를 비교한다.

검증 입력(RAG까지 끝난 상위 N건)을 스냅샷으로 고정해 두고, 검증만 반복 실행한다.
앞 단계 결과가 실행마다 달라져도 라벨이 어긋나지 않고, LLM 호출도 검증 1회/업체로 줄어든다.

1) 스냅샷 + 라벨 템플릿 (업체별 추천 Worker → RAG, LLM 키 필요. 기존 라벨은 유지)
   uv run python scripts/eval_verify.py snapshot
2) data/eval/verify_gold.json 의 label 칸 채우기: 추천 / 적격-비추천 / 부적격 (부적격이면 reason에 이유 한 줄)
   순위를 보지 않고 라벨링하도록 템플릿은 공고번호 순으로 섞어 두고 순위는 적지 않는다.
3) 평가 (같은 스냅샷으로 검증을 runs번 실행해 평균)
   uv run python scripts/eval_verify.py eval --runs 3
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
INPUTS = EVAL_DIR / "verify_inputs.json"
GOLD = EVAL_DIR / "verify_gold.json"
LABELS = {"추천": 2, "적격-비추천": 1, "부적격": 0}  # nDCG 이득


def top_n() -> int:
    return max(1, min(config.VERIFY_TOP_N, 5))


# ---------- 지표 (순수 함수) ----------
def ndcg(order: list[str], gold: dict[str, str]) -> float | None:
    gains = [LABELS[gold[no]] for no in order if no in gold]
    if not gains:
        return None
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)))
    return dcg / ideal if ideal else 1.0


def precision_at(order: list[str], gold: dict[str, str], k: int = 3) -> float | None:
    head = [no for no in order[:k] if no in gold]
    return sum(gold[no] == "추천" for no in head) / len(head) if head else None


def bad_below(order: list[str], gold: dict[str, str]) -> tuple[int, int]:
    """(부적격, 부적격 아닌 공고) 쌍 중 부적격이 아래에 있는 쌍 수, 전체 쌍 수.
    부적격이 이미 맨 아래면 더 내려갈 수 없어 '강등률'은 왜곡되므로 쌍 단위로 본다."""
    labeled = [no for no in order if no in gold]
    pairs = [(i, j) for i, a in enumerate(labeled) for j, b in enumerate(labeled)
             if gold[a] == "부적격" and gold[b] != "부적격"]
    return sum(i > j for i, j in pairs), len(pairs)


def company_metrics(before: list[str], after: list[str], flagged: set[str], gold: dict[str, str]) -> dict:
    """before/after: 상위 N건 공고번호 순서, flagged: 확인된 issue가 붙은 공고, gold: 라벨 붙은 공고만."""
    bad = {no for no in before if gold.get(no) == "부적격"}
    (ok_b, pairs), (ok_a, _) = bad_below(before, gold), bad_below(after, gold)
    return {"ndcg_before": ndcg(before, gold), "ndcg_after": ndcg(after, gold),
            "p3_before": precision_at(before, gold), "p3_after": precision_at(after, gold),
            "bad": len(bad), "pairs": pairs, "below_before": ok_b, "below_after": ok_a,
            "flagged": len(flagged), "flagged_bad": len(flagged & bad), "changed": before != after}


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    ratio = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"companies": len(rows),
            "nDCG@N": {"before": mean("ndcg_before"), "after": mean("ndcg_after")},
            "P@3": {"before": mean("p3_before"), "after": mean("p3_after")},
            "부적격 하위 배치율": {"before": ratio(total("below_before"), total("pairs")),
                             "after": ratio(total("below_after"), total("pairs"))},
            "플래그 정밀도": ratio(total("flagged_bad"), total("flagged")),
            "플래그 재현율": ratio(total("flagged_bad"), total("bad")),
            "순위 변경률": ratio(sum(r["changed"] for r in rows), len(rows))}


# ---------- 스냅샷 ----------
def snapshot():
    from backend.agents.rag import rag_recommend
    from backend.graphs.recommend import run_pipeline
    from backend.tools.loaders import load_company_file, load_region_limits, load_tenders

    if not config.llm_enabled():
        print("※ LLM 키가 없어 RAG 대신 룰 순위로 스냅샷을 만듭니다 (검증 평가에는 키가 필요합니다).")
    tenders, regions, n = load_tenders(), load_region_limits(), top_n()
    old = json.loads(GOLD.read_text(encoding="utf-8")) if GOLD.exists() else {}
    inputs, gold = {}, {}
    for path in sorted(config.COMPANIES_DIR.glob("*.json")):
        company = load_company_file(path)
        cid = company["업체ID"]
        out = run_pipeline(tenders, company, regions)
        state = {"company_profile": company, "top10": out["top"], "search": out["search"], "prefs": None}
        recs = rag_recommend(state)["recommendations"][:n]
        inputs[cid] = {"company_profile": company, "recommendations": recs}
        prev = {t["bidNtceNo"]: t for t in old.get(cid, {}).get("tenders", [])}
        gold[cid] = {"회사명": company.get("회사명"), "tenders": [
            {"bidNtceNo": r["bidNtceNo"], "bidNtceNm": r["bidNtceNm"],
             "label": prev.get(r["bidNtceNo"], {}).get("label"), "reason": prev.get(r["bidNtceNo"], {}).get("reason", "")}
            for r in sorted(recs, key=lambda r: r["bidNtceNo"])]}
        print(f"{cid}: 상위 {len(recs)}건")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    GOLD.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    todo = sum(t["label"] is None for g in gold.values() for t in g["tenders"])
    print(f"→ {GOLD.relative_to(ROOT)} (라벨 미입력 {todo}건: 추천/적격-비추천/부적격)")


# ---------- 평가 ----------
def load_gold() -> dict[str, dict[str, str]]:
    gold = {}
    for cid, g in json.loads(GOLD.read_text(encoding="utf-8")).items():
        for t in g["tenders"]:
            if t["label"] is None:
                continue
            if t["label"] not in LABELS:
                raise SystemExit(f"{cid} {t['bidNtceNo']}: 라벨은 추천/적격-비추천/부적격 중 하나 ({t['label']!r})")
            gold.setdefault(cid, {})[t["bidNtceNo"]] = t["label"]
    return gold


def evaluate(runs: int):
    from backend.agents.verify import verify_recommendations

    if not config.llm_enabled():
        raise SystemExit("LLM 키가 없으면 검증이 생략되어 평가할 수 없습니다 (.env 확인).")
    inputs, gold = json.loads(INPUTS.read_text(encoding="utf-8")), load_gold()
    if not gold:
        raise SystemExit(f"라벨이 없습니다. {GOLD.relative_to(ROOT)} 의 label 칸을 채우세요.")
    per_run, skipped = [], 0
    for run in range(1, runs + 1):
        rows = []
        for cid, labels in gold.items():
            recs = inputs[cid]["recommendations"]
            state = {"company_profile": inputs[cid]["company_profile"], "recommendations": recs, "prefs": None}
            out = verify_recommendations(state, {})
            v = out["verification"]
            if v["status"] != "passed":
                skipped += 1
                print(f"[{run}] {cid}: 검증 생략 ({v['reason']})")
                continue
            head = out["recommendations"][:v["checked"]]
            flagged = {r["bidNtceNo"] for r in head if r["verify"]["issues"]}
            rows.append(company_metrics([r["bidNtceNo"] for r in recs[:v["checked"]]], [r["bidNtceNo"] for r in head],
                                        flagged, labels))
            for r in head:
                for i in r["verify"]["issues"]:
                    print(f"[{run}] {cid} {r['bidNtceNo']} ({labels.get(r['bidNtceNo'], '라벨 없음')}) "
                          f"issue[{i['type']}] {i['text']}")
        per_run.append(summarize(rows))
        print(f"[{run}] {json.dumps(per_run[-1], ensure_ascii=False)}")
    report = {"runs": per_run, "skipped": skipped, "labeled": sum(map(len, gold.values()))}
    path = ROOT / "outputs" / "eval" / "verify_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["snapshot", "eval"])
    p.add_argument("--runs", type=int, default=3, help="eval 반복 횟수 (LLM 출력 편차 확인)")
    args = p.parse_args()
    snapshot() if args.cmd == "snapshot" else evaluate(args.runs)
