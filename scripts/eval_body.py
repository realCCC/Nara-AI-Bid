"""1단계 PDF 본문 순위(BODY_SEARCH) 효과 비교: 끔/켬 상위 10건을 LLM 판정(0 부적합·1 부분·2 적합)으로 채점.

업체마다 두 결과의 합집합만 판정한다 (pooling). 판정은 data/eval/body_judgments.json에 저장해 다시 쓴다.
판정 입력은 공고명·분류 + 과업 내용 청크 — 제목만 보고 판단하지 않게.
사전 준비: docker compose up -d, scripts/warm_chunk_vectors.py --all

uv run python scripts/eval_body.py
"""
import json
import math
import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.graphs.recommend import run_pipeline  # noqa: E402
from backend.tools import llm  # noqa: E402
from backend.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from backend.tools.pdf import tender_chunks  # noqa: E402
from backend.tools.retrieval import retrieve  # noqa: E402

OUT = ROOT / "data" / "eval" / "body_judgments.json"
TASK_QUERY = "과업 내용 과업 범위 주요 업무 수행 내용"

JUDGE_PROMPT = """공공 용역 공고가 아래 업체에 맞는지 판정하세요. 지역·금액·자격 조건은 보지 말고, 업체가 실제로 수행할 수 있는 일인지만 보세요.
2: 업체의 주력 분야와 맞는 일 / 1: 일부 업무만 겹침 / 0: 업체가 하는 일이 아님

[업체]
{company}

[공고]
{tenders}"""


class Grade(BaseModel):
    bidNtceNo: str
    grade: int = Field(ge=0, le=2)


class Grades(BaseModel):
    grades: list[Grade]


def tender_view(t: dict) -> str:
    chunks = retrieve(tender_chunks(t), TASK_QUERY, k=3, tender_no=t["bidNtceNo"])
    body = " / ".join(c["text"][:300] for c in chunks) or "(PDF 없음)"
    return (f"- {t['bidNtceNo']} | {t['bidNtceNm']} | 분류: {t.get('pubPrcrmntClsfcNm') or '-'}"
            f" | 구매대상: {t.get('purchsObjPrdctList') or '-'}\n  과업 내용: {body}")


def judge(company: dict, tenders: list[dict]) -> dict[str, int]:
    profile = {k: company.get(k) for k in ("업종", "역량키워드", "과거실적")}
    result = llm._invoke(Grades, JUDGE_PROMPT.format(company=json.dumps(profile, ensure_ascii=False),
                                                     tenders="\n".join(tender_view(t) for t in tenders)))
    if result is None:
        sys.exit("LLM 판정 실패")
    return {g.bidNtceNo: g.grade for g in result.grades}


def ndcg(grades: list[int], ideal: list[int], k: int = 10) -> float:
    dcg = lambda gs: sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))  # noqa: E731
    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else 0.0


if __name__ == "__main__":
    if not config.llm_enabled():
        sys.exit("LLM 키 필요 (.env)")
    tenders, regions = load_tenders(config.TENDERS_FILE), load_region_limits()
    by_no = {t["bidNtceNo"]: t for t in tenders}
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    rows = []
    for path in sorted(config.COMPANIES_DIR.glob("*.json")):
        company = load_company_file(path)
        cid = company["업체ID"]
        tops = {}
        for name, on in (("off", False), ("on", True)):
            config.BODY_SEARCH = on
            tops[name] = [r["bidNtceNo"] for r in run_pipeline(tenders, company, regions)["top"]]
        known = saved.setdefault(cid, {})
        todo = [no for no in dict.fromkeys(tops["off"] + tops["on"]) if no not in known]
        if todo:
            known.update(judge(company, [by_no[no] for no in todo]))
            OUT.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        pool = [known.get(no, 0) for no in dict.fromkeys(tops["off"] + tops["on"])]
        row = {"company": cid, "new": len(set(tops["on"]) - set(tops["off"]))}
        for name in ("off", "on"):
            gs = [known.get(no, 0) for no in tops[name]]
            row[name] = {"p@10": sum(g >= 1 for g in gs) / 10, "strict@10": sum(g == 2 for g in gs) / 10,
                         "ndcg": round(ndcg(gs, pool), 3)}
        rows.append(row)
        print(f"{cid} 새로 들어온 공고 {row['new']}건 | off {row['off']} | on {row['on']}", flush=True)
    for m in ("p@10", "strict@10", "ndcg"):
        off, on = (sum(r[n][m] for r in rows) / len(rows) for n in ("off", "on"))
        print(f"평균 {m:<10} off {off:.3f} → on {on:.3f} ({on - off:+.3f})")
