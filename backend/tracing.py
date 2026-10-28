"""LangSmith 추적: LangGraph 노드·LLM 호출은 LANGSMITH_TRACING=true만으로 자동 기록된다.
여기서는 LangChain 밖의 검색·RAG 도구(BM25·BGE·리랭커·PDF 청킹·출처 대조)를 같은 트레이스 트리에 넣는다.

입력이 큰 함수(공고 443건 문서, 공고서 청크 전체)는 요약만 남긴다. 추적이 꺼져 있으면 데코레이터는 그대로 통과.
"""
from langsmith import traceable

PREVIEW = 200  # 청크 본문 미리보기 글자 수


def chunk_view(chunks: list[dict]) -> list[dict]:
    return [{"file": c.get("file"), "page": c.get("page"), "section": c.get("section"),
             "text": (c.get("text") or "")[:PREVIEW],
             **({"match": c["match"]} if "match" in c else {})} for c in chunks]


def _drop_big(inputs: dict, big: tuple[str, ...]) -> dict:
    """큰 인자는 건수만 남긴다 (chunks → n_chunks)."""
    out = {}
    for k, v in inputs.items():
        if k in big and isinstance(v, dict) and "bidNtceNo" in v:  # 공고 1건은 공고번호만
            out["bidNtceNo"] = v["bidNtceNo"]
        elif k in big:
            out[f"n_{k}"] = len(v) if v is not None else 0
        elif k == "company" and isinstance(v, dict):
            out["업체ID"] = v.get("업체ID")
        else:
            out[k] = v
    return out


def traced(name: str, run_type: str = "tool", big: tuple[str, ...] = (), outputs=None):
    """traceable 래퍼. big: 건수만 남길 인자 이름, outputs: 출력 요약 함수."""
    return traceable(name=name, run_type=run_type,
                     process_inputs=lambda inputs: _drop_big(inputs, big),
                     process_outputs=(lambda o: outputs(o["output"] if isinstance(o, dict) and "output" in o else o))
                     if outputs else None)
