"""LLM 두 곳: RAG 순위·근거, 평가항목 추출. 키가 없거나 검증 실패 시 None → 호출부가 폴백."""
import json
import logging

from pydantic import BaseModel, Field

from app.config import get_llm

log = logging.getLogger(__name__)
MAX_RETRY = 2


class Reason(BaseModel):
    text: str
    source_type: str = Field(description="pdf 또는 field")
    file: str | None = None
    page: int | None = None
    quote: str | None = Field(None, description="공고서 원문 그대로 인용 (pdf일 때)")
    field: str | None = Field(None, description="tender.<필드명> 또는 company.<필드명> (field일 때)")
    value: str | None = None


class Recommendation(BaseModel):
    rank: int
    bidNtceNo: str
    reasons: list[Reason]
    qualification_flags: list[str] = []


class RankResult(BaseModel):
    recommendations: list[Recommendation]


class Criterion(BaseModel):
    item: str
    score: float | None = None
    file: str | None = None
    page: int | None = None
    quote: str | None = None


class CriteriaResult(BaseModel):
    items: list[Criterion]
    period: str | None = Field(None, description="과업 기간. 원문에 없으면 null")


RANK_PROMPT = """당신은 공공 입찰 공고 검토를 돕는 분석가입니다.
업체 프로필과 후보 공고(룰 점수 상위)의 필드·공고서 청크만 보고, 업체가 먼저 검토할 순서로 모든 후보의 순위를 매기세요.
규칙:
- 입력에 있는 정보로만 판단하고, 없는 정보는 "확인 필요"라고 쓰세요.
- 공고마다 근거 3-4개. 공고서 근거는 source_type=pdf, file·page와 원문 그대로의 quote를 붙이세요.
  필드 근거는 source_type=field, field(예: tender.pubPrcrmntClsfcNo)와 value를 붙이세요.
- 참가자격상 의심되는 점은 qualification_flags에 적으세요.
- 금지: 낙찰 확률 예측, 최종 입찰 결정, 입력에 없는 인용 생성.

[업체 프로필]
{company}

[후보 공고]
{candidates}"""

CRITERIA_PROMPT = """아래 공고서 청크에서 제안서 평가항목과 배점, 과업 기간을 추출하세요.
- 원문에 있는 항목만 추출하고, 항목마다 file·page·quote(원문 그대로)를 붙이세요.
- 배점이 없으면 score=null, 과업 기간이 없으면 period=null.

[공고] {name}
[청크]
{chunks}"""


def _invoke(schema, prompt: str):
    llm = get_llm()
    if llm is None:
        return None
    structured = llm.with_structured_output(schema)
    for attempt in range(MAX_RETRY + 1):
        try:
            return structured.invoke(prompt)
        except Exception as e:  # 파싱·검증 실패 → 재시도
            log.warning("LLM 호출 실패 (%d/%d): %s", attempt + 1, MAX_RETRY + 1, e)
    return None


def _chunks_text(chunks: list[dict]) -> str:
    return "\n".join(f"- ({c['file']} p.{c['page']}) {c['text']}" for c in chunks)


def rag_rank(company: dict, candidates: list[dict], chunks_by_tender: dict[str, list[dict]],
             tenders: dict[str, dict]) -> list[dict] | None:
    fields = ("bidNtceNm", "pubPrcrmntLrgClsfcNm", "pubPrcrmntClsfcNo", "pubPrcrmntClsfcNm", "asignBdgtAmt",
              "cntrctCnclsMthdNm", "bidPrtcptLmtYn", "rgnLmtBidLocplcJdgmBssNm")
    blocks = []
    for c in candidates:
        t = tenders[c["bidNtceNo"]]
        blocks.append(json.dumps({"bidNtceNo": c["bidNtceNo"], "rule_score": c["soft_score"],
                                  "fields": {f: t.get(f, "") for f in fields}}, ensure_ascii=False)
                      + "\n청크:\n" + (_chunks_text(chunks_by_tender.get(c["bidNtceNo"], [])) or "(PDF 근거 없음)"))
    result = _invoke(RankResult, RANK_PROMPT.format(company=json.dumps(company, ensure_ascii=False),
                                                    candidates="\n\n".join(blocks)))
    if result is None:
        return None
    ids = {c["bidNtceNo"] for c in candidates}
    recs = [r for r in result.recommendations if r.bidNtceNo in ids]
    if len({r.bidNtceNo for r in recs}) != len(ids):  # 후보 누락·중복 → 검증 실패
        log.warning("RAG 출력 검증 실패: 후보 %d건 중 %d건", len(ids), len(recs))
        return None
    return [r.model_dump() for r in sorted(recs, key=lambda r: r.rank)]


def extract_criteria(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(CriteriaResult, CRITERIA_PROMPT.format(name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result and result.items else None
