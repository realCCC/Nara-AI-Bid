"""LLM 네 곳: 검색 문장, RAG 순위·근거, 추천 검증(서브에이전트), 평가항목 추출. 키가 없거나 검증 실패 시 None → 호출부가 폴백."""
import json
import logging

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError

from backend.config import get_llm

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


class SearchQuery(BaseModel):
    query: str = Field(description="공고 검색용 자연어 문장 1-2개")


class FocusOption(BaseModel):
    label: str = Field(description="분야 이름 (12자 이내)")
    keywords: list[str] = Field(description="이 분야에 속하는 업체 역량 키워드 (입력 키워드 중에서만)")
    query: str = Field(description="이 분야 공공 용역 공고를 찾는 검색 문장 1문장")


class Expansion(BaseModel):
    keyword: str = Field(description="입력 역량 키워드 그대로")
    similar: list[str] = Field(description="공고명에 쓰일 법한 유사·관련 단어 2-3개 (공백 없는 2-8자 명사)")


class InterviewSuggest(BaseModel):
    focus: list[FocusOption] = Field(description="역량 키워드를 묶은 주력 분야 2-4개")
    expansions: list[Expansion]


class VerifyItem(BaseModel):
    bidNtceNo: str
    fit_score: int = Field(ge=0, le=100, description="업체 적합도 재평가 (0-100)")
    comment: str = Field(description="재평가 이유 1-2문장. 입력에 없는 사실은 쓰지 않는다")


class VerifyResult(BaseModel):
    items: list[VerifyItem]
    order: list[str] = Field(description="조정한 순위: 입력 후보 공고번호 전부를 적합한 순서로")


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

QUERY_PROMPT = """아래 업체 프로필을 보고, 이 업체가 수행할 만한 공공 용역 공고를 찾기 위한 검색 문장을 한국어 1-2문장으로 쓰세요.
업종·역량 키워드·과거 실적의 사업 유형을 담고, 지역·금액 같은 조건은 빼세요.

[업체 프로필]
{company}"""

SUGGEST_PROMPT = """업체의 역량 키워드를 공고 검색에 쓰기 좋게 정리하세요.
1) focus: 역량 키워드를 사업 분야 2-4개로 묶고(키워드는 입력 키워드 중에서만, 여러 분야에 공통인 넓은 키워드는 빼고),
   분야마다 그 분야 공고를 찾는 검색 문장 1문장을 쓰세요.
2) expansions: 키워드마다 공공 용역 공고명에 실제로 쓰일 법한 유사·관련 단어 2-3개를 쓰세요.
   공백 없는 단어로 (예: 행사 → 이벤트, 기념식 / 회의 → 콘퍼런스, 포럼).
   입력 키워드와 같은 단어, 지역·금액, 너무 넓은 단어(사업, 용역, 서비스, 운영, 지원)는 쓰지 마세요.

[업종] {biz}
[역량 키워드] {keywords}
[과거 실적] {past}"""

VERIFY_PROMPT = """당신은 추천 결과를 다시 확인하는 검증 담당입니다.
추천 Worker가 고른 상위 후보와 업체 프로필을 보고, 후보마다 적합도(0-100)를 재평가하고 순위를 조정하세요.
규칙:
- 입력에 있는 정보로만 판단하세요. 낙찰 가능성은 평가하지 않습니다.
- order에는 입력 후보 공고번호를 빠짐없이 한 번씩 넣으세요 (새 공고번호 금지).
- comment는 재평가 이유를 1-2문장으로.

[업체 프로필]
{company}

[상위 후보]
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


def search_query(company: dict) -> str | None:
    profile = {k: company.get(k) for k in ("업종", "역량키워드", "과거실적")}
    result = _invoke(SearchQuery, QUERY_PROMPT.format(company=json.dumps(profile, ensure_ascii=False)))
    return result.query.strip() if result and result.query.strip() else None


def suggest_interview(company: dict) -> dict | None:
    result = _invoke(InterviewSuggest, SUGGEST_PROMPT.format(
        biz=", ".join(company.get("업종", {}).get("대분류", [])), keywords=", ".join(company.get("역량키워드", [])),
        past=", ".join(h["프로젝트명"] for h in company.get("과거실적", []))))
    return result.model_dump() if result else None


def check_verify(result, ids: list[str]) -> tuple[dict | None, str | None]:
    """검증 출력 확인. 통과하면 (결과, None), 아니면 (None, 사유)."""
    if result is None:
        return None, "형식 오류: 빈 응답"
    try:
        result = result if isinstance(result, VerifyResult) else VerifyResult.model_validate(result)
    except ValidationError as e:
        return None, f"형식 오류: {e.errors()[0]['msg']}"
    unknown = sorted({x for x in [*result.order, *(i.bidNtceNo for i in result.items)] if x not in ids})
    if unknown:
        return None, f"없는 공고ID: {', '.join(unknown)}"
    if len(result.order) != len(ids) or set(result.order) != set(ids):
        return None, "형식 오류: 조정 순위에 후보 누락·중복"
    if {i.bidNtceNo for i in result.items} != set(ids):
        return None, "형식 오류: 재평가 누락"
    return result.model_dump(), None


def verify_ranking(company: dict, candidates: list[dict], tenders: dict[str, dict]) -> tuple[dict | None, str | None]:
    """검증 서브에이전트. 재시도 없이 1회 — 실패하면 호출부가 원순위를 유지한다."""
    llm = get_llm()
    if llm is None:
        return None, "LLM 키 없음"
    blocks = [json.dumps({"bidNtceNo": c["bidNtceNo"], "rank": c["rank"], "bidNtceNm": c["bidNtceNm"],
                          "rule_score": c["soft_score"], "reasons": [r["text"] for r in c.get("reasons", [])],
                          "계약방식": tenders[c["bidNtceNo"]].get("cntrctCnclsMthdNm"),
                          "예산": tenders[c["bidNtceNo"]].get("budget")}, ensure_ascii=False) for c in candidates]
    prompt = VERIFY_PROMPT.format(company=json.dumps(company, ensure_ascii=False), candidates="\n".join(blocks))
    try:
        result = llm.with_structured_output(VerifyResult).invoke(prompt)
    except (ValidationError, OutputParserException) as e:
        return None, f"형식 오류: {type(e).__name__}"
    except Exception as e:
        log.warning("검증 서브에이전트 예외: %s", e)
        return None, f"예외: {type(e).__name__}"
    return check_verify(result, [c["bidNtceNo"] for c in candidates])


def extract_criteria(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(CriteriaResult, CRITERIA_PROMPT.format(name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result and result.items else None
