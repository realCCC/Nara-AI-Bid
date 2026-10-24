"""LLM: 인터뷰 문항 생성·자유 입력 해석, 검색 문장, RAG 순위·근거, 추천 검증(서브에이전트), 평가항목 추출. 키가 없거나 검증 실패 시 None → 호출부가 폴백."""
import json
import logging

from langchain_core.exceptions import OutputParserException
from typing import Literal

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


class Effect(BaseModel):
    """선택지를 고르면 추천에 적용되는 효과. 해당 없는 칸은 비운다."""
    budget_min_won: int | None = Field(None, description="이 금액(원) 미만 공고 제외")
    budget_max_won: int | None = Field(None, description="이 금액(원) 이상 공고 제외")
    priority: Literal["S1", "S2", "S3", "S4"] | None = Field(
        None, description="가중치를 높일 점수: S1 업종 일치, S2 역량 키워드, S3 예산 규모(실적 유사), S4 선호 계약방식")
    region: Literal["bonus", "only"] | None = Field(None, description="bonus: 업체 소재지 발주처 가점, only: 소재지 발주처만")
    min_prep_days: int | None = Field(None, description="공고~마감 준비 기간이 이 일수 미만이면 제외")
    focus_label: str | None = Field(None, description="주력 분야 이름 (12자 이내)")
    focus_keywords: list[str] = Field([], description="주력 분야에 속하는 업체 역량 키워드 (입력 키워드 중에서만)")
    focus_query: str | None = Field(None, description="이 분야 공공 용역 공고를 찾는 검색 문장 1문장")
    expand_group: str | None = Field(None, description="expand_terms가 넓혀 주는 업체 역량 키워드 (입력 키워드 그대로)")
    expand_terms: list[str] = Field([], description="공고명에 쓰일 법한 유사 단어 (공백 없는 2-8자 명사)")
    exclude_terms: list[str] = Field([], description="공고명에 있으면 제외할 단어 (공백 없는 2-8자)")


class InterviewOption(BaseModel):
    label: str = Field(description="선택지 문구 (20자 이내)")
    effect: Effect


class InterviewQuestion(BaseModel):
    title: str = Field(description="질문 (25자 이내)")
    desc: str = Field(description="이 질문이 추천에 어떻게 쓰이는지 한 줄")
    type: Literal["single", "multi"]
    options: list[InterviewOption] = Field(description="선택지 2-5개 ('전체'는 코드가 붙이므로 넣지 않는다)")


class InterviewDraft(BaseModel):
    questions: list[InterviewQuestion] = Field(description="문항 4-6개")


class KeywordMeaning(BaseModel):
    keyword: str = Field(description="입력 역량 키워드 그대로")
    description: str = Field(description="이 업체에서 이 키워드가 뜻하는 공공 용역 업무 1문장 (40자 이내)")
    synonyms: list[str] = Field(description="같은 업무를 공고명에서 부르는 다른 말 2-4개 (공백 없는 2-8자 명사)")
    sample_title: str = Field(description="이 역량으로 수행할 법한 나라장터 용역 공고명 1개 (가상)")


class KeywordProfile(BaseModel):
    keywords: list[KeywordMeaning]


class NoteParse(BaseModel):
    include_terms: list[str] = Field([], description="사용자가 찾고 싶다고 한 사업·분야 단어 (공백 없는 2-8자 명사)")
    exclude_terms: list[str] = Field([], description="사용자가 빼 달라고 한 사업·분야 단어 (공백 없는 2-8자 명사)")
    budget_min_won: int | None = None
    budget_max_won: int | None = None
    priority: Literal["S1", "S2", "S3", "S4"] | None = None
    region: Literal["bonus", "only"] | None = None
    min_prep_days: int | None = None


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


class CheckItem(BaseModel):
    category: Literal["참가자격", "실적요건", "제출서류", "일정", "기타"]
    text: str = Field(description="담당자가 확인할 조건 1문장. 원문에 있는 사실만 (예: 행사대행 직접생산확인증명서 보유 업체)")
    file: str
    page: int
    quote: str = Field(description="공고서 원문 그대로 인용 (짧게)")


class ChecklistResult(BaseModel):
    items: list[CheckItem] = Field(description="확인 항목 3-8개")


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

INTERVIEW_PROMPT = """당신은 공공 입찰 공고 추천 전에 업체 담당자에게 물을 객관식 인터뷰를 설계합니다.
업체 프로필을 읽고, 이 업체에 맞는 추천 조건을 고를 수 있는 문항 4-6개를 만드세요.
규칙:
- 선택지마다 effect를 채우세요. effect는 추천에 실제로 적용되는 효과이고, 효과가 없는 선택지는 버려집니다.
- 한 문항의 선택지는 같은 종류의 effect 하나만 쓰고(예: 예산 문항은 budget만), 선택지 문구는 effect와 같은 뜻이어야 합니다.
- 이 업체에 맞게: 사업 규모는 참여가능금액·과거 실적 금액을 기준으로 구간을 나누고,
  주력 분야는 역량 키워드를 2-4개 분야로 묶고(focus_*), 지역은 소재지를 넣어 쓰세요.
- 유사 표현 문항은 type=multi로, 선택지 하나에 단어 하나(expand_group=넓혀 줄 역량 키워드, expand_terms=[그 단어]).
  공공 용역 공고명에 실제로 쓰일 법한 단어로 (예: 행사 → 이벤트 / 회의 → 콘퍼런스). 입력 키워드와 같은 단어는 쓰지 마세요.
- 넓은 단어(사업, 용역, 서비스, 운영, 지원, 관리)는 키워드로 쓰지 마세요.
- 중요 기준(priority), 준비 기간(min_prep_days), 피할 사업(exclude_terms) 문항도 업체 상황에 맞으면 넣으세요.
- '전체'·'상관없음' 선택지는 넣지 마세요 (코드가 붙입니다). 낙찰 가능성을 묻지 마세요.

[업체 프로필]
{company}"""

KEYWORD_PROMPT = """업체의 역량 키워드마다, 공공 용역 공고 검색에 쓸 의미 정보를 만드세요.
- description: 업종·과거 실적에 비추어 이 업체에서 그 키워드가 뜻하는 업무 (예: AI → 교원 대상 AI 활용 교육).
- synonyms: 나라장터 공고명에서 같은 업무를 부르는 다른 말 (예: 행사 → 이벤트, 기념식, 축제).
  입력 키워드와 같은 단어, 지역·금액, 넓은 단어(사업, 용역, 서비스, 운영, 지원, 관리)는 쓰지 마세요.
- sample_title: 이 역량으로 수행할 법한 공고명 1개 (실제 공고처럼, 기관명은 ○○로).
모든 입력 키워드를 한 번씩 다루세요.

[업종] {biz}
[역량 키워드] {keywords}
[과거 실적] {past}"""

NOTE_PROMPT = """업체 담당자가 공고 추천 조건을 자유롭게 적었습니다. 적힌 내용만 아래 항목으로 옮기세요.
- 찾고 싶은 사업 → include_terms, 빼 달라는 사업 → exclude_terms (공고명에 쓰일 공백 없는 명사).
- 금액은 원 단위 정수로 (예: 2억 이하 → budget_max_won=200000000).
- 지역: 업체 소재지({region}) 발주처만 원하면 only, 우대 정도면 bonus.
- 가장 중요하다고 한 기준이 있으면 priority (S1 업종 일치, S2 역량 키워드, S3 예산 규모, S4 선호 계약방식).
- 적히지 않은 항목은 비우세요. 추측하지 마세요.

[업체 역량 키워드] {keywords}
[입력]
{note}"""

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

CHECKLIST_PROMPT = """아래 공고서 청크에서, 업체 담당자가 입찰 참가 전에 확인해야 할 조건을 3-8개 뽑으세요.
- 분류: 참가자격(등록·면허·직접생산·중소기업 등), 실적요건, 제출서류, 일정(제출 마감·제안설명회 등), 기타(공동수급·하도급 제한 등).
- 원문에 있는 조건만 쓰고, 항목마다 file·page·quote(원문 그대로, 짧게)를 붙이세요.
- 일반적인 안내 문구나 원문에 없는 추측은 쓰지 마세요.

[공고] {name}
[청크]
{chunks}"""

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


def generate_interview(company: dict) -> dict | None:
    profile = {k: company.get(k) for k in ("업종", "소재지", "중소기업여부", "참여가능금액", "역량키워드", "선호계약방식", "과거실적")}
    result = _invoke(InterviewDraft, INTERVIEW_PROMPT.format(company=json.dumps(profile, ensure_ascii=False)))
    return result.model_dump() if result else None


def keyword_profile(company: dict) -> dict | None:
    result = _invoke(KeywordProfile, KEYWORD_PROMPT.format(
        biz=", ".join(company.get("업종", {}).get("대분류", [])), keywords=", ".join(company.get("역량키워드", [])),
        past=", ".join(h["프로젝트명"] for h in company.get("과거실적", []))))
    return result.model_dump() if result else None


def parse_note(note: str, company: dict) -> dict | None:
    result = _invoke(NoteParse, NOTE_PROMPT.format(region=company.get("소재지", "-"), note=note,
                                                   keywords=", ".join(company.get("역량키워드", []))))
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


def extract_checklist(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(ChecklistResult, CHECKLIST_PROMPT.format(name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result else None


def extract_criteria(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(CriteriaResult, CRITERIA_PROMPT.format(name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result and result.items else None
