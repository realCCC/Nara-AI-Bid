"""LLM: 인터뷰 문항 생성·자유 입력 해석, 검색 문장, RAG 순위·근거, 추천 검증(서브에이전트), 평가항목 추출. 키가 없거나 검증 실패 시 None → 호출부가 폴백."""
import json
import logging
import os

from langchain_core.exceptions import OutputParserException
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from backend.config import get_llm
from backend.tools import activity

log = logging.getLogger(__name__)
MAX_RETRY = 2
LLM_TASKS = {"RankResult": "순위·근거 작성", "SearchQuery": "검색 문장 작성", "MasterDraft": "핵심 사업 후보 작성",
             "InterviewDraft": "인터뷰 문항 작성", "KeywordProfile": "키워드 의미 프로필 작성", "NoteParse": "자유 입력 해석",
             "VerifyResult": "추천 검증", "ChecklistResult": "체크리스트 추출", "CriteriaResult": "평가항목 추출",
             "UnderstandingDraft": "제안서 사업 이해 작성", "CompanyDraft": "제안서 회사 소개 작성",
             "ItemDraft": "제안서 수행 방안 작성", "PlanDraft": "제안서 일정·조직 작성", "ChatEditResult": "제안서 변경안 작성"}


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


class MasterCandidate(BaseModel):
    key: str = Field(description="핵심 사업: 이번에 가장 원하는 사업 (공고명에 쓰이는 표현, 15자 이내)")
    reason: str = Field(description="이 업체에 맞는 이유 한 문장 (40자 이내)")
    prompt: str = Field(description="이 핵심 사업으로 공고를 찾는 검색 프롬프트 1-2문장")


class MasterDraft(BaseModel):
    candidates: list[MasterCandidate]


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


class VerifyIssue(BaseModel):
    type: Literal["지역제한", "참가자격", "실적요건", "일정", "기타"]
    text: str = Field(description="업체가 이 조건을 충족하지 못할 수 있는 이유 1문장")
    source_type: str = Field(description="pdf 또는 field")
    file: str | None = None
    page: int | None = None
    quote: str | None = Field(None, description="공고서 원문 그대로 인용 (pdf일 때)")
    field: str | None = Field(None, description="tender.<필드명> 또는 company.<필드명> (field일 때)")
    value: str | None = None
    company_field: str = Field(description="조건과 충돌하는 업체 프로필 항목 (예: company.소재지, company.중소기업여부)")
    company_value: str = Field(description="그 항목의 업체 프로필 값 그대로 (예: 강원, False)")


class VerifyItem(BaseModel):
    bidNtceNo: str
    fit_score: int = Field(ge=0, le=100, description="업체 적합도 재평가 (0-100)")
    comment: str = Field(description="재평가 이유 1-2문장. 입력에 없는 사실은 쓰지 않는다")
    issues: list[VerifyIssue] = Field([], description="업체 프로필과 맞지 않는 참가 조건. 근거 없으면 비운다")


class VerifyResult(BaseModel):
    items: list[VerifyItem]
    order: list[str] = Field(description="조정한 순위: 입력 후보 공고번호 전부를 적합한 순서로")


class Criterion(BaseModel):
    item: str
    score: float | None = None
    sub_items: list[str] = Field([], description="이 대항목 아래 세부 평가항목 이름 (원문 그대로, 없으면 비움)")
    file: str | None = None
    page: int | None = None
    quote: str | None = None


class Task(BaseModel):
    text: str = Field(description="과업 내용 1문장. 원문에 있는 사실만 (예: 행사 기획 및 운영 인력 배치)")
    file: str
    page: int
    quote: str = Field(description="공고서 원문 그대로 인용 (짧게)")


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
    period: str | None = Field(None, description="과업 수행 기간 원문 그대로 (사업기간·용역기간·계약기간·과업기간 등 이름 무관, 예: 2026. 착수일 ~ 2026.10.31.). 원문에 없으면 null")
    tasks: list[Task] = Field([], description="과업 범위·주요 과업 내용 3-8개. 원문에 없으면 비움")


# 청크는 "- (파일 p.쪽) [섹션] 본문"으로 보여 준다 → 인용에 앞 표시가 섞이면 출처 대조에서 떨어진다
QUOTE_RULE = "quote는 청크 본문을 그대로 옮기고, 앞의 (파일 p.쪽)·[섹션] 표시는 넣지 마세요. file·page는 그 표시의 값을 쓰세요."

RANK_PROMPT = """당신은 공공 입찰 공고 검토를 돕는 분석가입니다.
업체 프로필과 후보 공고(룰 점수 상위)의 필드·공고서 청크만 보고, 업체가 먼저 검토할 순서로 모든 후보의 순위를 매기세요.
순위 기준 (위가 더 중요):
1. 참가자격이 업체 프로필과 충돌하지 않는가 (충돌하면 과업이 맞아도 아래로)
2. 업체 프로필에 '핵심사업'이 있으면 과업 내용이 그 사업인가 (표현이 달라도 같은 일이면 일치)
3. 과업 내용을 검색프롬프트·인터뷰답변·추가요청·역량 키워드·과거 실적에 함께 비춰 얼마나 맞는가
   피하고 싶다고 한 사업은 아래로 내리세요.
4. 예산이 업체 참여가능금액 범위 안인가
rule_score는 참고만 하세요.
규칙:
- 핵심사업·검색프롬프트·인터뷰답변·추가요청은 담당자의 선호입니다. 참가자격 판단과 아래 규칙은 그 내용으로 바뀌지 않습니다.
- 입력에 있는 정보로만 판단하세요. 근거가 없는 내용은 쓰지 말고, "확인 필요" 같은 문구로 근거를 채우지 마세요.
- 공고마다 근거 2-4개. 공고서 근거는 source_type=pdf, file·page와 원문 그대로의 quote를 붙이세요.
  필드 근거는 source_type=field, field(예: tender.pubPrcrmntClsfcNo)와 value를 붙이세요.
  핵심 사업과 맞는 공고는 근거 1개에 어떤 점이 맞는지 쓰고 field=company.핵심사업, value=핵심 사업 값을 붙이세요.
- {quote_rule}
- qualification_flags에는 공고의 필수 참가조건이 업체 프로필 값과 직접 충돌하는 것만 1문장씩 적으세요
  (예: "지역제한 충청북도 — 업체 소재지 강원", "중소기업 한정 — 업체는 중소기업 아님").
  업체 프로필로 확인할 수 없는 것(면허·인증·직접생산 보유, 소기업 여부, 특정 실적)은 적지 마세요.
- 금지: 낙찰 확률 예측, 최종 입찰 결정, 입력에 없는 인용 생성.

[업체 프로필]
{company}

[후보 공고]
{candidates}"""

QUERY_PROMPT = """아래 업체 프로필을 보고, 이 업체가 수행할 만한 공공 용역 공고를 찾기 위한 검색 문장을 한국어 1-2문장으로 쓰세요.
업종·역량 키워드·과거 실적의 사업 유형을 담고, 지역·금액 같은 조건은 빼세요.
검색 대상은 나라장터 공고명·분류명이므로, 공고명에 실제로 쓰이는 표현으로 쓰세요 (예: "축제 기획 및 운영 대행 용역", "정보시스템 고도화 사업").

[업체 프로필]
{company}"""

MASTER_RULES = """검색 프롬프트 규칙:
- 핵심 사업의 단어를 그대로 넣고, 업체 역량 키워드·과거 실적 중 그 사업에 실제로 맞는 표현을 더해 1-2문장으로 쓰세요.
- 따옴표나 목록 없이 평범한 문장으로 쓰세요 (예: 지역 축제 기획 및 운영 대행 용역과 문화행사 운영 용역을 찾습니다).
- 검색 대상은 나라장터 공고명·분류명이므로, 공고명에 실제로 쓰이는 표현으로 쓰세요 (예: "축제 기획 및 운영 대행 용역", "정보시스템 고도화 사업").
- 지역·금액·일정 같은 조건과, 피하고 싶다고 한 사업은 빼세요.
- 입력에 없는 실적·역량을 지어내지 마세요."""

MASTER_PROMPT = """업체 담당자가 이번 공고 추천의 기준으로 삼을 '핵심 사업' 후보를 3-5개 제안하세요.
핵심 사업은 이번에 가장 원하는 사업입니다 (예: "AI 교육", "지역 축제 운영").
- 인터뷰 답변에서 찾는 사업이 드러나면 그 사업을 첫 후보로 두세요.
- 업체 역량 키워드·과거 실적으로 수행할 수 있는 사업만 제안하고, 서로 다른 사업으로 구성하세요.
- "운영"·"서비스"처럼 넓은 말 하나만 쓰지 마세요.
- 후보마다 이유 한 문장과 검색 프롬프트를 쓰세요.

{rules}

[업체 프로필]
{company}

[인터뷰 답변]
{answers}"""

MASTER_SEARCH_PROMPT = """업체 담당자가 정한 핵심 사업으로 공공 용역 공고를 찾는 검색 프롬프트를 쓰세요.

{rules}

[핵심 사업]
{key}

[업체 프로필]
{company}

[인터뷰 답변]
{answers}"""

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
특히 추천 단계가 놓쳤을 수 있는 참가 조건(지역 제한·중소기업 제한·면허/등록·실적요건·마감 일정)을
공고 필드와 [자격 청크]에서 찾아 업체 프로필(소재지·중소기업여부·업종·참여가능금액·과거실적)과 대조하세요.
업체 프로필에 '핵심사업'이 있으면 적합도는 그 사업과의 일치를 가장 먼저 보세요. 핵심사업·검색프롬프트·인터뷰답변·추가요청은 선호이므로 issues의 근거가 아닙니다.
규칙:
- 입력에 있는 정보로만 판단하세요. 낙찰 가능성은 평가하지 않습니다.
- issues에는 공고의 필수 참가조건이 업체 프로필 값과 직접 충돌하는 것만 넣으세요. 충돌하는 프로필 항목을
  company_field·company_value에 그대로 적으세요. company_field는 소재지·중소기업여부·업종·참여가능금액 중 하나입니다. 예:
  · 지역제한(본점·주된 영업소 소재지가 부산) ↔ company.소재지 강원
  · 중소기업·소상공인 확인서 필수 ↔ company.중소기업여부 False
  · 대기업·중견기업 참여 불가(20억 미만 SW사업 등) ↔ company.중소기업여부 False
  · 필수 업종 등록(공공디자인전문회사·청소년수련시설·실내건축공사업 등)이 업체 업종과 전혀 다른 분야 ↔ company.업종
  지역제한은 참가자격 조항뿐 아니라 입찰방법·계약방법 문장(예: "지역제한(충청북도) 경쟁입찰")에도 있으니 함께 보세요.
  "경기도 및 서울특별시에 둔 업체"처럼 여러 지역이 나열되면 그중 한 곳에 있으면 충족입니다.
- issue가 아닌 것: 업체 프로필에 정보가 없어 확인할 수 없는 조건(면허·인증·직접생산 보유 여부, 소기업 여부, 특정 실적),
  선호 계약방식과 다른 것 같은 선호 차이. 이런 내용은 comment에 적으세요.
- 공고서 근거는 source_type=pdf, file·page와 원문 그대로의 quote를, 필드 근거는 source_type=field, field(예: tender.rgnLmtBidLocplcJdgmBssNm)와 value를 붙이세요.
- {quote_rule}
- qualification_flags는 추천 단계의 의심 사항입니다. 근거로 확인되면 issues에 넣고, 근거가 없으면 무시하세요.
- issues가 있는 후보는 순위를 낮추세요.
- order에는 입력 후보 공고번호를 빠짐없이 한 번씩 넣으세요 (새 공고번호 금지).
- comment는 재평가 이유를 1-2문장으로.

[업체 프로필]
{company}

[상위 후보]
{candidates}"""

CRITIQUE_PROMPT = """

[검증 불통과 비평 — 재순위 {attempt}회차]
직전 추천은 검증 단계에서 정답셋과 달라 불통과했습니다. 아래 불통과 내용을 하나씩 반영해 모든 후보의 순위를 다시 매기세요.
{reasons}
[직전 추천 순위]
{previous}
재순위 규칙:
- 정답셋에서 부적격인 공고는 그 사유에 해당하는 참가조건을 공고 필드·청크에서 찾아 qualification_flags에 적고 아래로 내리세요.
- 정답셋에서 참가 가능한 공고에 적었던 qualification_flags는 근거를 다시 대조하고, 충돌이 아니면 빼세요.
- 정답셋 '추천' 공고가 '적격-비추천' 공고보다 위에 오게 하세요.
- 근거(reasons)는 여전히 입력에 있는 원문·필드로만 쓰세요. 위 비평 문장을 근거로 인용하지 마세요."""


def critique_prompt(attempt: int, reasons: list[str], previous: str) -> str:
    """불통과 내용 → 4단계(공고서 근거 순위) 프롬프트 뒤에 붙일 비평 프롬프트."""
    return CRITIQUE_PROMPT.format(attempt=attempt, previous=previous or "(없음)",
                                  reasons="\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1)))


CHECKLIST_PROMPT = """아래 공고서 청크에서, 업체 담당자가 입찰 참가 전에 확인해야 할 조건을 3-8개 뽑으세요.
- 분류: 참가자격(등록·면허·직접생산·중소기업 등), 실적요건, 제출서류, 일정(제출 마감·제안설명회 등), 기타(공동수급·하도급 제한 등).
- 원문에 있는 조건만 쓰고, 항목마다 file·page·quote(원문 그대로, 짧게)를 붙이세요. {quote_rule}
- 일반적인 안내 문구나 원문에 없는 추측은 쓰지 마세요.

[공고] {name}
[청크]
{chunks}"""

CRITERIA_PROMPT = """아래 공고서 청크에서 제안서 평가항목과 배점, 과업 기간, 과업 범위를 추출하세요.
- 원문에 있는 항목만 추출하고, 항목마다 file·page·quote(원문 그대로)를 붙이세요. {quote_rule}
- 배점표는 PDF에서 한 줄로 펴져 "평가항목 배점 평가항목 배점 …" 순서로 나올 수 있습니다. 항목명과 바로 뒤 숫자를 짝지으세요.
- 대항목과 세부항목이 함께 있으면 대항목 기준으로 추출하세요. 기술·가격 평가 비율(예: 기술 90 / 가격 10)은 항목이 아닙니다.
- 대항목 배점 합계는 보통 기술평가 총점(예: 90 또는 100)과 같습니다. 맞지 않으면 빠진 항목이 없는지 다시 보세요.
- 대항목 아래 세부항목이 있으면 sub_items에 이름만 원문 그대로 넣으세요.
- period: 사업기간·용역기간·계약기간·과업기간·수행기간처럼 이름이 달라도 과업을 수행하는 기간이면 원문 그대로 옮기세요 (입찰서·제안서 제출기간은 아님).
- 배점이 없으면 score=null, 수행 기간이 어디에도 없으면 period=null.
- tasks: 과업 지시서·과업 내용·과업 범위에 적힌 주요 과업을 3-8개, 항목마다 file·page·quote를 붙이세요. 원문에 없는 추측은 쓰지 마세요.

[공고] {name}
[청크]
{chunks}"""


def _invoke(schema, prompt: str):
    llm = get_llm()
    if llm is None:
        return None
    structured = llm.with_structured_output(schema).with_config(run_name=f"llm:{schema.__name__}")  # LangSmith 표시 이름
    task = LLM_TASKS.get(schema.__name__, schema.__name__)
    with activity.step("LLM", _llm_start(task)) as done:
        for attempt in range(MAX_RETRY + 1):
            try:
                result = structured.invoke(prompt)
                done["message"] = f"응답 완료 · {task}"
                return result
            except Exception as e:  # 파싱·검증 실패 → 재시도
                log.warning("LLM 호출 실패 (%d/%d): %s", attempt + 1, MAX_RETRY + 1, e)
                activity.note("LLM", f"호출 실패 ({attempt + 1}/{MAX_RETRY + 1}) · {task}", "warn")
        done |= {"message": f"최종 실패 · {task} → 대체 처리", "level": "error"}
    return None


def _llm_start(task: str) -> str:
    return f"호출 중 · {task} ({os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'LLM')})"


def _chunks_text(chunks: list[dict]) -> str:
    return "\n".join(f"- ({c['file']} p.{c['page']})" + (f" [{c['section']}]" if c.get("section") else "")
                     + f" {c['text']}" for c in chunks)


def rag_rank(company: dict, candidates: list[dict], chunks_by_tender: dict[str, list[dict]],
             tenders: dict[str, dict], critique: str | None = None) -> list[dict] | None:
    """critique: 검증(5단계) 불통과 시 비평 프롬프트 (critique_prompt). 순위 프롬프트 뒤에 붙여 다시 순위를 매긴다."""
    fields = ("bidNtceNm", "pubPrcrmntLrgClsfcNm", "pubPrcrmntClsfcNo", "pubPrcrmntClsfcNm", "asignBdgtAmt",
              "cntrctCnclsMthdNm", "bidPrtcptLmtYn", "rgnLmtBidLocplcJdgmBssNm")
    blocks = []
    for c in candidates:
        t = tenders[c["bidNtceNo"]]
        blocks.append(json.dumps({"bidNtceNo": c["bidNtceNo"], "rule_score": c["soft_score"],
                                  "fields": {f: t.get(f, "") for f in fields}}, ensure_ascii=False)
                      + "\n청크:\n" + (_chunks_text(chunks_by_tender.get(c["bidNtceNo"], [])) or "(PDF 근거 없음)"))
    result = _invoke(RankResult, RANK_PROMPT.format(quote_rule=QUOTE_RULE, company=json.dumps(company, ensure_ascii=False),
                                                    candidates="\n\n".join(blocks)) + (critique or ""))
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


def _master_profile(company: dict) -> str:
    return json.dumps({k: company.get(k) for k in ("업종", "역량키워드", "보유인력", "과거실적")}, ensure_ascii=False)


def master_candidates(company: dict, answers: list[str]) -> list[dict] | None:
    result = _invoke(MasterDraft, MASTER_PROMPT.format(rules=MASTER_RULES, company=_master_profile(company),
                                                       answers="\n".join(answers) or "(고른 답 없음)"))
    return [c.model_dump() for c in result.candidates] if result else None


def master_search_prompt(company: dict, key: str, answers: list[str]) -> str | None:
    result = _invoke(SearchQuery, MASTER_SEARCH_PROMPT.format(rules=MASTER_RULES, key=key, company=_master_profile(company),
                                                              answers="\n".join(answers) or "(고른 답 없음)"))
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
    got = {i.bidNtceNo for i in result.items}
    missing = [i for i in ids if i not in got]
    if len(missing) * 2 > len(ids):  # 절반 넘게 빠지면 결과 전체를 믿지 않는다
        return None, "형식 오류: 재평가 누락"
    if len(result.order) != len(ids) or set(result.order) != set(ids):
        # 재평가(items)는 전부 있는데 순위 목록만 누락·중복 → 버리지 않고 보정 (실측 C002: 같은 실수로 재시도 2회 모두 실패)
        # 중복은 처음 나온 자리만, 빠진 공고는 원순위대로 뒤에 붙인다
        seen = list(dict.fromkeys(result.order))
        log.warning("검증 순위 보정: 누락 %d건·중복 %d건", len(set(ids) - set(seen)), len(result.order) - len(seen))
        result.order = seen + [i for i in ids if i not in seen]
    out = result.model_dump()
    if missing:  # 일부만 빠지면 받은 재평가는 쓰고, 빠진 공고는 '검증 안 됨'으로 남긴다 (실측 C002)
        log.warning("검증 재평가 누락 %d건 → 해당 공고는 검증 없이 유지", len(missing))
        out["unchecked"] = missing
    return out, None


VERIFY_FIELDS = ("cntrctCnclsMthdNm", "budget", "rgnLmtBidLocplcJdgmBssNm", "bidPrtcptLmtYn", "pubPrcrmntClsfcNo",
                 "bidNtceDt", "bidClseDt", "dminsttNm")


def verify_ranking(*args, **kwargs) -> tuple[dict | None, str | None]:
    task = LLM_TASKS["VerifyResult"]
    with activity.step("LLM", _llm_start(task)) as done:
        out = _verify_ranking(*args, **kwargs)
        done |= ({"message": f"응답 완료 · {task}"} if out[0] is not None
                 else {"message": f"실패 · {task} ({out[1]})", "level": "error"})
    return out


def _verify_ranking(company: dict, candidates: list[dict], tenders: dict[str, dict],
                   chunks_by_tender: dict[str, list[dict]] | None = None) -> tuple[dict | None, str | None]:
    """검증 서브에이전트. 출력 확인에 실패하면 1회만 재시도 — 그래도 실패하면 호출부가 원순위를 유지한다.

    추천 단계와 다른 근거(참가 조건 필드·자격 청크)를 줘서, 앞 단계가 놓친 자격 문제를 찾게 한다.
    """
    llm = get_llm()
    if llm is None:
        return None, "LLM 키 없음"
    chunks_by_tender = chunks_by_tender or {}
    blocks = []
    for c in candidates:
        t = tenders[c["bidNtceNo"]]
        blocks.append(json.dumps({"bidNtceNo": c["bidNtceNo"], "rank": c["rank"], "bidNtceNm": c["bidNtceNm"],
                                  "rule_score": c["soft_score"], "reasons": [r["text"] for r in c.get("reasons", [])],
                                  "qualification_flags": c.get("qualification_flags", []),
                                  "fields": {f: t.get(f, "") for f in VERIFY_FIELDS}}, ensure_ascii=False)
                      + "\n[자격 청크]\n" + (_chunks_text(chunks_by_tender.get(c["bidNtceNo"], [])) or "(PDF 근거 없음)"))
    prompt = VERIFY_PROMPT.format(quote_rule=QUOTE_RULE, company=json.dumps(company, ensure_ascii=False), candidates="\n\n".join(blocks))
    ids = [c["bidNtceNo"] for c in candidates]
    for attempt in range(2):
        try:
            result = llm.with_structured_output(VerifyResult).with_config(run_name="llm:VerifyResult").invoke(prompt)
        except (ValidationError, OutputParserException) as e:
            out = None, f"형식 오류: {type(e).__name__}"
        except Exception as e:  # 네트워크·타임아웃 등은 재시도하지 않는다
            log.warning("검증 서브에이전트 예외: %s", e)
            return None, f"예외: {type(e).__name__}"
        else:
            out = check_verify(result, ids)
        if out[0] is not None:
            return out
        log.warning("검증 출력 확인 실패 (%d/2): %s", attempt + 1, out[1])
    return out


def extract_checklist(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(ChecklistResult, CHECKLIST_PROMPT.format(quote_rule=QUOTE_RULE, name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result else None


def extract_criteria(tender: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    result = _invoke(CriteriaResult, CRITERIA_PROMPT.format(quote_rule=QUOTE_RULE, name=tender.get("bidNtceNm"), chunks=_chunks_text(chunks)))
    return result.model_dump() if result and result.items else None


# ---------- 제안서 초안 섹션 작성 (SC-002) ----------
class Cite(BaseModel):
    file: str = Field(description="업체 문서 파일명 (청크 앞 표시의 값)")
    quote: str = Field(description="업체 문서 청크 본문 그대로 인용 (짧게)")


class Draft(BaseModel):
    text: str = Field(description="제안서 문장 2-4개 (개조식 문체)")
    cites: list[Cite] = Field([], description="업체 문서 근거. 일반론으로 쓴 문장이면 비움")


class UnderstandingDraft(BaseModel):
    strategy: Draft = Field(description="사업 목적 이해와 추진 전략")


class CompanyDraft(BaseModel):
    certifications: Draft | None = Field(None, description="보유 인증·면허·등록 나열. 업체 문서·프로필에 없으면 null")
    vision: Draft | None = Field(None, description="비전·차별화 포인트")


class SubItemDraft(BaseModel):
    name: str = Field(description="세부 평가항목 이름 (입력 그대로)")
    text: Draft


class ItemDraft(BaseModel):
    plan: Draft = Field(description="이 평가항목에 대한 수행 방안")
    evidence: Draft | None = Field(None, description="이 평가항목을 뒷받침하는 업체 근거(실적·인력·인증). 업체 문서에 없으면 null")
    sub_items: list[SubItemDraft] = Field([], description="세부 평가항목마다 대응 내용 (입력에 세부항목이 없으면 비움)")


class Phase(BaseModel):
    step: Literal["착수", "준비·기획", "수행", "결과 보고"]
    content: str = Field(description="주요 내용 1문장")
    period: str | None = Field(None, description="기간 (예: 1개월차, 2주). 과업 기간을 모르면 null")


class Staffing(BaseModel):
    role: str = Field(description="보유 인력의 역할명 (입력 그대로)")
    count: int | None = Field(None, description="투입 인원. 보유 인원을 넘지 않게")
    duty: str = Field(description="담당 업무 1문장")


class PlanDraft(BaseModel):
    phases: list[Phase] = Field(description="착수·준비·기획·수행·결과 보고 4단계")
    staffing: list[Staffing] = Field([], description="보유 인력 역할별 투입 계획")
    pm: Draft | None = Field(None, description="총괄 책임자(PM) 이름·경력. 업체 문서에 없으면 null")


PROPOSAL_COMMON = """당신은 공공 입찰 제안서 작성을 돕는 컨설턴트입니다. 아래 공고 정보와 업체 정보만 보고 제안서 초안을 쓰세요.
- 업체에 관한 사실(인증·면허명, 사람 이름, 경력 연수, 실적 금액·수치, 장비)은 [업체 프로필]·[업체 문서]에 있는 것만 쓰세요. 없는 사실은 절대 지어내지 마세요.
- 업체 문서에 근거한 문장이면 cites에 file과 quote를 붙이세요. {quote_rule}
- 근거 없이 일반적인 수행 방법으로 쓴 문장은 cites를 비우세요 (화면에 'AI 초안·검토 필요'로 표시됩니다).
- 공고서 제안서 문체(~함, ~임, ~제시)로 간결하게 쓰세요.

[공고] {name} / 수요기관: {agency} / 과업 기간: {period}
[과업 범위]
{tasks}
[업체 프로필]
{company}
[업체 문서]
{docs}

[작성할 내용]
"""

UNDERSTANDING_TASK = "Ⅱ. 사업 이해 — 사업 목적 이해와 추진 전략 (과업 범위를 업체의 실적·역량과 연결해서)."
COMPANY_TASK = """Ⅲ. 회사 소개
- certifications: 업체 문서·프로필에 있는 인증·면허·등록만 나열 (없으면 null, 일반론 금지)
- vision: 비전·차별화 포인트. 이 공고와 관련된 강점 위주로"""
ITEM_TASK = """Ⅳ. 수행 방안 — 평가항목 "{item}"{score}
세부 평가항목: {sub_items}
- plan: 이 평가항목에서 높은 점수를 받기 위한 수행 방안
- evidence: 이 항목을 뒷받침하는 업체 실적·인력·인증 (업체 문서·프로필에 없으면 null)
- sub_items: 세부 평가항목마다 대응 내용 1-2문장 (name은 위 목록 그대로)"""
PLAN_TASK = """Ⅴ. 추진 일정·Ⅵ. 조직 구성
- phases: 착수 / 준비·기획 / 수행 / 결과 보고 4단계의 주요 내용과 기간 (과업 기간 안에서 나눠서)
- staffing: 보유 인력 {staff} 의 역할별 투입 인원(보유 인원 이하)과 담당 업무
- pm: 총괄 책임자(PM) 이름과 경력 (업체 문서에 없으면 null)"""


def _proposal_prompt(task: str, tender: dict, criteria: dict, company: dict, docs: list[dict]) -> str:
    tasks = "\n".join(f"- {t['text']}" for t in criteria.get("tasks") or []) or "(추출된 과업 없음 — 공고명으로 판단)"
    profile = {k: v for k, v in company.items() if k not in ("비밀번호", "유사어")}
    return PROPOSAL_COMMON.format(
        quote_rule=QUOTE_RULE, name=tender.get("bidNtceNm"), agency=tender.get("dminsttNm") or "-",
        period=criteria.get("period") or "공고서에 없음", tasks=tasks, company=json.dumps(profile, ensure_ascii=False),
        docs=_chunks_text(docs) if docs else "(업체 문서 없음 — 업체 프로필만 근거)") + task


def draft_understanding(tender: dict, criteria: dict, company: dict, docs: list[dict]) -> dict | None:
    r = _invoke(UnderstandingDraft, _proposal_prompt(UNDERSTANDING_TASK, tender, criteria, company, docs))
    return r.model_dump() if r else None


def draft_company(tender: dict, criteria: dict, company: dict, docs: list[dict]) -> dict | None:
    r = _invoke(CompanyDraft, _proposal_prompt(COMPANY_TASK, tender, criteria, company, docs))
    return r.model_dump() if r else None


def draft_item(item: dict, tender: dict, criteria: dict, company: dict, docs: list[dict]) -> dict | None:
    task = ITEM_TASK.format(item=item["item"], score=f" (배점 {item['score']:g}점)" if item.get("score") is not None else "",
                            sub_items=", ".join(item.get("sub_items") or []) or "없음")
    r = _invoke(ItemDraft, _proposal_prompt(task, tender, criteria, company, docs))
    return r.model_dump() if r else None


def draft_plan(tender: dict, criteria: dict, company: dict, docs: list[dict]) -> dict | None:
    staff = ", ".join(f"{k} {v}명" for k, v in (company.get("보유인력") or {}).items()) or "(정보 없음)"
    r = _invoke(PlanDraft, _proposal_prompt(PLAN_TASK.format(staff=staff), tender, criteria, company, docs))
    return r.model_dump() if r else None


# ---------- 제안서 대화형 보완 ----------
class SlotEdit(BaseModel):
    key: str = Field(description="바꿀 칸의 key (입력 목록 그대로)")
    text: str = Field(description="칸에 넣을 새 내용. 인원 칸(…/count)은 숫자만")
    source: Literal["user", "company_doc", "tender", "general"] = Field(
        description="user: 사용자가 대화에서 직접 알려준 사실 / company_doc: 업체 문서 근거 / tender: 공고서 근거 / general: 근거 없는 일반론")
    cites: list[Cite] = Field([], description="company_doc·tender일 때 원문 인용 (청크 본문 그대로, 짧게)")


class ChatEditResult(BaseModel):
    reply: str = Field(description="사용자에게 보낼 답변 1-3문장 (무엇을 바꾸자고 제안하는지, 못 채운 칸은 왜인지)")
    edits: list[SlotEdit] = Field([], description="바꿀 칸 목록. 바꿀 것이 없으면 비움")


CHAT_PROMPT = """당신은 공공 입찰 제안서 초안을 사용자와 함께 보완하는 컨설턴트입니다.
사용자의 요청을 읽고 제안서의 어떤 칸을 어떻게 바꿀지 제안하세요. 실제 반영은 사용자가 확인한 뒤에 합니다.
- [칸 목록]에 있는 key만 바꿀 수 있습니다. 요청과 관계없는 칸은 건드리지 마세요.
- 사용자가 대화에서 직접 알려준 사실이면 source=user. 업체 문서에서 찾았으면 source=company_doc, 공고서에서 찾았으면 source=tender로 하고 cites를 붙이세요. {quote_rule}
- 사실(인증·면허, 사람 이름, 경력, 실적 수치, 기간)은 사용자 말·업체 문서·업체 프로필·공고서에 있는 것만 쓰세요. 없으면 지어내지 말고 reply에 사용자에게 알려 달라고 요청하세요.
- 사실형 칸(인증·면허, PM, 업체 근거)은 근거가 없으면 바꾸지 마세요.
- 서술형 칸(전략·수행 방안 등)은 근거가 없어도 일반론으로 쓸 수 있고 그때는 source=general입니다.
- 공고서 제안서 문체(~함, ~임)로 간결하게 쓰세요.

[공고] {name} / 수요기관: {agency}
[업체 프로필]
{company}
[업체 문서]
{docs}
[공고서]
{tender_chunks}
[칸 목록] key | 칸 이름 | 현재 내용
{slots}
[이전 대화]
{history}
[사용자 요청]
{message}"""


def chat_edit(message: str, history: list[dict], slots: list[dict], tender: dict, company: dict,
              docs: list[dict], tender_chunks: list[dict]) -> dict | None:
    profile = {k: v for k, v in company.items() if k not in ("비밀번호", "유사어")}
    slot_lines = "\n".join(f"{s['key']} | {s['label']} | {s['current'] or '[확인 필요]'}" for s in slots)
    hist = "\n".join(f"{'사용자' if h['role'] == 'user' else 'AI'}: {h['text']}" for h in history[-6:]) or "(없음)"
    prompt = CHAT_PROMPT.format(
        quote_rule=QUOTE_RULE, name=tender.get("bidNtceNm"), agency=tender.get("dminsttNm") or "-",
        company=json.dumps(profile, ensure_ascii=False), docs=_chunks_text(docs) if docs else "(없음)",
        tender_chunks=_chunks_text(tender_chunks) if tender_chunks else "(없음)", slots=slot_lines, history=hist, message=message)
    r = _invoke(ChatEditResult, prompt)
    return r.model_dump() if r else None
