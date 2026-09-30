# Agent 페르소나 및 시스템 설계

## 1. Agent 페르소나 및 시스템 프롬프트

Agent의 정체성, 역할, 톤앤매너를 정의합니다.

### 1.1 Agent 정의

| 항목 | 정의 |
|-----|------|
| Agent 이름 | 나라장터 입찰 어시스턴트 (Proposal Assistant) |
| 주요 역할 | 입찰 담당자를 위해 공고 추천 + 의사결정 지원 및 제안서 초안 생성 담당 |
| 핵심 목표 | 1) 공고 발굴/제안서 작성 시간 70% 단축 2) 낙찰 가능성 높은 공고만 선별 제출 3) 제안서 품질 향상으로 수주확률 극대화 |
| 기본 전제 | 최종 의사결정은 반드시 담당자(사람)이 한다. Agent는 판단에 필요한 정보와 분석 근거만 제공 |
| 톤앤매너 | 전문가형 조력자: 공식적, 근거 기반, 신뢰감 중심. 항상 판단 근거를 3~4가지로 명시 |
| 제약사항 | 추측/불확실한 정보 금지, 주도권 금지, 과장된 표현 금지, 근거 제시 필수 |

### 1.2 System Prompt

당신은 나라장터 입찰을 지원하는 전문가형 의사결정 조력자입니다.

역할과 책임:
- 입찰 담당자의 공고 발굴, 선별, 제안서 작성을 돕는 조력자 역할
- 업체 역량과 공고 요구사항을 객관적으로 분석하고 연결
- 최종 의사결정은 담당자가 하도록 필요한 정보와 근거만 제공
- 낙찰 가능성 높은 공고 선별로 수주확률 극대화

의사결정 원칙:
- 항상 데이터와 객관적 근거로 판단 (추측 금지)
- 확률/점수 제시 시 계산 과정과 근거를 명확히 설명
- 최종 의사결정은 담당자에게 맡기고, 판단 근거만 제공

답변 구조:
1. 왜 이 공고를 추천하는지 3~4가지 근거 명시 (기술, 실적, 예산, 인력 등)
2. 만족하는 항목과 부족한 항목을 명확히 구분 표시
3. 공고 선택에 대한 최종 판단은 담당자가 하도록 가이드
4. 평가기준별 대응 전략을 제안서 생성 시 포함

금지사항:
- 실제 낙찰 확률 예측 (정부 입찰은 예측 불가)
- "적합하지 않습니다"처럼 일방적 거절 (부족한 자격 분석 제시)
- 최종 입찰 여부 결정 (담당자만 가능)
- 공고 신뢰성 판단 (담당자 책임)

---

## 2. 워크플로우 및 오케스트레이션

### 2.1 공고 추천 워크플로우 (SC-001)

입력: 업체 프로필 (기술, 인력, 실적) + 필터 조건 (선택)

Step 1: 데이터 로드
- 업체 프로필을 벡터 임베딩 (RAG)
- 공고 데이터 로드 (20~30개)

Step 2: 필터링
- 자격요건 체크 (경험 수 비교)
- 예산대 체크 (최소/최대)
- 지역 조건 체크 (선택시)
- 필터 탈락 공고 제외

Step 3: 유사도 계산
- 필터 통과한 공고들에 대해 코사인 유사도 계산
- 점수: 0~100 (업체 기술과 공고 요구사항 매칭도)

Step 4: 랭킹 & 추천근거 생성
- 유사도 상위 3~5개 선별
- 각 공고별로 "왜 추천하는가" 근거 3~4가지 작성
- 예: "RAG 기술 매칭", "유사 실적 5건", "예산대 충족"

Step 5: 결과 출력
- 추천 공고 Top 5 + 적합도 % + 근거 + 추천정보 (마크다운 테이블 형식)

### 2.2 제안서 생성 워크플로우 (SC-002)

입력: 선택된 공고 + 업체 프로필

Step 1: 공고 분석
- 과업 내용 추출
- 필수 자격/기술 항목 추출
- 평가기준 및 배점 추출
- 예산, 기간, 고객 요구사항 파악

Step 2: 데이터 매핑
- 공고 요구사항 <-> 업체 보유 정보 매핑
- "RAG 기술 필요" -> 업체 "RAG 3건 실적 있음" 매칭
- "AI 개발자 3명 이상" -> 업체 "AI 개발자 3명" 매칭

Step 3: 섹션별 초안 생성
- 섹션 1: 회사소개 (회사 개요, 보유 기술, 주요 실적, 주요 인력)
- 섹션 2: 수행방안 (기술 아키텍처, 공고 요구사항별 대응, 위험 대응)
- 섹션 3: 추진일정 (마일스톤별 일정, 산출물, 투입 인력)
- 섹션 4: 조직구성 (팀원 구성, 경력, 공고 필수자격 충족 여부)

Step 4: 평가기준 매핑표 생성
- 공고의 평가기준에 대해 우리의 대응 내용을 매핑

Step 5: 최종 조립 & 출력
- 마크다운 형식 제안서 초안
- 평가기준 매핑표
- 편집 가능한 에디터 UI

### 2.3 상태 관리

| 상태 | 저장 정보 | 목적 |
|------|---------|------|
| Session 상태 | 선택된 공고 ID, 업체 프로필 | SC-001 -> SC-002 연계 |
| 쿼리 히스토리 | 최근 조회한 필터, 추천결과 | "이전 조회 다시 보기" 기능 |
| 생성 초안 캐시 | 생성된 제안서 텍스트 | 담당자의 수정사항 추적 |

---

## 3. 도구 및 함수 명세 (Capability)

### 3.1 데이터 조회 도구

| 도구명 | 기능 설명 | 입력 파라미터 | 출력 |
|-------|---------|-----------|------|
| load_company_profile | 업체 프로필 로드 | company_id: str | name, established_year, technologies, certifications, employees, past_projects |
| load_all_notices | 공고 데이터 일괄 로드 | limit: int (default 30) | List[notice_id, title, budget, deadline, requirements, evaluation_criteria] |
| get_notice_detail | 특정 공고 상세정보 | notice_id: str | 공고명, 업종, 요구사항, 과업내용, 평가기준, 필요 기술 |

### 3.2 분석 도구

| 도구명 | 기능 설명 | 입력 파라미터 | 출력 |
|-------|---------|-----------|------|
| calculate_similarity | 공고와 업체 유사도 계산 | company_profile: dict, notice_list: list | List[notice_id, similarity_score (0~100), reasons] |
| filter_notices | 필터 조건에 따른 공고 필터링 | budget_range: tuple, regions: list, skills: list | List[notice_ids] |
| match_qualifications | 업체 자격요건 매칭 분석 | company: dict, requirements: list | matched[], missing[] |

### 3.3 생성 도구

| 도구명 | 기능 설명 | 입력 파라미터 | 출력 |
|-------|---------|-----------|------|
| generate_proposal_section | 제안서 특정 섹션 생성 | section_type: str, company: dict, notice: dict | Generated markdown text (한국어) |
| generate_matching_table | 평가기준 매핑표 생성 | notice_criteria: dict, company_strengths: dict | Markdown 테이블 형식 |
| suggest_improvements | 평가기준별 개선안 제시 | current_proposal: str, evaluation_criteria: dict | List[criterion, current_point, suggestion] |

### 3.4 저장/다운로드 도구

| 도구명 | 기능 설명 | 입력 파라미터 | 출력 |
|-------|---------|-----------|------|
| save_proposal_draft | 제안서 초안 저장 | proposal_md: str, notice_id: str | draft_id, saved_at, file_path |
| export_proposal | 제안서 형식 변환 | draft_id: str, format: str ("docx"/"pdf"/"md") | File URL (다운로드 링크) |

---

## 4. 지식 베이스 및 메모리 전략

### 4.1 RAG (검색 증강 생성) 전략

참조 데이터 소스:
- 공고 데이터 (JSON): 20~30개 더미 공고
- 업체 프로필 (JSON): 5개 가상 회사
- 나라장터 입찰 가이드 문서 (참고)

청킹(Chunking) 방식:
- 공고: 섹션별 (공고명, 과업내용, 필수자격, 평가기준)
- 업체: 정보 영역별 (회사정보, 기술보유, 실적, 인력)

임베딩 모델: Claude의 내장 텍스트 분석 능력 활용

Vector DB: 초기 PoC는 JSON 기반 메모리 활용, 추후 확장시 Pinecone 또는 Chroma 고려

### 4.2 대화 메모리 (Conversation History)

메모리 유형: 윈도우 버퍼 (최근 10개 턴의 대화만 유지)

저장 전략:
- 세션 시작: 업체 프로필 로드 (기본 컨텍스트)
- 공고 추천: 조회 조건 + 결과 저장
- 제안서 생성: 공고 ID + 생성 초안 버전 관리
- 세션 종료: 히스토리 삭제 (선택사항: 담당자 요청시만 저장)

---

## 5. 핵심 Agent 기술 스택

| 구분 | 선정 기술 | 선정 사유 |
|------|---------|---------|
| LLM Model | Claude 3.5 Sonnet (Anthropic) | 한국어 처리 우수, 긴 문맥 처리 (200K 토큰), 구조화된 출력 안정성, 비용 효율성 |
| Agent Framework | LangGraph (LangChain) | 명확한 워크플로우 제어 (Node/Edge), Tool Calling 안정성, State 관리 용이 |
| Prompt Strategy | Chain-of-Thought (CoT) + Few-Shot | CoT: 추천근거 단계별 설명 / Few-Shot: 제안서 생성 시 예시 제시 |
| Output Parsing | Claude JSON Mode | 구조화된 응답 강제, Pydantic 모델로 검증 |
| Monitoring | Python logging + Langfuse (선택사항) | Tool Call 로깅, Token 사용량 추적, 오류 패턴 분석 |

---

## 6. 에러 처리 및 Fallback 전략

| 시나리오 | 처리 방식 |
|---------|---------|
| 공고 데이터 로드 실패 | "데이터를 불러올 수 없습니다. 관리자에 문의하세요" |
| 유사도 계산 오류 | 대체: 기본 필터링만으로 공고 제시 |
| 제안서 생성 중단 | 생성된 부분까지만 저장 + "다시 생성" 옵션 제공 |
| 사용자 입력 부족 | "필수 정보가 빠졌습니다. [필드명]을 입력해주세요" |

---

## 7. PoC 범위 & 기술 로드맵

PoC 단계 (현재, 3~4주):
- 더미 데이터 기반 공고 추천 엔진
- 기본 제안서 초안 생성 (4개 섹션)
- Streamlit 기반 간단한 UI
- 평가기준 매핑표 자동 생성

Phase 1 (추후, 2~3개월):
- 실제 나라장터 OpenAPI 연동
- 더 정교한 유사도 계산 (Vector DB)
- 사용자 피드백 학습 (선호도 기반 추천)

Phase 2 (향후, 6개월~):
- 낙찰가 예측 모델
- 모바일 앱 개발
- 프로덕션 보안 강화

---

## 8. Claude API 호출 예시

### 예시 1: 공고 추천 생성

from anthropic import Anthropic

client = Anthropic()

system_prompt = """
당신은 나라장터 입찰을 지원하는 전문가형 의사결정 조력자입니다.
[위의 System Prompt 내용 삽입]
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    system=system_prompt,
    messages=[
        {
            "role": "user",
            "content": f"""
다음 업체 정보를 바탕으로 공고 20개 중 상위 3~5개를 추천해주세요.

[업체 프로필]
회사명: AI 데이터 스타트업
기술: RAG, LLM, 벡터DB
인력: AI개발자 3명
실적: 5건

[공고 리스트]
{공고_데이터_json}

각 공고별로 왜 추천하는지 3가지 근거를 명확히 제시하세요.
"""
        }
    ]
)

print(response.content[0].text)

### 예시 2: 제안서 초안 생성

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    system=system_prompt,
    messages=[
        {
            "role": "user",
            "content": f"""
다음 공고에 대해 제안서 초안을 생성해주세요.
마크다운 형식으로 다음 4개 섹션을 포함하세요:
1. 회사소개
2. 수행방안
3. 추진일정
4. 조직구성

[공고 정보]
{공고_상세_json}

[업체 정보]
{업체_프로필_json}

각 섹션은 300~500자 분량으로 구체적이고 설득력 있게 작성하세요.
"""
        }
    ]
)

print(response.content[0].text)

---

작성일: 2026-10-01
