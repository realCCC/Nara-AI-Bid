# Agent 페르소나 및 시스템 설계

task2(문제 정의)와 task3(시나리오)에서 확정한 목표·흐름을 Agent 구조, 도구, 상태, 프롬프트 수준으로 구체화한 문서입니다.

### 기준 수치 (task2·task3와 동일)

| 항목 | 기준 | 근거 |
|------|------|------|
| 업무 시간 단축 | SC-001 + SC-002 합산 50% 이상 | task2 KPI 목표, Phase 4 실측 |
| 추천 정확도 | 상위 3건 중 '추천' 라벨 공고 1건 이상 포함 비율 90% 이상 | task2 KPI 목표 |
| 초안 커버율 | 공고 필수 항목 반영률 80% 이상 | task2 KPI 목표 |
| 초안 근거성 | 기업 프로필에 없는 사실 기재 0건 | task2 KPI 목표 |
| 정답셋 | IT 용역 공고 30건 × 가상 기업 10개사 = 300조합, 3단계 라벨(부적격 / 적격-비추천 / 추천) (초기 5개사로 시작) | 8주 안에 수작업 라벨링 가능한 규모, 라벨 기준은 task2 '적합 공고 정의' |
| 커버율 측정 대상 | 초안 30건 (공고별 추천 1순위 기업) | task3 KPI 1 |
| PoC 기간 | 8주 (하루 1~2시간) | 프로젝트 조건 |

---

## 1. Agent 페르소나 및 시스템 프롬프트

### 1.1 Agent 정의

| 항목 | 정의 |
|-----|------|
| Agent 이름 | 나라장터 입찰 어시스턴트 (Proposal Assistant) |
| 주요 역할 | 입찰 담당자를 위한 공고 추천·의사결정 지원 및 제안서 초안 생성 |
| 핵심 목표 | 1) 공고 검토 + 제안서 작성 시간 50% 이상 단축 2) 추천 근거 제시로 객관적 공고 선별 지원 3) 필수 항목 80% 이상 반영한 초안으로 제안서 품질 표준화 |
| 기본 전제 | 최종 의사결정(공고 선택, 제안서 확정)은 반드시 담당자가 한다. Agent는 판단에 필요한 정보와 분석 근거만 제공 |
| 톤앤매너 | 전문가형 조력자: 공식적, 근거 기반, 신뢰감 중심. 추천 시 판단 근거를 3-4가지로 명시 |
| 제약사항 | 추측·불확실한 정보 금지, 의사결정 주도 금지, 과장된 표현 금지, 근거 제시 필수 |

### 1.2 System Prompt (공통)

```text
당신은 나라장터 입찰을 지원하는 전문가형 의사결정 조력자입니다.

[역할과 책임]
- 입찰 담당자의 공고 검토, 선별, 제안서 작성을 돕는 조력자 역할
- 업체 역량과 공고 요구사항을 객관적으로 분석하고 연결
- 최종 의사결정은 담당자가 하도록 필요한 정보와 근거만 제공

[의사결정 원칙]
- 항상 입력된 데이터와 객관적 근거로 판단 (추측 금지)
- 점수 제시 시 계산 항목(기술/경험/인력)과 근거를 함께 설명
- 입력 데이터에 없는 정보는 "확인 필요"로 표시
- 수치를 쓸 때는 출처(입력 데이터의 어느 항목인지)를 함께 표시

[답변 구조]
1. 추천 이유 3-4가지 명시 (기술, 실적, 예산, 인력 등)
2. 충족 항목과 부족 항목을 명확히 구분 표시
3. 최종 판단은 담당자가 하도록 안내
4. 제안서 생성 시 평가기준별 대응 전략 포함

[금지사항]
- 실제 낙찰 확률·평가 점수 예측 (정부 입찰 결과는 예측 불가)
- "적합하지 않습니다"처럼 일방적 거절 (대신 부족한 자격 분석 제시)
- 최종 입찰 여부 결정 (담당자만 가능)
- 공고 신뢰성 판단 (담당자 책임)
```

### 1.3 Agent별 역할 프롬프트 요약

공통 System Prompt 위에 Agent별 역할 지시를 추가합니다.

| Agent | 역할 지시 요약 | 출력 형식 |
|-------|-------------|---------|
| Supervisor | 요청을 하위 작업으로 분해하고 Agent 호출 순서·병렬 여부 결정, 결과 취합 | 다음 실행 노드 이름 |
| 공고 분석 Agent | 공고 원문에서 과업내용·필수자격·평가기준·예산·기간 추출 | `TenderInfo` (JSON) |
| 매칭 Agent | 기업 정보와 공고 요구사항 비교, 적합도 점수와 근거 생성 | `MatchResult` (JSON) |
| 제안서 작성 Agent (Sub-Agent ×4) | 회사소개 / 수행방안 / 추진일정 / 조직구성 섹션별 초안 작성 | HTML 조각 (`<section>` 1개) |

**HTML 출력 지시 (제안서 작성 Agent 공통)**

```text
- <section id="{섹션ID}"> 하나만 출력하고, 그 밖의 텍스트는 쓰지 않는다.
- 허용 태그: section, h2, h3, p, ul, ol, li, table, thead, tbody, tr, th, td, strong, em, br
- script, style, iframe, img, a, on* 속성, 인라인 style 속성은 쓰지 않는다.
- 입력 데이터에 없는 수치는 쓰지 않고 "확인 필요"로 표시한다.
```

---

## 2. Agent 아키텍처 및 오케스트레이션

### 2.1 전체 구조 (Supervisor 패턴)

```
    React 화면  ──(LangGraph SDK: 실행 · 스트리밍 · 재개)──▶  LangGraph Server
                                                                 │
                                                          ┌──────┴───────┐
                                                          │  Supervisor  │
                                                          └──────┬───────┘
                                       ┌─────────────────────────┼──────────────────┐
                                       ▼                         ▼                  ▼
                                ┌─────────────┐           ┌─────────────┐   ┌──────────────────┐
                                │ 공고 분석   │           │  매칭       │   │ 제안서 작성       │
                                │ Agent       │           │  Agent      │   │ Agent (HTML)      │
                                └─────────────┘           └─────────────┘   │ ├ Sub1 회사소개   │
                                                                            │ ├ Sub2 수행방안   │
                                                                            │ ├ Sub3 추진일정   │
                                                                            │ └ Sub4 조직구성   │
                                                                            └──────────────────┘
                                                 [HITL ①] 공고 선택     [HITL ②] 제안서 확정
```

| 구성 요소 | PoC 범위 | 비고 |
|---------|---------|------|
| Supervisor | **필수** | LangGraph 조건 분기로 흐름 제어 |
| 공고 분석 Agent | **필수** | SC-001, SC-002 공통 사용 |
| 매칭 Agent | **필수** | SC-001 추천, SC-002 평가기준 대응표 생성 |
| 제안서 작성 Agent (Sub-Agent 4개 병렬) | **필수** | SC-002, HTML 출력 |
| Human-in-the-loop 승인 2곳 | **필수** | `interrupt` → React에서 응답 → `Command(resume=...)`로 재개 |
| 검증 Agent (자기 검증·재작성 루프) | **확장 (2차 목표)** | task2 기능 4. 재작성 최대 2회 상한 (근거: task2 보조 지표) |

### 2.2 공고 추천 워크플로우 (SC-001)

| 단계 | 담당 | 처리 내용 |
|------|-----|---------|
| 1. 입력 | 담당자 + System | React에서 기업 프로필 JSON 업로드, 스키마 검증 |
| 2. 공고 분석 | 공고 분석 Agent | 공고 30건 병렬 파싱 → 과업내용·필수자격·평가기준 구조화 |
| 3. 필터링 | 매칭 Agent (규칙) | 필수 자격 미충족, 예산 범위 밖, 지역 조건 불일치 공고 제외 |
| 4. 적합도 계산 | 매칭 Agent | 하이브리드 점수 계산 (2.4절) → 상위 5건 + 추천 근거 3-4가지 |
| 5. **[HITL ①] 공고 선택** | 담당자 | React 화면에서 상위 5건 검토 후 1건 선택 → 승인 |
| 6. 저장 및 전환 | System | 선택 결과를 같은 thread의 State에 저장, SC-002로 진행 |

### 2.3 제안서 생성 워크플로우 (SC-002)

| 단계 | 담당 | 처리 내용 |
|------|-----|---------|
| 1. 입력 | System | 선택 공고 + 기업 프로필 State에서 자동 로드 |
| 2. 섹션별 초안 생성 | 제안서 작성 Agent (Sub-Agent 4개 병렬) | 회사소개 / 수행방안 / 추진일정 / 조직구성 HTML 생성 → 구조 검증 → Supervisor가 취합 |
| 3. 평가기준 대응표 | 매칭 Agent | 공고 평가기준 ↔ 초안 내용 매핑, 항목별 강점·보완점 HTML 표 |
| 4. **[HITL ②] 검토 및 확정** | 담당자 | React 화면에서 sanitize된 초안 확인, 섹션 수정·재생성 후 "제안서 완성" 승인 |
| 5. 최종 저장 | System | 섹션 + 대응표를 단일 HTML 문서로 조립, 다운로드 제공, checkpoint로 버전 기록 |

### 2.4 적합도 점수 계산 방식 (하이브리드)

task2 '적합 공고 정의'의 세 단계를 그대로 처리 순서로 사용합니다: **입찰 자격(Eligibility)** 은 규칙 필터, **기업 적합도(Suitability)** 는 아래 점수, **추천 순위(Ranking)** 는 점수 내림차순 상위 5건. 적합도 점수는 task3에서 정의한 가중치(기술 40% + 경험 40% + 인력 20%)를 유지하되, 항목 특성에 따라 계산 방식을 나눕니다.

| 항목 | 가중치 | 계산 방식 | 근거 |
|------|------|---------|------|
| 기술 적합도 | 40% | BGE-M3 임베딩 코사인 유사도 (기업 보유 기술 ↔ 공고 요구 기술) | 표현이 달라도 의미가 같은 기술 매칭 (예: "벡터 검색" ↔ "벡터DB") |
| 경험 적합도 | 40% | BGE-M3 임베딩 코사인 유사도 (기업 과거 실적 ↔ 공고 과업내용) | 유사 프로젝트 경험을 의미 단위로 비교 |
| 인력 적합도 | 20% | 규칙 기반: 요구 인원 대비 보유 인원 충족 비율 | 숫자 비교라 규칙이 더 정확 |
| 필수 자격 | 필터 | 규칙 기반: 하나라도 미충족 시 추천 제외 (부족 항목은 별도 표시) | 자격 미달 공고는 점수와 무관하게 제외 |

**최종 점수** = (기술 × 0.4 + 경험 × 0.4 + 인력 × 0.2) × 100 → 0-100점  
(가중치 근거: 팀 설정 초기값, Phase 4 정답셋 결과로 조정)

### 2.5 상태(State) 관리

LangGraph의 공유 State 하나로 SC-001과 SC-002를 연결합니다. LangGraph Server의 **thread** 1개가 담당자 작업 세션 1개에 대응합니다.

| State 필드 | 내용 | 생성 시점 | 사용처 |
|-----------|------|---------|------|
| `company_profile` | 기업 프로필 (파싱 완료) | SC-001 1단계 | 전 단계 |
| `tenders` | 구조화된 공고 목록 | SC-001 2단계 | 매칭, 제안서 작성 |
| `recommendations` | 상위 5건 + 점수 + 근거 | SC-001 4단계 | HITL ① |
| `selected_tender_id` | 담당자가 선택한 공고 ID | HITL ① | SC-002 전체 |
| `draft_sections` | 섹션별 HTML 4개 (`{section_id: html}`) | SC-002 2단계 | 취합, HITL ② |
| `criteria_table` | 평가기준 대응표 (HTML 표) | SC-002 3단계 | HITL ② |
| `proposal_html` | 최종 조립된 HTML 문서 | SC-002 5단계 | 다운로드 |
| `approval_status` | 승인 대기 / 승인 완료 | HITL ①, ② | 흐름 재개 판단 |

**Checkpoint**: LangGraph Server가 thread별 checkpoint를 자동 저장하므로, 담당자가 화면을 닫았다가 돌아와도 HITL 대기 지점부터 이어서 실행합니다. 그래프 코드에서 checkpointer를 따로 지정하지 않습니다.

---

## 3. 도구 및 함수 명세 (Capability)

task3 부록 F의 모듈 매핑과 이름을 맞췄습니다.

### 3.1 데이터 조회 도구

| 도구명 | 기능 설명 | 입력 | 출력 | 연결 모듈 |
|-------|---------|-----|------|---------|
| `load_company_profile` | 기업 프로필 로드 및 스키마 검증 | `file: JSON` | `CompanyProfile` | FileUploadModule |
| `load_tenders` | 공고 데이터 일괄 로드 | `limit: int = 30` | `List[Tender]` | FileUploadModule |
| `get_tender_detail` | 특정 공고 상세정보 조회 | `tender_id: str` | `Tender` | TenderAnalysisAgent |

### 3.2 분석 도구

| 도구명 | 기능 설명 | 입력 | 출력 | 연결 모듈 |
|-------|---------|-----|------|---------|
| `filter_tenders` | 필수 자격·예산·지역 조건 필터링 | `company, tenders, filters` | `List[tender_id]`, 제외 사유 | FilterModule |
| `embed_texts` | BGE-M3 임베딩 생성 | `texts: List[str]` | `List[vector]` | EmbeddingModule |
| `calculate_match_score` | 하이브리드 적합도 계산 (2.4절) | `company, tender` | `score(0-100)`, 항목별 점수, 근거 | SimilarityModule |
| `match_qualifications` | 필수 자격 충족 여부 분석 (Eligibility) | `company, requirements` | `matched[]`, `missing[]` | SimilarityModule |
| `check_groundedness` | 초안 사실 항목(실적명·금액·기간·인원·자격)을 기업 프로필과 대조 (근거성 KPI 측정용) | `proposal_html, company` | 불일치 항목 목록 | 평가 스크립트 |

### 3.3 생성 도구

| 도구명 | 기능 설명 | 입력 | 출력 | 연결 모듈 |
|-------|---------|-----|------|---------|
| `generate_section` | 제안서 특정 섹션 생성 | `section_type, company, tender` | HTML `<section>` | SectionGeneratorAgent |
| `validate_section_html` | 허용 태그·섹션 구조 검증 | `html: str` | `ok: bool`, 위반 항목 | SectionGeneratorAgent |
| `generate_criteria_table` | 평가기준 대응표 생성 | `tender_criteria, draft_sections` | HTML `<table>` | SimilarityModule |
| `assemble_proposal` | 섹션 4개 + 대응표를 단일 HTML 문서로 조립 (인쇄용 CSS 포함) | `draft_sections, criteria_table` | 최종 HTML 문서 | ProposalAssemblyModule |

### 3.4 저장/내보내기 도구

| 도구명 | 기능 설명 | 입력 | 출력 | PoC 범위 |
|-------|---------|-----|------|---------|
| `save_draft` | 초안 저장 및 버전 기록 | `proposal_html, tender_id` | `draft_id, saved_at, file_path` | **필수** |
| `export_html` | HTML 파일 다운로드 | `draft_id` | `.html` 파일 | **필수** |
| `export_document` | PDF/Word 변환 | `draft_id, format` | `.pdf` / `.docx` 파일 | 확장 |

### 3.5 도구 호출 규약 (하네스)

| 규칙 | 내용 |
|------|------|
| 출력 스키마 검증 | JSON 출력은 Pydantic 모델로 검증 (`TenderInfo`, `MatchResult` 등), HTML 출력은 `validate_section_html`로 검증 |
| 재시도 제한 | 검증 실패 시 최대 2회 재시도, 이후 폴백 경로 실행 (근거: 비용·지연 상한으로 팀 설정) |
| 단계 상한 | Supervisor 루프 최대 10단계 (근거: 전체 노드 6개 + 재시도 여유분, 무한 루프 방지) |
| 화면 렌더링 | React는 서버 HTML을 DOMPurify로 한 번 더 sanitize한 뒤 표시 (서버 검증과 이중 방어) |
| 로깅 | 도구 호출, 입력/출력 요약, 토큰 사용량을 단계별로 기록 |

---

## 4. 지식 베이스 및 메모리 전략

### 4.1 RAG (검색 증강 생성) 전략

RAG는 목표 자체가 아니라 Agent가 호출하는 도구 중 하나로 구성합니다 (task1 하네스 원칙).

| 항목 | 결정 |
|------|------|
| 데이터 소스 | IT 용역 공고 30건 (조달청 API 수집 + 수동 보완), 가상 기업 프로필 10개사 |
| 청킹 방식 | 공고: 섹션별 (공고명 / 과업내용 / 필수자격 / 평가기준), 기업: 영역별 (회사정보 / 기술 / 실적 / 인력) |
| 임베딩 모델 | BGE-M3 (오픈소스, 한국어 지원) |
| Vector DB | Chroma (로컬 실행) |
| 활용처 | 매칭 Agent의 기술·경험 유사도 계산, 제안서 작성 시 유사 실적 검색 |

### 4.2 대화 메모리

| 항목 | 결정 |
|------|------|
| 메모리 유형 | LangGraph State + Server 내장 checkpoint (대화 이력이 아닌 작업 상태 중심) |
| 세션 단위 | LangGraph Server thread 1개 = 담당자 작업 1건 (SC-001 → SC-002) |
| 세션 시작 | 기업 프로필 로드 (기본 컨텍스트) |
| SC-001 | 추천 결과, 선택 공고 저장 |
| SC-002 | 섹션별 HTML 초안, 담당자 수정 버전 기록 |
| 세션 종료 | 최종 HTML 제안서와 선택 이력만 보존, 나머지 중간 상태 삭제 |
| 개인정보 | 발주처 담당자 연락처 등 PII는 저장하지 않거나 마스킹 (task2 제약사항) |

---

## 5. 핵심 Agent 기술 스택

| 구분 | 선정 기술 | 선정 사유 |
|------|---------|---------|
| LLM | 생성형 LLM API (프로젝트 리소스 확정 후 지정) | 한국어 처리, 긴 문서 처리, 구조화된 출력 안정성 |
| Agent Framework | LangGraph (+ LangChain) | Supervisor 패턴, 조건 분기, 병렬 실행, interrupt 기반 HITL |
| 백엔드 서버 | LangGraph Server (로컬: `langgraph dev`) | 그래프를 API로 바로 노출, thread·checkpoint·스트리밍 내장 → 별도 API 서버 코드 불필요 |
| 프론트엔드 | React (Vite) + LangGraph JS SDK | 실행·진행 상태 스트리밍·HITL 재개를 SDK로 호출 |
| HTML 보안 | DOMPurify + 서버 측 허용 태그 검증 | LLM 생성 HTML의 XSS 방지 |
| 임베딩 | BGE-M3 | 한국어 지원, 무료 오픈소스 |
| Vector DB | Chroma | 로컬 실행, 예제 풍부 |
| Prompt Strategy | Chain-of-Thought + Few-Shot | CoT: 추천 근거 단계별 설명 / Few-Shot: 섹션별 HTML 예시 제공 |
| Output Parsing | 구조화 출력 + Pydantic 검증 (JSON), 태그 화이트리스트 검증 (HTML) | 스키마 강제, 실패 시 재시도 |
| 실행환경 | Docker | 개발 환경 통일 (LangGraph Server Docker 배포 조건은 Phase 1에 확인) |
| Monitoring | Python logging (+ LangSmith/Langfuse 선택) | 단계별 실행 트레이스, 토큰 비용, 실패 Agent 추적 |

---

## 6. 에러 처리 및 Fallback 전략

| 시나리오 | 처리 방식 | 사용자 메시지 |
|---------|---------|-------------|
| 기업 프로필 JSON 스키마 오류 | 업로드 단계에서 차단, 누락 필드 표시 | "필수 정보가 빠졌습니다. [필드명]을 입력해주세요" |
| 공고 데이터 로드 실패 | 재시도 1회 후 중단 | "공고 데이터를 불러올 수 없습니다. 관리자에게 문의하세요" |
| 임베딩·유사도 계산 오류 | 규칙 기반 점수(키워드 교집합)로 대체 계산 | "간이 방식으로 계산된 점수입니다" 표시 |
| Agent JSON 출력 검증 실패 | 최대 2회 재시도 → 실패 시 해당 항목 "확인 필요"로 표시 | 실패 항목 하이라이트 |
| 섹션 HTML 검증 실패 (금지 태그, 구조 깨짐) | 최대 2회 재생성 → 실패 시 해당 섹션만 "다시 생성" 버튼 제공 | "[섹션명] 생성에 실패했습니다. 다시 생성하시겠습니까?" |
| 섹션 생성 일부 실패 | 성공한 섹션만 저장, 실패 섹션만 재생성 | 위와 동일 |
| LLM 응답 지연 | 타임아웃 설정, 섹션 단위 병렬 처리, React에 진행 상태 스트리밍 | 진행률 표시 |
| 서버 연결 끊김 (HITL 대기 중) | thread checkpoint에서 재개 | "이전 작업을 이어서 진행합니다" |

---

## 7. PoC 범위 & 로드맵

task3 부록 B 로드맵과 동일한 8주(하루 1~2시간) 기준입니다.

| Phase | 기간 | 주요 작업 | Milestone |
|-------|-----|---------|----------|
| Phase 1 기초 구축 | 1-2주 | 공고 30건·가상 기업 5개사 데이터 준비, BGE-M3·Chroma 구성, `langgraph dev` 실행 확인, 공고 분석 Agent | SC-001 기본 동작 |
| Phase 2 Agent 구현 | 3-4주 | Supervisor, 매칭 Agent, 제안서 작성 Agent(Sub-Agent 4개, HTML 출력), HITL 2곳 | SC-002 기본 동작 |
| Phase 3 UI & 안정화 | 5-6주 | React 화면(추천 목록, 초안 미리보기·섹션 편집), SDK 연동, DOMPurify, HTML 다운로드, 에러 처리, 기업 10개사로 확장 | PoC 필수 기능 완성 |
| Phase 4 검증 & 평가 | 7-8주 | 정답셋 300조합(3단계 라벨) 추천 정확도, 초안 30건 커버율·근거성, 시간 측정 3건, 담당자 피드백 반영 | PoC 완성 |

**2차 목표 (PoC 이후)**
- 검증 Agent 기반 자기 검증·재작성 루프
- WYSIWYG 편집기, PDF/Word 내보내기
- 나라장터 API 실시간 연동 및 자동 수집

**범위 밖**: 입찰 자동 제출, 낙찰가 예측, 프로덕션 보안·성능 최적화 (task2 Out of Scope)

---

## 8. LangGraph 구조 예시

상세 설계 단계의 출발점이 되는 그래프 골격입니다. (의사코드 수준, 모델명은 설정값으로 분리)

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


class ProposalState(TypedDict):
    company_profile: dict
    tenders: list[dict]
    recommendations: list[dict]
    selected_tender_id: str | None
    draft_sections: dict[str, str]   # {section_id: "<section>...</section>"}
    criteria_table: str              # "<table>...</table>"
    proposal_html: str
    approval_status: str


def analyze_tenders(state: ProposalState) -> dict:
    # 공고 분석 Agent: 공고 원문 → TenderInfo 구조화
    ...

def match_tenders(state: ProposalState) -> dict:
    # 매칭 Agent: 필터링 → 하이브리드 점수 → 상위 5건 + 근거
    ...

def select_tender(state: ProposalState) -> dict:
    # HITL ①: React에서 공고 1건 선택할 때까지 대기
    tender_id = interrupt({"recommendations": state["recommendations"]})
    return {"selected_tender_id": tender_id}

def generate_sections(state: ProposalState) -> dict:
    # 제안서 작성 Agent: Sub-Agent 4개 병렬 실행 → 섹션별 HTML 검증 후 취합
    ...

def build_criteria_table(state: ProposalState) -> dict:
    # 매칭 Agent: 평가기준 대응표(HTML 표) 생성
    ...

def review_proposal(state: ProposalState) -> dict:
    # HITL ②: React에서 검토·수정 후 확정할 때까지 대기
    edited = interrupt({"draft": state["draft_sections"], "table": state["criteria_table"]})
    return {"draft_sections": edited, "approval_status": "approved"}

def assemble_proposal(state: ProposalState) -> dict:
    # 섹션 4개 + 대응표 → 단일 HTML 문서
    ...


builder = StateGraph(ProposalState)
builder.add_node("analyze_tenders", analyze_tenders)
builder.add_node("match_tenders", match_tenders)
builder.add_node("select_tender", select_tender)
builder.add_node("generate_sections", generate_sections)
builder.add_node("build_criteria_table", build_criteria_table)
builder.add_node("review_proposal", review_proposal)
builder.add_node("assemble_proposal", assemble_proposal)

builder.add_edge(START, "analyze_tenders")
builder.add_edge("analyze_tenders", "match_tenders")
builder.add_edge("match_tenders", "select_tender")
builder.add_edge("select_tender", "generate_sections")
builder.add_edge("generate_sections", "build_criteria_table")
builder.add_edge("build_criteria_table", "review_proposal")
builder.add_edge("review_proposal", "assemble_proposal")
builder.add_edge("assemble_proposal", END)

# LangGraph Server가 checkpointer를 주입하므로 compile 시 지정하지 않음
graph = builder.compile()
```

**`langgraph.json` (서버 설정 예시)**

```json
{
  "dependencies": ["."],
  "graphs": { "proposal": "./agent/graph.py:graph" },
  "env": ".env"
}
```

**React 측 호출 흐름 (개념)**

| 동작 | SDK 호출 |
|------|---------|
| 작업 시작 | thread 생성 → `proposal` 그래프 실행 (진행 상태 스트리밍) |
| HITL ① 공고 선택 | interrupt 데이터(추천 5건) 표시 → 선택한 `tender_id`로 재개 |
| HITL ② 제안서 확정 | interrupt 데이터(섹션 HTML) sanitize 후 표시 → 수정본으로 재개 |
| 다운로드 | State의 `proposal_html`을 `.html` 파일로 저장 |

**설계 포인트**
- `interrupt`로 HITL ①, ②에서 실행을 멈추고, LangGraph Server의 thread checkpoint가 State를 보존해 React에서 재개 요청 시 같은 지점부터 이어서 실행
- `generate_sections` 내부는 Sub-Agent 4개를 병렬 실행 (상세 설계 단계에서 서브그래프 또는 `Send` API로 구체화)
- 검증 Agent(2차 목표)는 `generate_sections` 뒤에 조건 분기 노드로 추가 예정

---

작성일: 2026-10-01  
**다음 단계:** '상세 설계 및 개발 환경 구축' Task에서 State 스키마, 도구 인터페이스, Pydantic 모델, React 화면 구성 확정
