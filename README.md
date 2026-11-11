# 입찰에이전트 PoC

나라장터 기업 맞춤 공고 추천 (+ 제안서 초안)

## 📌 개요

- **문제**: 입찰 담당자가 매일 올라오는 공고 중 자사에 맞는 공고를 일일이 찾고, 제안서 골격을 처음부터 써야 함
- **솔루션**: 업체 프로필로 맞춤 인터뷰 → 이번에 가장 원하는 사업인 **마스터키**를 정하고 역량에 맞춘 **검색 프롬프트**를 확인 → 공고 443건을 **자격 필터 → 의미 검색·리랭크 → 공고서 PDF 근거 순위(RAG) → 추천 검증**으로 좁혀 **근거·출처와 함께 추천**(메인). 담당자가 확인 체크리스트를 거쳐 공고를 확정하면 업체 문서를 근거로 **제안서 초안**을 만든다. 최종 판단은 담당자 (HITL)

## 🎯 핵심 KPI

| 구분 | 지표 | 목표 |
|---|---|---|
| 추천 (1순위) | 추천 정확도 | Hit Rate@3 80% 이상, Precision@3 0.67 이상 |
| 추천 (1순위) | 자격 필터 정확성 | 하드 필터 통과 공고 중 '부적격' 0건 |
| 추천 (1순위) | 근거 설명 가능성 | 근거 출처(PDF 인용·필드 값) 확인 비율 95% 이상 |
| 업무 | 시간 단축 | 공고 검토 + 제안서 골격 작성 50% 이상 단축 |
| 제안서 (보조) | 초안 근거성 | 업체 프로필·업체 문서·공고서에 없는 사실 기재 0건 |

## 🔄 추천 흐름

LangGraph 그래프 하나(`backend/graphs/proposal.py`)를 규칙 기반 Supervisor가 단계별로 실행하고, HITL 지점에서 멈췄다가 재개한다.

| 단계 | 하는 일 | 코드 |
|---|---|---|
| ⓪ 맞춤 인터뷰 (HITL) | LLM이 업체정보로 문항·선택지 생성, 자유 입력도 같은 효과로 해석 | `tools/interview.py` |
| ① 마스터키·검색 프롬프트 (HITL) | LLM이 역량·실적·인터뷰 답으로 후보와 후보별 검색 프롬프트 제안 → 담당자가 고르거나 직접 입력하고, 프롬프트를 확인·수정한 뒤 추천 시작 | `tools/interview.py` |
| ② 자격 필터 | PDF 보유·업종·참가 제한·지역·예산(H1~H6) + 인터뷰 조건 | `tools/rules.py` |
| ③ 소프트 점수 | S1 업종 · S2 역량 키워드(문자열·의미 유사도, 마스터키 포함) · S3 예산 · S4 계약방식 | `tools/rules.py` |
| ④ 의미 검색·리랭크 | 확인한 검색 프롬프트로 dense + BM25 + 룰 순위를 RRF로 합치고, 검색 문장을 절 단위로 나눠 리랭크(절별 최댓값) → 마스터키 일치 공고를 앞으로 → 상위 10건 | `graphs/recommend.py`, `tools/search.py` |
| ⑤ 근거 순위 (RAG) | 10건의 공고서 청크를 하이브리드 검색, LLM이 순위·근거 작성, 인용은 원문 대조 | `agents/rag.py` |
| ⑥ 추천 검증 | 검증 서브에이전트가 상위 10건(`VERIFY_TOP_N`)의 적합도를 재평가하고, 출처 대조를 통과한 자격 불충족(issue)이 있는 공고는 전체 목록 맨 뒤로 보냄 | `agents/verify.py` |
| ⑦ 확인 체크리스트 (HITL) | 공고 조건 + 공고서에서 뽑은 자격·서류·일정 + 검증 issue, 전부 체크해야 확정 | `tools/checklist.py` |
| ⑧ 제안서 초안 | 평가항목·배점 추출 → 템플릿 + 섹션별 LLM 작성(업체 문서 근거) → 대화로 칸 단위 보완, 공고서 PDF 분할 보기 | `tools/proposal*.py` |

- **마스터키는 최우선 기준이지만 공고를 제외하지 않는다.** 맞지 않는 공고도 뒤에 남는다. 예산·지역·제외어 필터는 인터뷰 답을 따르며, 검색 프롬프트를 고쳐도 바뀌지 않는다.
- **순위 기준 (⑤)**: 참가자격 충돌 없음 > 마스터키 일치 > 검색 프롬프트·인터뷰 답·역량·실적 종합 > 예산.
- 마스터키 일치 판정은 키의 단어가 공고명·분류명에 있는지로 본다(`운영`·`지역` 같은 일반어 제외). 표현이 다른 공고는 의미 검색과 LLM 순위 판단이 맡는다.
- LLM·임베딩·벡터 DB가 없거나 출력 검증에 실패하면 단계마다 룰·BM25·기본값으로 폴백한다. 마스터키 후보는 역량·실적 키워드로 대체한다.
- LLM 검색 문장은 업체 프로필별로 `data/query_cache.json`에 고정해 실행마다 결과가 흔들리지 않게 한다.
- **추천 카드**: 매칭 점수(0~100)를 등급(매우 적합·적합·보통·낮음) 게이지로 보이고, 공고 요약(예산·수요기관·마감·계약방식·업종·지역 제한), 업체 가중치 순 점수 구성, 자격·검증 확인 사항, 핵심 근거 3개를 펼치지 않고 보여 준다. 상세 근거·점수 계산·검색 점수는 접어 둔다.
- 근거가 없는 제안서 칸은 `[확인 필요]` 또는 `[AI 초안·검토 필요]`로 남긴다.

## 🔧 기술 스택

| 구분 | 기술 |
|---|---|
| LLM | Azure OpenAI (structured output). 모델은 `AZURE_OPENAI_CHAT_DEPLOYMENT` 배포 이름을 따른다 (`.env.example`은 gpt-4.1) |
| Agent | LangGraph + LangChain, LangSmith 추적 |
| 검색 | BM25(kiwipiepy) + 임베딩(Azure text-embedding-3-large 1024차원 또는 로컬 BGE-M3) + RRF, bge-reranker-v2-m3 |
| 벡터 DB | PostgreSQL 16 + pgvector (Docker) — 공고서·업체 문서 청크, HNSW. 공고 검색 벡터는 DB가 아니라 메모리·`emb_cache.npz` 캐시 |
| PDF | PyMuPDF (조항 구조 청킹, 300~800자) |
| 서버·DB | FastAPI, Jinja2, SQLite(업체·이력) |
| 프론트 | 단일 HTML (`frontend/index.html`) |
| 기타 | Python 3.11, uv, pytest (108건) |

## 💾 데이터

| 데이터 | 구성 |
|---|---|
| 공고 | 실제 용역 공고 500건(2026-05) → 유효 443건, PDF 보유 410건 ([수집 방법](공고데이터수집방법.md)) |
| 업체 | 가상 업체 10곳 (`data/companies/`) + 회사소개서 (`data/companies/docs/`) |
| 지역 제한 | 허용 시·도 수동 매핑 (`data/region_limits.csv`) |
| 평가 | 검증 정답셋 50건(`data/eval/verify_gold.json`, 담당자 최종 확인 전) |

## 📁 폴더 구조

```
backend/
  server.py     FastAPI (화면·API·실행 로그·추천 진행·제안서 대화 SSE)
  graphs/       Supervisor·전체 그래프, 추천 Worker 서브그래프
  agents/       interview · rag · verify · selection · draft · state
  tools/        interview · keyword_profile · rules · search · retrieval · pdf · evidence · checklist · llm · proposal · proposal_writer · proposal_chat · company_docs · activity(실행 로그) · loaders
  db/           SQLite(repository.py), pgvector(vector_store.py)
  templates/    proposal.html.j2
frontend/       index.html
scripts/        run_demo · seed_companies · check_env · validate_data · warm_chunk_vectors · eval_verify · eval_body · eval_search · diagnose
tests/          pytest 108건
data/           공고·업체·업체 문서·평가셋·공고 벡터 캐시(emb_cache.npz)·검색 문장 캐시(query_cache.json)
docs/           1~7 단계 문서
```

## 📅 일정 (8주, 2026-09-28 시작)

| Phase | 기간 | 상태 (11/11 기준) |
|---|---|---|
| 기획 | 09/28-10/11 | 완료 |
| 1 기초 구축 | 10/12-10/25 | 완료 |
| 2 Agent 구현 | 10/26-11/01 | 완료 (RAG·검증·체크리스트·제안서 LLM 작성까지) |
| 3 UI & 안정화 | 11/02-11/08 | 완료 — 마스터키 추천·실행 로그·추천 카드 한눈에 보기 (업체 확장·정답셋 최종 확인은 Phase 4로 이월) |
| 4 검증 & 평가 | 11/09-11/22 | 진행 중 — LangSmith 기반 실패 원인 진단·개선(검색 nDCG@10 0.870 → 0.892, P@10 0.93 → 0.95), KPI 측정·최종 산출물 남음 |

**검증 에이전트 평가 요약** (2차 검토 라벨 50건, 마스터키 도입 전 측정): 부적격 하위 배치율 0.64 → 1.00, 플래그 정밀도/재현율 1.00 / 0.92. 평가 방법은 [6번 문서](docs/6.테스트및고도화.md) 1-5절 참고.

## ▶️ 실행

```powershell
uv sync                                        # 기본 설치 (키워드 검색 모드)
uv sync --extra rag                            # (선택) BGE-M3·리랭커·pgvector 클라이언트
docker compose up -d                           # (선택) 청크 벡터 DB (Docker Desktop을 먼저 실행)
uv run python scripts/warm_chunk_vectors.py --all  # (선택) 공고 청크 벡터 미리 계산
uv run python scripts/seed_companies.py        # 업체 → data/app.db
uv run uvicorn backend.server:app --port 8000  # http://localhost:8000 (C001~C010 / 비밀번호 1)
uv run pytest                                  # 테스트 (실제 LLM 호출 없음)
```

- `.env.example`을 `.env`로 복사해 Azure 키를 넣으면 AI 단계가 켜진다. 키가 없으면 **룰 모드**로 동작한다.
- `EMBEDDING_PROVIDER`: `keyword`(기본) · `azure`(임베딩 API) · `bge`(로컬, `--extra rag` 필요). 리랭크는 의미 검색 경로(`azure`·`bge`)에서 시도하며 `--extra rag`가 없거나 실패하면 생략한다.
- 벡터 DB에 연결하지 못하면 청크 검색은 BM25·키워드만으로 하고 추천은 계속된다(30초마다 재연결).
- **실행 로그**: 로그인 후 우측 상단 [로그 확인]을 누르면 팝업이 열린다. 주요 기능과 LLM·RAG·벡터DB·임베딩·리랭크 호출을 2초마다 보여 주고, 룰·BM25로 넘어가면 `대체`, 실패하면 `오류`로 표시한다. 서버 메모리에 최근 300줄만 보관한다(재시작하면 사라짐).
- `BODY_SEARCH=1`: 1단계 검색에 PDF 본문 순위 추가. 실측 차이가 없어(nDCG 0.904 → 0.908) 기본값은 끔.
- LangSmith: `.env`에 `LANGSMITH_TRACING=true`와 키를 넣으면 추천 1회가 트레이스 하나로 기록된다(하이브리드 검색 단계별 순위, 리랭크 전후 순위 포함).
- 평가·진단: `scripts/eval_search.py`(검색 nDCG·P@10), `scripts/diagnose.py`(라벨 공고별 실패 단계 판정 — 필터/검색 누락/리랭크 밀림/자격 검증 — 과 트레이스 링크).

## ⚠️ 범위 밖

조달청 공고 API 실시간 연동·마감일 필터, HWP 파싱, 완성형 제안서·자동 제출, 낙찰가 예측, 실제 인증

## 📚 문서

[1. 역량·기술스택](docs/1.역량및기술스택확인.md) · [2. 문제정의·서비스계획](docs/2.문제정의및서비스계획.md) · [3. 시나리오](docs/3.시나리오수립.md) · [4. 상세설계·개발환경](docs/4.상세설계및개발환경구축.md) · [5. POC 모듈](docs/5.POC%20모듈%20구현.md) · [6. 테스트·고도화](docs/6.테스트및고도화.md) · [7. E2E 서비스](docs/7.E2E서비스개발.md)
