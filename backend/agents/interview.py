"""인터뷰 Worker (HITL ⓪): 문항 준비 → 객관식 + 자유 입력 인터뷰 → prefs → 마스터키 후보 → 담당자 확인."""
from langgraph.types import interrupt

from backend.agents.state import ProposalState
from backend.tools import keyword_profile
from backend.tools.interview import (MASTER_KEY_MAX, MASTER_PROMPT_MAX, answer_lines, build_masters, build_questions,
                                      clean_answers, parse_note, resolve_master, to_prefs, with_master)


def prepare_interview(state: ProposalState):
    """업체정보로 문항을 미리 만든다(LLM). interrupt 재개 때 노드가 다시 실행돼도 LLM을 또 부르지 않도록 분리.
    키워드 의미 프로필도 여기서 미리 만들어 둔다 (추천 단계는 캐시를 쓴다)."""
    keyword_profile.build(state["company_profile"])
    return {"interview_questions": build_questions(state["company_profile"])}


def interview(state: ProposalState):  # HITL ⓪: 객관식 + 자유 입력 인터뷰. 입력에 interview_answers가 있으면 묻지 않음
    questions = state["interview_questions"]
    answers = state.get("interview_answers")
    asked = answers is None
    if asked:
        # 재개 값은 {"answers": {...}} — 빈 dict를 그대로 넘기면 LangGraph가 인터럽트 ID 매핑으로 해석한다
        answers = (interrupt({"questions": questions}) or {}).get("answers")
    answers = answers if isinstance(answers, dict) else {}
    note = clean_answers(answers, questions).get("note")
    prefs = to_prefs(answers, questions=questions, note_effect=parse_note(note, state["company_profile"]))
    return {"interview_answers": prefs["answers"], "prefs": prefs, "master_review": asked}


def prepare_master(state: ProposalState):
    """마스터키 후보와 후보별 검색 프롬프트 (LLM). interrupt 재개 때 다시 실행되지 않도록 확인 노드와 분리."""
    return {"master_draft": build_masters(state["company_profile"], state["interview_questions"], state["prefs"])}


def review_master(state: ProposalState):  # HITL: 담당자가 마스터키를 고르고 검색 프롬프트를 확인·수정한 뒤 추천을 시작한다
    draft, picked = state["master_draft"], None
    if state.get("master_review"):
        picked = interrupt({**draft, "key_max": MASTER_KEY_MAX, "prompt_max": MASTER_PROMPT_MAX})
    master = resolve_master(draft, picked, state["company_profile"])
    lines = answer_lines(state["interview_questions"], state["prefs"])
    return {"prefs": with_master(state["prefs"], master, lines)}
