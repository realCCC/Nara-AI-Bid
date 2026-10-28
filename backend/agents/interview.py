"""인터뷰 Worker (HITL ⓪): 문항 준비 → 객관식 + 자유 입력 인터뷰 → prefs."""
from langgraph.types import interrupt

from backend.agents.state import ProposalState
from backend.tools import keyword_profile
from backend.tools.interview import build_questions, clean_answers, parse_note, to_prefs


def prepare_interview(state: ProposalState):
    """업체정보로 문항을 미리 만든다(LLM). interrupt 재개 때 노드가 다시 실행돼도 LLM을 또 부르지 않도록 분리.
    키워드 의미 프로필도 여기서 미리 만들어 둔다 (추천 단계는 캐시를 쓴다)."""
    keyword_profile.build(state["company_profile"])
    return {"interview_questions": build_questions(state["company_profile"])}


def interview(state: ProposalState):  # HITL ⓪: 객관식 + 자유 입력 인터뷰. 입력에 interview_answers가 있으면 묻지 않음
    questions = state["interview_questions"]
    answers = state.get("interview_answers")
    if answers is None:
        # 재개 값은 {"answers": {...}} — 빈 dict를 그대로 넘기면 LangGraph가 인터럽트 ID 매핑으로 해석한다
        answers = (interrupt({"questions": questions}) or {}).get("answers")
    answers = answers if isinstance(answers, dict) else {}
    note = clean_answers(answers, questions).get("note")
    prefs = to_prefs(answers, questions=questions, note_effect=parse_note(note, state["company_profile"]))
    return {"interview_answers": prefs["answers"], "prefs": prefs}
