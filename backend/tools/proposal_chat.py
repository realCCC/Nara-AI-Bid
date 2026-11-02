"""제안서 대화형 보완: 사용자 요청 → LLM이 칸 단위 변경안 → 코드가 검증 → 사용자가 고른 변경만 적용.

칸(slot) key: understanding · certifications · vision · pm · period
             items/<i>/plan · items/<i>/evidence · items/<i>/sub/<j>   (i: 평가항목 순번, j: 세부항목 순번)
             phases/<단계>/content|period · staffing/<역할>/count|duty
근거 표시: user → 사용자가 실제로 한 말에 있으면 (담당자 입력), company_doc → 인용이 업체 문서에 있으면 (업체 문서 · 파일),
          tender → 인용이 공고서에 있으면 (공고서 p.쪽), 그 밖에는 [AI 초안·검토 필요].
사실형 칸(인증·면허, PM, 업체 근거)은 담당자 입력·업체 문서로 확인될 때만 변경안에 넣는다.
공고서는 발주기관의 요구 조건이라 우리 회사 사실의 근거가 될 수 없다.
ctx: 서버가 작업(thread)마다 들고 있는 dict — company, tender, criteria, rec, checks, ai.
"""
import copy
import re

from backend import config
from backend.tools import company_docs, llm, proposal, proposal_writer
from backend.tools.evidence import _norm
from backend.tools.retrieval import retrieve
from backend.tools.rules import tokenize

SOURCE_LABEL = {"user": "담당자 입력", "company_doc": "업체 문서", "tender": "공고서", "general": "AI 일반론"}
FACT_KEYS = {"certifications", "pm"}
CHAT_K = 4
# 담당자 말 대조에서 빼는 일반어 (제안서 문체로 바꾸며 LLM이 덧붙이는 말)
USER_STOP = {"보유", "보유함", "인증", "경력", "등록", "면허", "취득", "담당", "총괄", "책임자", "수행", "있음", "투입", "인원"}


def said_by_user(text: str, user_text: str) -> bool:
    """LLM이 source=user라고 한 내용이 사용자가 실제로 한 말에 있는지.
    숫자·영문이 든 단어(ISO9001, 12년, 5명)는 전부, 나머지 단어는 절반 이상이 사용자 말에 있어야 한다."""
    said = _norm(user_text).lower()
    if not said:
        return False
    words = [w.strip(".:;'\"") for w in tokenize(text)]
    words = [w for w in words if len(w) >= 2 and w not in USER_STOP]
    marked = [w for w in words if re.search(r"[0-9A-Za-z]", w)]
    plain = [w for w in words if w not in marked]
    if not words or not all(_norm(w).lower() in said for w in marked):
        return False
    return not plain or sum(_norm(w) in said for w in plain) * 2 >= len(plain)


def _text(f) -> str | None:
    return (f or {}).get("text") if isinstance(f, dict) else (str(f) if f not in (None, "") else None)


def slots(ctx: dict) -> list[dict]:
    """대화에서 바꿀 수 있는 칸 목록 (현재 값 포함)."""
    ai, crit, company, tender = ctx.get("ai") or {}, ctx["criteria"], ctx["company"], ctx["tender"]
    out = []

    def add(key, label, kind, current):
        out.append({"key": key, "label": label, "kind": kind, "current": current})

    add("understanding", "Ⅱ 사업 목적 이해·추진 전략", "fill", _text(ai.get("understanding")))
    add("certifications", "Ⅲ 보유 인증·면허", "fact", _text(ai.get("certifications")))
    add("vision", "Ⅲ 비전·차별화 포인트", "fill", _text(ai.get("vision")))
    add("period", "과업 기간", "plain", crit.get("period"))
    for i, it in enumerate(crit.get("items") or []):
        d = (ai.get("items") or {}).get(it["item"]) or {}
        rule = None if d.get("evidence_edited") else proposal._evidence(it["item"], tender, company)
        add(f"items/{i}/plan", f"Ⅳ-{i + 1} {it['item']} · 수행 방안", "fill", _text(d.get("plan")))
        add(f"items/{i}/evidence", f"Ⅳ-{i + 1} {it['item']} · 업체 근거", "fact", rule or _text(d.get("evidence")))
        for j, s in enumerate(it.get("sub_items") or []):
            add(f"items/{i}/sub/{j}", f"Ⅳ-{i + 1} 세부항목 · {s}", "fill", _text((d.get("sub_items") or {}).get(s)))
    for step in proposal_writer.STEPS:
        p = (ai.get("phases") or {}).get(step) or {}
        add(f"phases/{step}/content", f"Ⅴ 추진 일정 · {step} 주요 내용", "plain", p.get("content"))
        add(f"phases/{step}/period", f"Ⅴ 추진 일정 · {step} 기간", "plain", p.get("period"))
    for role, n in (company.get("보유인력") or {}).items():
        a = (ai.get("staffing") or {}).get(role) or {}
        add(f"staffing/{role}/count", f"Ⅵ 조직 · {role} 투입 인원 (보유 {n}명)", "count", a.get("count"))
        add(f"staffing/{role}/duty", f"Ⅵ 조직 · {role} 담당 업무", "plain", a.get("duty"))
    add("pm", "Ⅵ 총괄 책임자(PM)", "fact", _text(ai.get("pm")))
    return [s | {"current": None if s["current"] in (None, "") else str(s["current"])} for s in out]


def _tender_fill(text: str, cites: list[dict], chunks: list[dict]) -> dict:
    """공고서 인용 검증: 인용이 전부 청크 원문에 있으면 (공고서 p.쪽)."""
    pages = []
    for c in cites:
        q = _norm(c.get("quote"))
        hit = next((ch for ch in chunks if q and q in _norm(ch["text"])), None)
        if hit is None:
            return {"text": text, "grounded": False, "sources": []}
        pages.append(f"공고서 p.{hit['page']}")
    return {"text": text, "grounded": bool(cites), "sources": list(dict.fromkeys(pages))}


def validate(raw: dict, ctx: dict, company_chunks: list[dict], tender_chunks: list[dict],
             user_text: str = "") -> tuple[list[dict], list[dict]]:
    """LLM 변경안 → (적용 가능한 변경안, 제외한 칸과 이유). user_text: 사용자가 대화에서 한 말 (source=user 대조용)."""
    by_key = {s["key"]: s for s in slots(ctx)}
    staff = ctx["company"].get("보유인력") or {}
    edits, skipped, seen = [], [], set()
    for e in raw.get("edits") or []:
        s, text = by_key.get(e.get("key")), (e.get("text") or "").strip()
        if not s or not text or s["key"] in seen:
            continue
        src, cites = e.get("source") or "general", e.get("cites") or []
        user_ok = src == "user" and said_by_user(text, user_text)
        if s["kind"] == "count":
            nums = set(re.findall(r"\d+", text))
            role = s["key"].split("/")[1]
            if len(nums) != 1:  # "PM 1명 포함 3명"·"2~3명"처럼 숫자가 여러 개면 몇 명인지 정할 수 없다
                skipped.append({"key": s["key"], "label": s["label"], "reason": "투입 인원이 숫자 하나로 정해지지 않음"})
                continue
            n = int(nums.pop())
            if not 0 < n <= staff.get(role, 0):
                skipped.append({"key": s["key"], "label": s["label"], "reason": f"보유 인원({staff.get(role, 0)}명) 범위를 벗어남"})
                continue
            value = n
        else:
            if s["kind"] == "fact" and src == "tender":
                skipped.append({"key": s["key"], "label": s["label"],
                                "reason": "공고서는 요구 조건이라 우리 회사 사실의 근거가 될 수 없음"})
                continue
            if src == "user":  # 사용자 말에서 확인되지 않으면 AI 초안으로 낮춘다 (사실형 칸은 아래에서 제외)
                value = {"text": text, "grounded": user_ok, "sources": ["담당자 입력"] if user_ok else []}
            elif src == "company_doc":
                value = proposal_writer.fill({"text": text, "cites": cites}, company_chunks)
            elif src == "tender":
                value = _tender_fill(text, cites, tender_chunks)
            else:
                value = {"text": text, "grounded": False, "sources": []}
            if s["kind"] == "plain":  # 일반 텍스트 칸은 문자열만 저장하고, 출처 검증 결과는 변경안 표시에만 쓴다
                checked = value
                value = text
            elif s["kind"] == "fact" and not value["grounded"]:
                skipped.append({"key": s["key"], "label": s["label"], "reason": "근거(담당자 입력·업체 문서)가 확인되지 않음"})
                continue
        seen.add(s["key"])
        info = checked if s["kind"] == "plain" else value if isinstance(value, dict) else {"grounded": user_ok, "sources": []}
        grounded = info["grounded"]
        label = ", ".join(info["sources"]) if info["sources"] else SOURCE_LABEL.get(src, "AI")
        edits.append({"key": s["key"], "label": s["label"], "before": s["current"],
                      "after": f"{value}명" if s["kind"] == "count" else text,
                      "value": value, "source": label if grounded else "AI 초안·검토 필요", "grounded": grounded})
    return edits, skipped


def apply(ctx: dict, edits: list[dict]) -> tuple[dict, dict]:
    """변경안을 ai·criteria 사본에 적용해 (ai, criteria)를 돌려준다."""
    ai, crit = copy.deepcopy(ctx.get("ai") or {}), copy.deepcopy(ctx["criteria"])
    items = crit.get("items") or []
    for e in edits:
        parts, v = e["key"].split("/"), e["value"]
        if parts[0] == "period":
            crit["period"] = v
        elif len(parts) == 1:
            ai[parts[0]] = v
        elif parts[0] == "items":
            it = items[int(parts[1])]
            d = ai.setdefault("items", {}).setdefault(it["item"], {"plan": None, "evidence": None, "sub_items": {}})
            if parts[2] == "sub":
                d.setdefault("sub_items", {})[it["sub_items"][int(parts[3])]] = v
            else:
                d[parts[2]] = v
                if parts[2] == "evidence":
                    d["evidence_edited"] = True
        elif parts[0] == "phases":
            ai.setdefault("phases", {}).setdefault(parts[1], {"content": None, "period": None})[parts[2]] = v
        elif parts[0] == "staffing":
            ai.setdefault("staffing", {}).setdefault(parts[1], {"count": None, "duty": None})[parts[2]] = v
    return ai, crit


def _src_label(c: dict, kind: str) -> str:
    sec = (c.get("section") or "").split(" | ")[-1][:20]
    name = c["file"] if len(c["file"]) <= 24 else c["file"][:20] + "…"
    return f"{kind} · {name}" + (f" p.{c['page']}" if kind == "공고서" else "") + (f" [{sec}]" if sec else "")


def chat_steps(ctx: dict, message: str, history: list[dict]):
    """사용자 요청 하나를 단계별로 처리하며 이벤트를 낸다 (화면 진행 표시용).
    ("step", {"id", "status": running|done, "label", "detail"}) … 마지막에 ("result", {"reply", "edits", "skipped"}).
    적용은 하지 않는다."""
    import time

    def step(sid, status, label, detail=None, t0=None):
        if status == "done" and t0 is not None:
            detail = " · ".join(x for x in (detail, f"{time.monotonic() - t0:.1f}초") if x)
        return "step", {"id": sid, "status": status, "label": label, "detail": detail}

    if not config.llm_enabled():
        yield "result", {"reply": "LLM이 설정되지 않아 대화형 보완을 쓸 수 없습니다.", "edits": [], "skipped": []}
        return
    if "company_chunks" not in ctx:  # 작업마다 처음 한 번만 (이후 서버 메모리에서 재사용)
        from backend.tools.pdf import tender_chunks

        t0 = time.monotonic()
        yield step("prepare", "running", "업체 문서·공고서 불러오는 중")
        ctx["company_chunks"] = company_docs.load_chunks(ctx["company"])
        ctx["company_vecs"] = company_docs.vectors(ctx["company"], ctx["company_chunks"])
        ctx["tender_chunks"] = tender_chunks(ctx["tender"])
        yield step("prepare", "done", "업체 문서·공고서 불러옴",
                   f"업체 문서 {len(ctx['company_chunks'])}개 · 공고서 {len(ctx['tender_chunks'])}개 청크", t0)

    t0 = time.monotonic()
    yield step("search", "running", "요청과 관련된 근거 검색 중")
    docs = company_docs.search(ctx["company_chunks"], message, CHAT_K, ctx["company_vecs"])
    tch = retrieve(ctx["tender_chunks"], message, k=CHAT_K, tender_no=ctx["tender"]["bidNtceNo"])
    found = [_src_label(c, "업체 문서") for c in docs] + [_src_label(c, "공고서") for c in tch]
    yield step("search", "done", f"근거 {len(found)}건 찾음", ", ".join(dict.fromkeys(found)) or "관련 근거 없음", t0)

    cur = slots(ctx)
    t0 = time.monotonic()
    yield step("llm", "running", f"AI가 변경안 작성 중 ({len(cur)}개 칸 검토)", f"모델 {config.chat_model_name()}")
    raw = llm.chat_edit(message, history, cur, ctx["tender"], ctx["company"], docs, tch)
    if raw is None:
        yield step("llm", "done", "AI 응답 실패", None, t0)
        yield "result", {"reply": "AI 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.", "edits": [], "skipped": []}
        return
    yield step("llm", "done", f"AI가 변경안 {len(raw.get('edits') or [])}건 제안", None, t0)

    t0 = time.monotonic()
    yield step("verify", "running", "인용·인원 등 근거 검증 중")
    user_text = " ".join([*(h["text"] for h in history if h["role"] == "user"), message])
    edits, skipped = validate(raw, ctx, ctx["company_chunks"], ctx["tender_chunks"], user_text)
    grounded = sum(e["grounded"] for e in edits)
    yield step("verify", "done", f"검증 완료: 적용 가능 {len(edits)}건 · 제외 {len(skipped)}건",
               f"근거 확인 {grounded}건 · AI 초안 {len(edits) - grounded}건", t0)
    yield "result", {"reply": raw.get("reply") or "", "edits": edits, "skipped": skipped}


def chat(ctx: dict, message: str, history: list[dict]) -> dict:
    """chat_steps를 끝까지 돌려 결과만 돌려준다."""
    result = None
    for ev, data in chat_steps(ctx, message, history):
        if ev == "result":
            result = data
    return result
