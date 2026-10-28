"""공고서 PDF 다운로드(캐시)·텍스트 추출·청킹."""
import logging
import re
from pathlib import Path

import httpx

from backend import config
from backend.tracing import traced

log = logging.getLogger(__name__)


def fetch_tender_pdfs(tender: dict, pdf_dir: Path | None = None) -> list[dict]:
    """첨부 PDF를 data/pdfs/<공고번호>/ 에 저장. 이미 있으면 다시 받지 않음. 실패한 파일은 건너뜀."""
    out = Path(pdf_dir or config.PDF_DIR) / tender["bidNtceNo"]
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(tender["pdfs"], 1):
        path = out / f"{i}.pdf"
        if not path.exists() and config.FETCH_PDFS:
            try:
                r = httpx.get(f["url"], timeout=30, follow_redirects=True)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("PDF 형식 아님")
                path.write_bytes(r.content)
            except Exception as e:  # 다운로드 실패 → 해당 파일은 룰 근거만
                log.warning("PDF 다운로드 실패 %s %s: %s", tender["bidNtceNo"], f["name"], e)
                continue
        if path.exists():
            saved.append({"file": f["name"], "path": path})
    return saved


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


# ---------- 조항 구조 청킹 ----------
# 공고서는 "1. 입찰개요 / 2. 입찰참가자격 / 가. … / ○ … / ※ …"처럼 조항 기호로 시작하는 줄이 단위다.
# 줄바꿈은 문장 중간에서도 일어나므로, 조항 기호로 시작하는 줄에서만 새 단위를 연다.
CHUNK_MIN, CHUNK_MAX, HARD_OVERLAP = 300, 800, 100
_PAGE_NO = re.compile(r"^-\s*\d+\s*-$")                     # "- 3 -"
_RULE = re.compile(r"^([-=*_─━·.~]\s*){5,}$")               # "-----", "*****"
_SECTION = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[.)]|\d{1,2}\s*\.(?!\d)|제\s*\d+\s*(?:조|장|절)|[□■▣◆◇])")
_ITEM = re.compile(r"^(?:[가-하]\s*[.)]|\d{1,2}\s*\)|\(\s*\d{1,2}\s*\)|\(\s*[가-하]\s*\)|[①-⑳]"
                   r"|[○◦❍●ㅇ▶▷※*§•∙]|-\s|\d{1,2}\s*-\s*\d)")
_SENT_END = re.compile(r"(?<=[다함음임요]\.)\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 일부 PDF는 단어마다 \x01이 붙어 추출됨 → 인용 대조·검색이 어긋남
# 거의 모든 공고에 똑같이 들어가는 청렴계약·신고 안내. 검색 상위를 차지해서 뺀다 (자격 핵심어가 있으면 남김)
_BOILER = re.compile(r"청렴계약|금품|향응|담합|알선|청탁|손해배상액|공익제보|클린신고|부정당업자로\s*제재|콜센터")
_KEEP = re.compile(r"소재지|지역제한|주된\s*영업소|본점|직접생산|중소기업|소기업|소상공인|업종코드|실적|면허|배점|평가항목|과업")


def _is_heading(line: str) -> bool:
    """섹션 제목: 섹션 기호로 시작하고 짧으며 문장으로 끝나지 않는 줄 (예: '2. 입찰참가자격')."""
    return bool(_SECTION.match(line)) and len(line) <= 40 and not re.search(r"[다함음임]\.?$", line)


def _units(pages: list[tuple[int, str]]) -> list[dict]:
    """줄 → 조항 단위 (본문, 페이지들, 섹션 제목)."""
    units, cur, section = [], None, ""
    for page, text in pages:
        for raw in text.split("\n"):
            line = re.sub(r"\s+", " ", _CTRL.sub("", raw)).strip()
            if not line or _PAGE_NO.match(line) or _RULE.match(line):
                continue
            if _is_heading(line):
                section = line
            if cur is None or _is_heading(line) or _SECTION.match(line) or _ITEM.match(line):
                cur = {"lines": [], "pages": set(), "section": section}
                units.append(cur)
            cur["lines"].append(line)
            cur["pages"].add(page)
    out = []
    for u in units:
        text = " ".join(u["lines"])
        if _BOILER.search(text) and not _KEEP.search(text):
            continue
        out.append({"text": text, "pages": u["pages"], "section": u["section"]})
    return out


def _split_long(text: str) -> list[str]:
    """CHUNK_MAX보다 긴 조항은 문장 끝에서 자르고 앞 문장 하나를 겹친다. 문장 하나가 너무 길면 글자 수로 자른다."""
    if len(text) <= CHUNK_MAX:
        return [text]
    sents = []
    for s in _SENT_END.split(text):
        if len(s) <= CHUNK_MAX:
            sents.append(s)
        else:
            step = CHUNK_MAX - HARD_OVERLAP
            sents += [s[i:i + CHUNK_MAX] for i in range(0, len(s) - HARD_OVERLAP, step)]
    pieces, buf = [], []
    for s in sents:
        if buf and len(" ".join([*buf, s])) > CHUNK_MAX:
            pieces.append(" ".join(buf))
            buf = [buf[-1]] if len(buf[-1]) <= CHUNK_MAX // 3 and len(buf[-1]) + len(s) < CHUNK_MAX else []
        buf.append(s)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_pages(file: str, pages: list[tuple[int, str]]) -> list[dict]:
    """조항 단위로 나눈 뒤, 같은 섹션의 짧은 조항은 CHUNK_MIN까지 합치고 긴 조항은 CHUNK_MAX 안으로 자른다.
    그래도 CHUNK_MIN보다 짧은 청크(한두 줄짜리 섹션, 섹션 끝 조각)는 이웃 청크에 붙인다 (섹션 표시는 둘 다 남김).

    text: 원문(출처 대조용, 인용은 여기에 있어야 함). search_text: 섹션 제목을 앞에 붙인 검색·임베딩용.
    page: 첫 페이지, pages: 걸친 페이지 전부 (조항이 페이지를 넘어갈 수 있음).
    """
    bufs, buf = [], None
    for u in _units(pages):
        for piece in _split_long(u["text"]):
            if (buf and buf["sections"][-1] == u["section"] and len(buf["text"]) < CHUNK_MIN
                    and len(buf["text"]) + 1 + len(piece) <= CHUNK_MAX):
                buf["text"] += " " + piece
                buf["pages"] |= u["pages"]
            else:
                buf = {"text": piece, "pages": set(u["pages"]), "sections": [u["section"]]}
                bufs.append(buf)
    merged = []
    for b in bufs:
        prev = merged[-1] if merged else None
        if (prev and (len(prev["text"]) < CHUNK_MIN or len(b["text"]) < CHUNK_MIN)
                and len(prev["text"]) + 1 + len(b["text"]) <= CHUNK_MAX):
            prev["text"] += " " + b["text"]
            prev["pages"] |= b["pages"]
            prev["sections"] += [s for s in b["sections"] if s not in prev["sections"]]
        else:
            merged.append(b)
    chunks = []
    for b in merged:
        if len(b["text"]) > 30:
            sec = " | ".join(s for s in b["sections"] if s)
            chunks.append({"file": file, "page": min(b["pages"]), "pages": sorted(b["pages"]), "section": sec,
                           "text": b["text"], "search_text": f"[{sec}] {b['text']}" if sec else b["text"]})
    return chunks


@traced("tender_chunks", big=("tender",), outputs=lambda cs: {"n_chunks": len(cs), "files": sorted({c["file"] for c in cs})})
def tender_chunks(tender: dict) -> list[dict]:
    chunks = []
    for f in fetch_tender_pdfs(tender):
        try:
            chunks += chunk_pages(f["file"], parse_pdf(f["path"]))
        except Exception as e:
            log.warning("PDF 추출 실패 %s: %s", f["path"], e)
    return chunks
