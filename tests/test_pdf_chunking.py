from backend.tools import pdf
from backend.tools.evidence import check_reason
from backend.tools.pdf import CHUNK_MAX, chunk_pages

QUAL = ("가. 「지방자치단체를 당사자로 하는 계약에 관한 법률 시행령」 제13조에 의한 자격을 갖추고\n"
        "입찰공고일 전일부터 법인등기부상 본점 소재지가 강원특별자치도 내에 있는 업체이어야 합니다.\n")
PAGES = [
    (1, "- 1 -\n< 이 계약은 청렴계약(서약)제가 적용됩니다 >\n"
        "1. 금품·향응, 사례, 증여 등을 직접적·간접적으로 요구하거나 수수하지 않겠습니다.\n"
        "----------------------------------------\n"
        "2. 입찰참가자격\n" + QUAL +
        "나. 「중소기업제품 구매촉진 및 판로지원에 관한 법률」에 따른 직접생산확인증명서(건설폐기물처리서비스)를\n"),
    (2, "소지한 업체이어야 합니다.\n"
        "3. 과업 내용\n○ 과업명: 건설폐기물 처리 용역\n○ 과업 범위: 폐콘크리트 428톤, 폐아스콘 7,502톤 중간처리\n"),
]


def test_clauses_sections_and_boilerplate():
    chunks = chunk_pages("공고문.pdf", PAGES)
    text = " ".join(c["text"] for c in chunks)
    assert "금품" not in text and "- 1 -" not in text and "-----" not in text  # 청렴계약·페이지 번호·구분선 제거
    qual = next(c for c in chunks if "본점 소재지" in c["text"])
    assert "2. 입찰참가자격" in qual["section"] and qual["search_text"].startswith("[")
    assert "직접생산확인증명서(건설폐기물처리서비스)를 소지한 업체" in qual["text"]  # 페이지를 넘는 조항도 한 청크
    assert qual["pages"] == [1, 2] and qual["page"] == 1


def test_evidence_matches_any_page_of_chunk():
    chunks = chunk_pages("공고문.pdf", PAGES)
    ev = {"source_type": "pdf", "file": "공고문.pdf", "quote": "직접생산확인증명서(건설폐기물처리서비스)를"}
    assert check_reason({**ev, "page": 2}, {}, {}, chunks)  # 조항이 1→2쪽에 걸쳐 있어 2쪽으로 적어도 통과
    assert check_reason({**ev, "page": 1}, {}, {}, chunks)
    assert not check_reason({**ev, "page": 3}, {}, {}, chunks)


def test_long_clause_split_with_overlap_and_short_chunks_merged():
    sent = "참가업체는 과업지시서에 따른 세부 수행계획을 제출하여야 합니다. "
    pages = [(1, "1. 과업 개요\n가. " + sent * 40 + "\n2. 계약보증금\n계약보증금은 현금으로 납부합니다.\n")]
    chunks = chunk_pages("a.pdf", pages)
    assert all(len(c["text"]) <= CHUNK_MAX for c in chunks) and len(chunks) >= 2
    assert chunks[1]["text"].startswith(sent.strip())  # 앞 조각의 마지막 문장을 겹친다
    assert "계약보증금은 현금으로" in chunks[-1]["text"] and "2. 계약보증금" in chunks[-1]["section"]  # 짧은 섹션은 이웃에 붙음


def test_control_characters_removed():
    pages = [(1, "2. 입찰참가자격\n가.\x01 본점\x01 소재지가\x01 강원특별자치도\x01 내에 있는 업체이어야 합니다.\n")]
    chunks = chunk_pages("a.pdf", pages)
    assert "\x01" not in chunks[0]["text"]
    assert check_reason({"source_type": "pdf", "file": "a.pdf", "page": 1, "quote": "본점 소재지가 강원특별자치도"}, {}, {}, chunks)


def test_quote_with_prompt_prefix_still_matches():
    """LLM이 프롬프트의 '(파일 p.쪽) [섹션]' 표시까지 인용해도 본문으로 대조한다."""
    chunks = chunk_pages("공고문.pdf", PAGES)
    q = "(공고문.pdf p.1) [2. 입찰참가자격] 본점 소재지가 강원특별자치도 내에 있는 업체"
    assert check_reason({"source_type": "pdf", "file": "공고문.pdf", "page": 1, "quote": q}, {}, {}, chunks)
    assert not check_reason({"source_type": "pdf", "file": "공고문.pdf", "page": 1, "quote": "[2. 입찰참가자격] 없는 문장"},
                            {}, {}, chunks)


def test_heading_detection():
    assert pdf._is_heading("2. 입찰참가자격") and pdf._is_heading("Ⅲ. 제안서 평가")
    assert not pdf._is_heading("1. 금품·향응 등을 요구하거나 수수하지 않겠습니다.")
    assert not pdf._is_heading("가. 전자입찰접수개시일시")


def test_file_name_brackets_tolerated():
    """LLM이 파일명 괄호를 틀리게 적어도('붙임1)공고서.pdf' → '(붙임1)공고서.pdf') 인용·페이지가 맞으면 통과."""
    chunks = chunk_pages("붙임1)공고서.pdf", PAGES)
    ev = {"source_type": "pdf", "page": 1, "quote": "본점 소재지가 강원특별자치도"}
    assert check_reason({**ev, "file": "(붙임1)공고서.pdf"}, {}, {}, chunks)
    assert check_reason({**ev, "file": "붙임1) 공고서.PDF"}, {}, {}, chunks)
    assert not check_reason({**ev, "file": "붙임2)공고서.pdf"}, {}, {}, chunks)
