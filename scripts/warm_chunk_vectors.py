"""RAG 청크 벡터 미리 계산: 업체마다 추천 Worker(인터뷰 없음) 상위 공고의 PDF 청크만 임베딩해 CHUNK_VEC_DIR에 저장.

전체 PDF를 받지 않고, 실제로 추천에 오르는 공고만 계산한다. 이미 저장된 청크는 건너뛴다.
EMBEDDING_PROVIDER=bge 필요 (uv sync --extra rag).

uv run python scripts/warm_chunk_vectors.py                          # data/companies/*.json 전부
uv run python scripts/warm_chunk_vectors.py --company data/companies/company_01.json
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.graphs.recommend import run_pipeline  # noqa: E402
from backend.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from backend.tools.pdf import tender_chunks  # noqa: E402
from backend.tools.retrieval import chunk_vectors  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--company", type=Path, action="append", help="업체 파일 (여러 번 지정 가능, 생략하면 전체)")
    p.add_argument("--tenders", type=Path, default=config.TENDERS_FILE)
    args = p.parse_args()
    if config.EMBEDDING_PROVIDER != "bge" or not config.CHUNK_EMBEDDING:
        sys.exit("EMBEDDING_PROVIDER=bge, CHUNK_EMBEDDING=1 이어야 합니다 (.env)")
    tenders, regions = load_tenders(args.tenders), load_region_limits()
    by_no = {t["bidNtceNo"]: t for t in tenders}
    done: set[str] = set()
    for path in args.company or sorted(config.COMPANIES_DIR.glob("*.json")):
        company = load_company_file(path)
        top = run_pipeline(tenders, company, regions)["top"]
        for r in top:
            no = r["bidNtceNo"]
            if no in done:
                continue
            done.add(no)
            start = time.time()
            chunks = tender_chunks(by_no[no])
            if chunks:
                chunk_vectors(no, chunks)
            print(f"{company['업체ID']} {no} 청크 {len(chunks):>3}개 {time.time() - start:5.1f}s | {r['bidNtceNm'][:40]}")
    print(f"→ 공고 {len(done)}건, {config.CHUNK_VEC_DIR}")
