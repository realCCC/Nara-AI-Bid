"""공고·업체·지역 매핑 로드와 정규화."""
import csv
import json
from functools import lru_cache
from pathlib import Path

from backend import config

# 발주처 담당자 개인정보 — 로드 시 제거 (2번 문서 보안 제약)
PII_FIELDS = ("ntceInsttOfclNm", "ntceInsttOfclTelNo", "ntceInsttOfclEmailAdrs", "exctvNm", "dminsttOfclEmailAdrs")


def _to_int(v) -> int | None:
    try:
        return int(str(v).replace(",", "").strip())
    except ValueError:
        return None


def normalize_tender(raw: dict) -> dict:
    t = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k not in PII_FIELDS}
    files = []
    for i in range(1, 11):
        name, url = t.get(f"ntceSpecFileNm{i}", ""), t.get(f"ntceSpecDocUrl{i}", "")
        if name and url:
            files.append({"name": name, "url": url})
    t["files"] = files
    t["pdfs"] = [f for f in files if f["name"].lower().endswith(".pdf")]
    t["budget"] = _to_int(t.get("asignBdgtAmt"))
    return t


def load_tenders(path: Path | str | None = None, latest_only: bool = True) -> list[dict]:
    """latest_only: 같은 공고번호의 차수(변경·취소공고)가 여러 건이면 마지막 차수만 남기고, 취소공고는 뺀다."""
    path = Path(path or config.TENDERS_FILE)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["response"]["body"]["items"] if isinstance(data, dict) else data
    tenders = [normalize_tender(t) for t in items]
    if not latest_only:
        return tenders
    latest: dict[str, dict] = {}
    for t in tenders:
        cur = latest.get(t["bidNtceNo"])
        if cur is None or t.get("bidNtceOrd", "") > cur.get("bidNtceOrd", ""):
            latest[t["bidNtceNo"]] = t
    return [t for t in latest.values() if t.get("ntceKindNm") != "취소공고"]


@lru_cache
def tenders_by_no(path: str | None = None) -> dict[str, dict]:
    return {t["bidNtceNo"]: t for t in load_tenders(path)}


def load_region_limits(path: Path | str | None = None) -> dict[str, list[str]]:
    """지역 제한 공고별 허용 시·도 (수동 매핑표). 파일이 없으면 빈 dict → 지역 제한 공고는 보수적으로 제외."""
    path = Path(path or config.REGION_LIMITS_FILE)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            row["bidNtceNo"].strip(): [r.strip() for r in row["allowed_regions"].split(";") if r.strip()]
            for row in csv.DictReader(f)
        }


def load_company_file(path: Path | str) -> dict:
    """시드 JSON 1건 (최소 실행 경로·DB 적재용). 비밀번호는 제거해서 돌려준다."""
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    c.pop("비밀번호", None)
    return c
