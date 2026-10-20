"""업체 시드 JSON 스키마 (seed_companies·validate_data에서 검증)."""
from pydantic import BaseModel, ConfigDict, Field


class Industry(BaseModel):
    대분류: list[str]
    세부분류번호: list[str] = []


class BudgetRange(BaseModel):
    최소: int = Field(ge=0)
    최대: int = Field(gt=0)


class PastProject(BaseModel):
    프로젝트명: str
    계약금액: int
    키워드: list[str] = []


class CompanySeed(BaseModel):
    model_config = ConfigDict(extra="allow")

    업체ID: str = Field(pattern=r"^C\d{3}$")
    비밀번호: str
    회사명: str
    중소기업여부: bool
    소재지: str
    업종: Industry
    참여가능금액: BudgetRange
    역량키워드: list[str]
    선호계약방식: list[str] = []
    보유인력: dict[str, int] = {}
    과거실적: list[PastProject] = []
