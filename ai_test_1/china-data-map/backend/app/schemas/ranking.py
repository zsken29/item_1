from pydantic import BaseModel
from typing import Optional


class ProvinceRanking(BaseModel):
    rank: int
    province: str
    value: float


class CityRanking(BaseModel):
    rank: int
    city: str
    province: str
    value: float


class RankingResponse(BaseModel):
    ranking: list
