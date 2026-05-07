from pydantic import BaseModel
from typing import Optional


class CityBase(BaseModel):
    name: str
    province: str
    lat: float
    lng: float
    level: int = 2
    city_code: Optional[str] = None


class CityCreate(CityBase):
    pass


class CityResponse(CityBase):
    id: int

    class Config:
        from_attributes = True


class CityStatsBase(BaseModel):
    year: int
    per_gdp: Optional[float] = None
    per_income: Optional[float] = None
    house_price: Optional[float] = None
    population: Optional[float] = None
    gdp_total: Optional[float] = None


class CityStatsCreate(CityStatsBase):
    city_id: int


class CityStatsResponse(CityStatsBase):
    id: int
    city_id: int

    class Config:
        from_attributes = True


class CityWithStats(BaseModel):
    id: int
    name: str
    province: str
    lat: float
    lng: float
    level: int
    city_code: Optional[str] = None
    stats: Optional[CityStatsResponse] = None

    class Config:
        from_attributes = True


class ProvinceRanking(BaseModel):
    rank: int
    province: str
    value: float


class CityRanking(BaseModel):
    rank: int
    city: str
    province: str
    value: float


class TrendData(BaseModel):
    years: list[int]
    per_gdp: list[Optional[float]]
    per_income: list[Optional[float]]
    house_price: list[Optional[float]]
    population: list[Optional[float]]
    gdp_total: list[Optional[float]]


class TrendResponse(BaseModel):
    city: str
    province: str
    trend: TrendData
