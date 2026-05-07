"""
API Router - Cities
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional
from collections import defaultdict

from ..database import get_db
from ..models.city import City, CityStats
from ..schemas.city import CityResponse, CityWithStats, TrendResponse, TrendData

router = APIRouter(prefix="/api", tags=["cities"])


@router.get("/cities")
def get_cities(db: Session = Depends(get_db)):
    """获取所有城市基础信息"""
    cities = db.query(City).all()
    return {"cities": [CityResponse.model_validate(c) for c in cities]}


@router.get("/cities/with-stats")
def get_cities_with_stats(
    year: int = 2023,
    db: Session = Depends(get_db)
):
    """获取城市列表(含最新统计数据)"""
    stats_map = {}
    stats_list = db.query(CityStats).filter(CityStats.year == year).all()
    for s in stats_list:
        stats_map[s.city_id] = s

    cities = db.query(City).filter(City.level == 2).all()

    result = []
    for c in cities:
        stats = stats_map.get(c.id)
        result.append({
            "id": c.id,
            "name": c.name,
            "province": c.province,
            "lat": c.lat,
            "lng": c.lng,
            "level": c.level,
            "city_code": c.city_code,
            "stats": {
                "id": stats.id if stats else None,
                "city_id": stats.city_id if stats else None,
                "year": stats.year if stats else None,
                "per_gdp": stats.per_gdp if stats else None,
                "per_income": stats.per_income if stats else None,
                "house_price": stats.house_price if stats else None,
                "population": stats.population if stats else None,
                "gdp_total": stats.gdp_total if stats else None,
            } if stats else None
        })

    return {"cities": result, "year": year}


@router.get("/cities/{city_id}")
def get_city(city_id: int, db: Session = Depends(get_db)):
    """获取单个城市详情"""
    city = db.query(City).filter(City.id == city_id).first()
    if not city:
        return {"error": "City not found"}, 404
    return CityWithStats.model_validate(city)


@router.get("/trend/{city_id}")
def get_city_trend(
    city_id: int,
    metrics: str = "per_gdp,per_income,house_price",
    db: Session = Depends(get_db)
):
    """获取城市历年数据趋势"""
    city = db.query(City).filter(City.id == city_id).first()
    if not city:
        return {"error": "City not found"}, 404

    stats_list = (
        db.query(CityStats)
        .filter(CityStats.city_id == city_id)
        .order_by(CityStats.year)
        .all()
    )

    years = [s.year for s in stats_list]
    metric_list = metrics.split(",")

    trend_data = {}
    for m in metric_list:
        trend_data[m] = [getattr(s, m, None) for s in stats_list]

    # 也获取人口和GDP总量
    trend_data["population"] = [s.population for s in stats_list]
    trend_data["gdp_total"] = [s.gdp_total for s in stats_list]

    return TrendResponse(
        city=city.name,
        province=city.province,
        trend=TrendData(
            years=years,
            per_gdp=trend_data.get("per_gdp"),
            per_income=trend_data.get("per_income"),
            house_price=trend_data.get("house_price"),
            population=trend_data.get("population"),
            gdp_total=trend_data.get("gdp_total"),
        )
    )


@router.get("/provinces")
def get_provinces(db: Session = Depends(get_db)):
    """获取所有省份列表"""
    provinces = db.query(City.province).distinct().all()
    return {"provinces": [p[0] for p in provinces]}
