"""
API Router - Statistics & Ranking
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from collections import defaultdict

from ..database import get_db
from ..models.city import City, CityStats
from ..schemas.ranking import RankingResponse, ProvinceRanking, CityRanking

router = APIRouter(prefix="/api", tags=["statistics"])


@router.get("/ranking/province")
def get_province_ranking(
    metric: str = "per_gdp",
    year: int = 2023,
    limit: int = 34,
    db: Session = Depends(get_db)
):
    """获取省份排名"""
    # 按省份聚合数据
    province_stats = (
        db.query(
            City.province,
            func.sum(CityStats.gdp_total).label("total_gdp"),
            func.sum(CityStats.population).label("total_pop"),
            func.avg(CityStats.per_gdp).label("avg_per_gdp"),
            func.avg(CityStats.per_income).label("avg_per_income"),
            func.avg(CityStats.house_price).label("avg_house_price"),
        )
        .join(CityStats, City.id == CityStats.city_id)
        .filter(CityStats.year == year)
        .group_by(City.province)
        .all()
    )

    ranking = []
    for ps in province_stats:
        value = None
        if metric == "per_gdp":
            value = ps.avg_per_gdp
        elif metric == "per_income":
            value = ps.avg_per_income
        elif metric == "house_price":
            value = ps.avg_house_price
        elif metric == "population":
            value = ps.total_pop
        elif metric == "gdp_total":
            value = ps.total_gdp

        if value is not None:
            ranking.append({
                "province": ps.province,
                "value": round(value, 2) if value else 0
            })

    # 排序
    ranking.sort(key=lambda x: x["value"], reverse=True)
    ranking = ranking[:limit]

    return RankingResponse(
        ranking=[
            ProvinceRanking(rank=i+1, province=r["province"], value=r["value"])
            for i, r in enumerate(ranking)
        ]
    )


@router.get("/ranking/city")
def get_city_ranking(
    metric: str = "per_gdp",
    year: int = 2023,
    limit: int = 50,
    province: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取城市排名"""
    query = (
        db.query(City, CityStats)
        .join(CityStats, City.id == CityStats.city_id)
        .filter(CityStats.year == year, City.level == 2)
    )

    if province:
        query = query.filter(City.province == province)

    results = query.all()

    ranking = []
    for city, stats in results:
        value = getattr(stats, metric, None)
        if value is not None:
            ranking.append({
                "city": city.name,
                "province": city.province,
                "value": round(value, 2)
            })

    ranking.sort(key=lambda x: x["value"], reverse=True)
    ranking = ranking[:limit]

    return RankingResponse(
        ranking=[
            CityRanking(rank=i+1, city=r["city"], province=r["province"], value=r["value"])
            for i, r in enumerate(ranking)
        ]
    )


@router.get("/scatter")
def get_scatter_data(
    x_metric: str = "per_income",
    y_metric: str = "house_price",
    year: int = 2023,
    db: Session = Depends(get_db)
):
    """获取散点图数据（用于展示两个指标的关系）"""
    results = (
        db.query(City, CityStats)
        .join(CityStats, City.id == CityStats.city_id)
        .filter(CityStats.year == year, City.level == 2)
        .all()
    )

    data = []
    for city, stats in results:
        x_value = getattr(stats, x_metric, None)
        y_value = getattr(stats, y_metric, None)
        if x_value and y_value:
            data.append({
                "name": city.name,
                "province": city.province,
                "x": round(x_value, 2),
                "y": round(y_value, 2)
            })

    return {"data": data, "x_metric": x_metric, "y_metric": y_metric}
