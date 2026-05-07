from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base


class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, index=True)
    province = Column(String(50), nullable=False, index=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    level = Column(Integer, default=2)  # 1=省 2=地级市 3=县
    city_code = Column(String(20), unique=True)

    stats = relationship("CityStats", back_populates="city")


class CityStats(Base):
    __tablename__ = "city_stats"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    year = Column(Integer, nullable=False, index=True)
    per_gdp = Column(Float, nullable=True)  # 人均GDP(万元)
    per_income = Column(Float, nullable=True)  # 人均可支配收入(元)
    house_price = Column(Float, nullable=True)  # 商品房均价(元/㎡)
    population = Column(Float, nullable=True)  # 常住人口(万人)
    gdp_total = Column(Float, nullable=True)  # GDP总量(亿元)
    created_at = Column(DateTime, default=datetime.utcnow)

    city = relationship("City", back_populates="stats")
