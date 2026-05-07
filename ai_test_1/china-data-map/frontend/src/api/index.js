import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 获取所有城市
export const getCities = () => api.get('/cities')

// 获取带数据的城市列表
export const getCitiesWithStats = (year = 2023) =>
  api.get('/cities/with-stats', { params: { year } })

// 获取城市详情
export const getCity = (cityId) => api.get(`/cities/${cityId}`)

// 获取城市趋势
export const getCityTrend = (cityId, metrics = 'per_gdp,per_income,house_price') =>
  api.get(`/trend/${cityId}`, { params: { metrics } })

// 获取省份排名
export const getProvinceRanking = (metric = 'per_gdp', year = 2023, limit = 34) =>
  api.get('/ranking/province', { params: { metric, year, limit } })

// 获取城市排名
export const getCityRanking = (metric = 'per_gdp', year = 2023, limit = 50, province = null) =>
  api.get('/ranking/city', { params: { metric, year, limit, province } })

// 获取散点图数据
export const getScatterData = (xMetric = 'per_income', yMetric = 'house_price', year = 2023) =>
  api.get('/scatter', { params: { x_metric: xMetric, y_metric: yMetric, year } })

export default api
