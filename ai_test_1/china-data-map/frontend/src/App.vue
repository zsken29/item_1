<template>
  <div class="dashboard">
    <header class="header">
      <h1>中国城市人均数据地图</h1>
      <div class="controls">
        <select v-model="selectedMetric" @change="onMetricChange">
          <option value="per_gdp">人均GDP</option>
          <option value="per_income">人均可支配收入</option>
          <option value="house_price">商品房均价</option>
          <option value="population">常住人口</option>
          <option value="gdp_total">GDP总量</option>
        </select>
        <select v-model="selectedYear" @change="onYearChange">
          <option value="2023">2023年</option>
          <option value="2022">2022年</option>
          <option value="2021">2021年</option>
        </select>
      </div>
    </header>

    <div class="content">
      <div class="map-container">
        <ChinaMap
          :data="mapData"
          :metric="selectedMetric"
          :selectedCity="selectedCity"
          @city-click="onCityClick"
        />
      </div>

      <div class="charts-grid">
        <div class="chart-item">
          <h3>省份排名 TOP10</h3>
          <BarChart :data="provinceRanking" :metric="selectedMetric" />
        </div>
        <div class="chart-item">
          <h3>城市排名 TOP10</h3>
          <BarChart :data="cityRanking" :metric="selectedMetric" type="city" />
        </div>
        <div class="chart-item">
          <h3>历年趋势</h3>
          <LineChart v-if="selectedCity" :cityId="selectedCity.id" :cityName="selectedCity.name" />
          <div v-else class="placeholder">点击地图选择城市</div>
        </div>
        <div class="chart-item">
          <h3>{{ metricLabels[selectedMetric] }} vs 人均可支配收入</h3>
          <ScatterChart :xMetric="'per_income'" :yMetric="selectedMetric" />
        </div>
      </div>
    </div>

    <div v-if="selectedCity" class="data-panel">
      <h3>{{ selectedCity.name }} - 数据详情</h3>
      <div class="data-grid">
        <div class="data-item">
          <span class="label">人均GDP</span>
          <span class="value">{{ selectedCity.stats?.per_gdp || '-' }} 万元</span>
        </div>
        <div class="data-item">
          <span class="label">人均收入</span>
          <span class="value">{{ selectedCity.stats?.per_income || '-' }} 元</span>
        </div>
        <div class="data-item">
          <span class="label">房价</span>
          <span class="value">{{ selectedCity.stats?.house_price || '-' }} 元/㎡</span>
        </div>
        <div class="data-item">
          <span class="label">常住人口</span>
          <span class="value">{{ selectedCity.stats?.population || '-' }} 万人</span>
        </div>
        <div class="data-item">
          <span class="label">GDP总量</span>
          <span class="value">{{ selectedCity.stats?.gdp_total || '-' }} 亿元</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import ChinaMap from './components/ChinaMap.vue'
import BarChart from './components/BarChart.vue'
import LineChart from './components/LineChart.vue'
import ScatterChart from './components/ScatterChart.vue'
import * as api from './api/index.js'

const selectedMetric = ref('per_gdp')
const selectedYear = ref(2023)
const selectedCity = ref(null)
const allCities = ref([])

const metricLabels = {
  per_gdp: '人均GDP',
  per_income: '人均可支配收入',
  house_price: '商品房均价',
  population: '常住人口',
  gdp_total: 'GDP总量'
}

const mapData = computed(() => {
  return allCities.value.map(c => ({
    name: c.name,
    value: c.stats?.[selectedMetric.value] || 0,
    province: c.province,
    lng: c.lng,
    lat: c.lat,
    id: c.id
  }))
})

const provinceRanking = ref([])
const cityRanking = ref([])

const loadData = async () => {
  try {
    const [citiesRes, provinceRes, cityRes] = await Promise.all([
      api.getCitiesWithStats(selectedYear.value),
      api.getProvinceRanking(selectedMetric.value, selectedYear.value),
      api.getCityRanking(selectedMetric.value, selectedYear.value)
    ])
    allCities.value = citiesRes.data.cities
    provinceRanking.value = provinceRes.data.ranking
    cityRanking.value = cityRes.data.ranking
  } catch (err) {
    console.error('加载数据失败:', err)
  }
}

const onMetricChange = () => loadData()
const onYearChange = () => loadData()

const onCityClick = (city) => {
  selectedCity.value = allCities.value.find(c => c.id === city.id) || null
}

onMounted(() => loadData())
</script>

<style scoped>
.dashboard {
  min-height: 100vh;
  background: #1a1a2e;
  color: #fff;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 40px;
  background: #16213e;
  border-bottom: 1px solid #0f3460;
}
.header h1 {
  font-size: 24px;
  color: #e94560;
}
.controls {
  display: flex;
  gap: 16px;
}
.controls select {
  padding: 8px 16px;
  background: #0f3460;
  color: #fff;
  border: 1px solid #e94560;
  border-radius: 4px;
  cursor: pointer;
}
.content {
  padding: 20px;
}
.map-container {
  height: 500px;
  background: #16213e;
  border-radius: 8px;
  margin-bottom: 20px;
}
.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}
.chart-item {
  background: #16213e;
  border-radius: 8px;
  padding: 16px;
}
.chart-item h3 {
  margin-bottom: 16px;
  color: #e94560;
  font-size: 16px;
}
.placeholder {
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #666;
}
.data-panel {
  position: fixed;
  right: 20px;
  top: 100px;
  width: 280px;
  background: #16213e;
  border-radius: 8px;
  padding: 20px;
  border: 1px solid #e94560;
}
.data-panel h3 {
  color: #e94560;
  margin-bottom: 16px;
}
.data-grid {
  display: grid;
  gap: 12px;
}
.data-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #0f3460;
}
.data-item .label {
  color: #888;
}
.data-item .value {
  color: #fff;
  font-weight: bold;
}
</style>
