<template>
  <div ref="chartRef" class="line-chart"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import * as api from '../api/index.js'

const props = defineProps({
  cityId: { type: Number, default: null },
  cityName: { type: String, default: '' }
})

const chartRef = ref(null)
let chart = null

const formatValue = (val, metric) => {
  if (val == null) return null
  switch (metric) {
    case 'per_gdp': return val
    case 'per_income': return val / 10000
    case 'house_price': return val
    case 'population': return val
    case 'gdp_total': return val
    default: return val
  }
}

const initChart = () => {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value)
  loadTrend()
}

const loadTrend = async () => {
  if (!props.cityId) return
  try {
    const res = await api.getCityTrend(props.cityId)
    const { trend } = res.data

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(22, 33, 62, 0.9)',
        borderColor: '#e94560',
        textStyle: { color: '#fff' }
      },
      legend: {
        data: ['人均GDP', '人均收入(万)', '房价', '人口'],
        textStyle: { color: '#888' },
        top: 0
      },
      grid: { left: '3%', right: '4%', bottom: '3%', top: '25%', containLabel: true },
      xAxis: {
        type: 'category',
        data: trend.years,
        axisLabel: { color: '#888' },
        axisLine: { lineStyle: { color: '#0f3460' } }
      },
      yAxis: [
        {
          type: 'value',
          name: 'GDP/收入',
          axisLabel: { color: '#888' },
          axisLine: { lineStyle: { color: '#0f3460' } },
          splitLine: { lineStyle: { color: '#1a1a2e' } }
        },
        {
          type: 'value',
          name: '房价/人口',
          axisLabel: { color: '#888' },
          axisLine: { lineStyle: { color: '#0f3460' } },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: '人均GDP',
          type: 'line',
          data: trend.per_gdp.map(v => formatValue(v, 'per_gdp')),
          smooth: true,
          lineStyle: { color: '#e94560', width: 2 }
        },
        {
          name: '人均收入(万)',
          type: 'line',
          yAxisIndex: 0,
          data: trend.per_income.map(v => formatValue(v, 'per_income')),
          smooth: true,
          lineStyle: { color: '#3498db', width: 2 }
        }
      ]
    }

    chart.setOption(option)
  } catch (err) {
    console.error('加载趋势数据失败:', err)
  }
}

watch(() => props.cityId, loadTrend)

onMounted(() => initChart())
onUnmounted(() => { if (chart) chart.dispose() })
</script>

<style scoped>
.line-chart { width: 100%; height: 300px; }
</style>
