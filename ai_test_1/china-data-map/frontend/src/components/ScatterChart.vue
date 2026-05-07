<template>
  <div ref="chartRef" class="scatter-chart"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import * as api from '../api/index.js'

const props = defineProps({
  xMetric: { type: String, default: 'per_income' },
  yMetric: { type: String, default: 'house_price' }
})

const chartRef = ref(null)
let chart = null

const formatValue = (val, metric) => {
  if (val == null) return '-'
  switch (metric) {
    case 'per_gdp': return val.toFixed(1) + '万'
    case 'per_income': return (val / 10000).toFixed(1) + '万'
    case 'house_price': return val.toFixed(0) + '元'
    case 'population': return val.toFixed(0) + '万'
    case 'gdp_total': return val.toFixed(0) + '亿'
    default: return val.toFixed(2)
  }
}

const metricLabels = {
  per_gdp: '人均GDP(万)',
  per_income: '人均收入(万)',
  house_price: '房价(元/㎡)',
  population: '人口(万)',
  gdp_total: 'GDP(亿)'
}

const initChart = () => {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value)
  loadData()
}

const loadData = async () => {
  try {
    const res = await api.getScatterData(props.xMetric, props.yMetric)
    const scatterData = res.data.data

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(22, 33, 62, 0.9)',
        borderColor: '#e94560',
        textStyle: { color: '#fff' },
        formatter: (p) => `${p.data.name}<br/>X: ${formatValue(p.data.x, props.xMetric)}<br/>Y: ${formatValue(p.data.y, props.yMetric)}`
      },
      grid: { left: '3%', right: '8%', bottom: '3%', top: '5%', containLabel: true },
      xAxis: {
        type: 'value',
        name: metricLabels[props.xMetric],
        nameLocation: 'center',
        nameGap: 30,
        axisLabel: { color: '#888' },
        axisLine: { lineStyle: { color: '#0f3460' } },
        splitLine: { lineStyle: { color: '#1a1a2e' } }
      },
      yAxis: {
        type: 'value',
        name: metricLabels[props.yMetric],
        nameLocation: 'center',
        nameGap: 50,
        axisLabel: { color: '#888' },
        axisLine: { lineStyle: { color: '#0f3460' } },
        splitLine: { lineStyle: { color: '#1a1a2e' } }
      },
      series: [{
        type: 'scatter',
        symbolSize: 12,
        data: scatterData.map(d => ({
          name: d.name,
          value: [d.x / (props.xMetric === 'per_income' ? 10000 : 1), d.y / (props.yMetric === 'per_income' ? 10000 : 1)],
          x: d.x,
          y: d.y
        })),
        itemStyle: { color: '#e94560', opacity: 0.7 }
      }]
    }

    chart.setOption(option)
  } catch (err) {
    console.error('加载散点图数据失败:', err)
  }
}

watch(() => [props.xMetric, props.yMetric], loadData)

onMounted(() => initChart())
onUnmounted(() => { if (chart) chart.dispose() })
</script>

<style scoped>
.scatter-chart { width: 100%; height: 300px; }
</style>
