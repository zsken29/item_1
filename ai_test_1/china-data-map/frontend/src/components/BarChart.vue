<template>
  <div ref="chartRef" class="bar-chart"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Array, default: () => [] },
  metric: { type: String, default: 'per_gdp' },
  type: { type: String, default: 'province' }
})

const chartRef = ref(null)
let chart = null

const formatValue = (val) => {
  if (val == null) return '-'
  switch (props.metric) {
    case 'per_gdp': return val.toFixed(1) + '万'
    case 'per_income': return (val / 10000).toFixed(1) + '万'
    case 'house_price': return val.toFixed(0) + '元'
    case 'population': return val.toFixed(0) + '万'
    case 'gdp_total': return val.toFixed(0) + '亿'
    default: return val.toFixed(2)
  }
}

const initChart = () => {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value)
  updateChart()
}

const updateChart = () => {
  if (!chart) return

  const top10 = props.data.slice(0, 10)
  const labels = top10.map(d => props.type === 'province' ? d.province : d.city)
  const values = top10.map(d => d.value)

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(22, 33, 62, 0.9)',
      borderColor: '#e94560',
      textStyle: { color: '#fff' },
      formatter: (p) => `${p[0].name}<br/>${formatValue(p[0].value)}`
    },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#888', rotate: 45, fontSize: 10 },
      axisLine: { lineStyle: { color: '#0f3460' } }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#888', formatter: formatValue },
      axisLine: { lineStyle: { color: '#0f3460' } },
      splitLine: { lineStyle: { color: '#1a1a2e' } }
    },
    series: [{
      type: 'bar',
      data: values,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#e94560' },
          { offset: 1, color: '#0f3460' }
        ])
      },
      barRadius: 4
    }]
  }

  chart.setOption(option)
}

watch(() => props.data, updateChart, { deep: true })

onMounted(() => initChart())
onUnmounted(() => { if (chart) chart.dispose() })
</script>

<style scoped>
.bar-chart { width: 100%; height: 300px; }
</style>
