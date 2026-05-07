<template>
  <div ref="chartRef" class="china-map"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Array, default: () => [] },
  metric: { type: String, default: 'per_gdp' },
  selectedCity: { type: Object, default: null }
})

const emit = defineEmits(['cityClick'])
const chartRef = ref(null)
let chart = null

const formatValue = (val, metric) => {
  if (val == null) return '-'
  switch (metric) {
    case 'per_gdp': return val.toFixed(1) + '万'
    case 'per_income': return val.toFixed(0) + '元'
    case 'house_price': return val.toFixed(0) + '元/㎡'
    case 'population': return val.toFixed(0) + '万'
    case 'gdp_total': return val.toFixed(0) + '亿'
    default: return val
  }
}

const initChart = () => {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value)

  const option = {
    backgroundColor: '#16213e',
    title: {
      text: '中国城市数据分布图',
      subtext: '人均指标热力图',
      textStyle: { color: '#e94560' },
      subtextStyle: { color: '#888' },
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(22, 33, 62, 0.9)',
      borderColor: '#e94560',
      textStyle: { color: '#fff' },
      formatter: (p) => {
        if (p.data) {
          return `<strong>${p.data.name}</strong><br/>${p.seriesName}: ${formatValue(p.data.value, props.metric)}`
        }
        return ''
      }
    },
    geo: {
      map: 'china',
      roam: true,
      zoom: 1.2,
      center: [105, 36],
      label: { show: false },
      itemStyle: {
        areaColor: '#0f3460',
        borderColor: '#1a4a7a',
        borderWidth: 1
      },
      emphasis: {
        itemStyle: { areaColor: '#1a6a9a' },
        label: { show: true, color: '#fff' }
      }
    },
    visualMap: {
      min: 0,
      max: 20,
      text: ['高', '低'],
      textStyle: { color: '#fff' },
      realtime: false,
      calculable: true,
      inRange: { color: ['#0a3d62', '#1e8449', '#f1c40f', '#e74c3c', '#9b59b6'] },
      seriesIndex: [0]
    },
    series: [{
      name: props.metric,
      type: 'scatter',
      coordinateSystem: 'geo',
      symbolSize: (val, p) => {
        const base = 8
        const max = Math.max(...props.data.map(d => d.value || 0), 1)
        return base + (p.data.value / max) * 20
      },
      data: props.data.map(d => ({
        name: d.name,
        value: [d.lng, d.lat, d.value || 0],
        id: d.id
      })),
      label: { show: false },
      emphasis: {
        scale: 1.5,
        label: { show: true, formatter: '{b}', color: '#fff' }
      }
    }]
  }

  chart.setOption(option)

  chart.on('click', (p) => {
    if (p.data) {
      emit('cityClick', { id: p.data.id, name: p.data.name })
    }
  })
}

const updateChart = () => {
  if (!chart) return
  const maxVal = Math.max(...props.data.map(d => d.value || 0), 1)
  const visualMapMax = props.metric === 'per_income' ? 80000 :
                        props.metric === 'house_price' ? 40000 :
                        props.metric === 'population' ? 3000 :
                        props.metric === 'gdp_total' ? 50000 : 20

  chart.setOption({
    visualMap: { max: visualMapMax },
    series: [{
      name: props.metric,
      data: props.data.map(d => ({
        name: d.name,
        value: [d.lng, d.lat, d.value || 0],
        id: d.id
      }))
    }]
  })
}

watch(() => [props.data, props.metric], updateChart, { deep: true })

onMounted(() => {
  fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_china.json')
    .then(r => r.json())
    .then(geoData => {
      echarts.registerMap('china', geoData)
      initChart()
    })
})

onUnmounted(() => {
  if (chart) chart.dispose()
})
</script>

<style scoped>
.china-map { width: 100%; height: 100%; min-height: 480px; }
</style>
