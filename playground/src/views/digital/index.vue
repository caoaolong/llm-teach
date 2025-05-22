<template>
  <div class="digital-container">
    <!-- 左侧画板区域 -->
    <div class="drawing-section">
      <div class="drawing-area">
        <canvas ref="canvas" width="280" height="280" @mousedown="startDrawing" @mousemove="draw" @mouseup="stopDrawing"
          @mouseleave="stopDrawing"></canvas>
        <div class="canvas-controls">
          <el-button type="primary" @click="clearCanvas">清除画板</el-button>
          <el-button type="success" @click="predict" :disabled="!modelLoaded || !canPredict">
            开始识别
          </el-button>
        </div>
      </div>
    </div>

    <!-- 右侧预测结果区域 -->
    <div class="predict-section">
      <div class="model-status">
        <div class="model-input">
          <el-input
            v-model="modelPath"
            placeholder="输入模型路径，如: /models/mnist.onnx"
            :disabled="modelLoaded"
          >
            <template #prepend>/models/</template>
          </el-input>
          <el-button 
            type="primary" 
            @click="loadModel" 
            :loading="loading"
            :disabled="!modelPath || modelLoaded"
          >
            加载模型
          </el-button>
        </div>
        <div class="model-info">
          <el-tag :type="modelLoaded ? 'success' : 'info'">
            模型状态：{{ modelLoaded ? '已加载' : '未加载' }}
          </el-tag>
        </div>
      </div>
      <div class="predict-result" v-if="modelLoaded">
        <h3>预测结果</h3>
        <div class="result-display" v-if="prediction !== null">
          <span class="number">{{ prediction }}</span>
          <div class="confidence">
            可信度: {{ confidence.toFixed(2) }}%
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import { mnistApi } from '@/api/mnist'

export default {
  name: 'DigitalRecognition',
  data() {
    return {
      isDrawing: false,
      context: null,
      session: null,
      modelLoaded: false,
      loading: false,
      prediction: null,
      confidence: 0,
      canPredict: false,
      modelPath: '', // 新增的 data 属性
    }
  },
  async mounted() {
    this.initCanvas()
  },
  methods: {
    initCanvas() {
      const canvas = this.$refs.canvas
      this.context = canvas.getContext('2d')
      this.context.lineWidth = 15
      this.context.lineJoin = 'round'
      this.context.lineCap = 'round'
      this.context.strokeStyle = '#ffffff'
      this.clearCanvas()
    },
    startDrawing(event) {
      this.isDrawing = true
      const rect = this.$refs.canvas.getBoundingClientRect()
      this.context.beginPath()
      this.context.moveTo(
        event.clientX - rect.left,
        event.clientY - rect.top
      )
    },
    draw(event) {
      if (!this.isDrawing) return
      const rect = this.$refs.canvas.getBoundingClientRect()
      this.context.lineTo(
        event.clientX - rect.left,
        event.clientY - rect.top
      )
      this.context.stroke()
      this.canPredict = true
    },
    stopDrawing() {
      this.isDrawing = false
    },
    clearCanvas() {
      this.context.fillStyle = 'var(--el-bg-color)'
      this.context.fillRect(0, 0, 280, 280)
      this.prediction = null
      this.confidence = 0
      this.canPredict = false
    },
    async loadModel() {
      if (!this.modelPath) {
        this.$message.warning('请输入模型路径')
        return
      }

      this.loading = true
      try {
        await mnistApi.loadModel('models/' + (this.modelPath || 'mnist.onnx'))
        this.modelLoaded = true
        this.$message.success('模型加载成功')
      } catch (error) {
        console.error('模型加载失败:', error)
        this.$message.error('模型加载失败，请检查路径是否正确')
      }
      this.loading = false
    },

    async predict() {
      if (!this.modelLoaded) return

      try {
        const imageData = this.$refs.canvas.toDataURL('image/png')
        const result = await mnistApi.predict(imageData)
        
        this.prediction = result.prediction
        this.confidence = result.confidence
      } catch (error) {
        console.error('预测失败:', error)
        this.$message.error('预测失败')
      }
    }
  }
}
</script>

<style scoped>
.digital-container {
  display: flex;
  flex-direction: row; /* 改为水平方向 */
  gap: 24px;
  padding: 32px;
  margin: 0 auto;
  background-color: var(--el-bg-color);
  color: var(--el-text-color-primary);
}

.drawing-section {
  flex: 1; /* 占据剩余空间 */
  background-color: var(--el-bg-color-overlay);
  border-radius: 8px;
  padding: 24px;
  box-shadow: var(--el-box-shadow-dark);
  border: 1px solid var(--el-border-color-darker);
}

.predict-section {
  width: 380px; /* 固定宽度 */
  background-color: var(--el-bg-color-overlay);
  border-radius: 8px;
  padding: 24px;
  box-shadow: var(--el-box-shadow-dark);
  border: 1px solid var(--el-border-color-darker);
}

.drawing-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.drawing-area canvas {
  border: 2px solid var(--el-border-color);
  background: var(--el-bg-color-overlay);
  cursor: crosshair;
  border-radius: 4px;
}

.canvas-controls {
  display: flex;
  gap: 16px;
}

.model-status {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background-color: var(--el-bg-color);
  border-radius: 4px;
  margin-bottom: 24px;
  border: 1px solid var(--el-border-color);
}

.model-input {
  display: flex;
  gap: 16px;
  align-items: center;
}

.model-input :deep(.el-input) {
  flex: 1;
}

.model-info {
  display: flex;
  justify-content: center;
}

.predict-result {
  text-align: center;
}

.predict-result h3 {
  font-size: 20px;
  color: var(--el-text-color-primary);
  margin-bottom: 24px;
}

.result-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 24px;
  background-color: var(--el-bg-color);
  border-radius: 8px;
  border: 1px solid var(--el-border-color);
}

.result-display .number {
  font-size: 64px;
  font-weight: bold;
  color: var(--el-color-primary);
}

.confidence {
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
</style>