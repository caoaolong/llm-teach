<template>
  <div class="sentiment-container">
    <el-row :gutter="20">
      <!-- 左侧输入区域 -->
      <el-col :span="12">
        <el-card class="input-card">
          <template #header>
            <div class="card-header">
              <span>文本输入</span>
              <el-button type="primary" @click="predict" :loading="loading">
                开始预测
              </el-button>
            </div>
          </template>
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="15"
            placeholder="请输入要进行情感分析的文本..."
          />
        </el-card>
      </el-col>

      <!-- 右侧结果区域 -->
      <el-col :span="12">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <span>预测结果</span>
            </div>
          </template>
          
          <!-- 模型状态显示 -->
          <div class="model-status">
            <div class="model-input">
              <el-input
                v-model="modelPath"
                placeholder="输入模型路径，如: sentiment.onnx"
                :disabled="modelLoaded"
              >
                <template #prepend>/models/</template>
              </el-input>
              <el-button 
                type="primary" 
                @click="loadModel" 
                :loading="modelLoading"
                :disabled="!modelPath || modelLoaded"
              >
                加载模型
              </el-button>
            </div>
            <el-alert
              :title="modelStatus.title"
              :type="modelStatus.type"
              :description="modelStatus.description"
              show-icon
            />
          </div>

          <!-- 预测结果显示 -->
          <div v-if="result" class="prediction-result">
            <h3>情感倾向</h3>
            <el-progress 
              :percentage="result.score" 
              :color="result.score > 50 ? '#67C23A' : '#F56C6C'"
            />
            <div class="result-text">
              可信度: {{ result.sentiment }}
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { sentimentApi } from '@/api/sentiment'
import { ElMessage, ElAlert, ElButton, ElInput, ElCard, ElRow, ElCol, ElProgress, ElTag } from 'element-plus'

// 状态变量
const inputText = ref('')
const loading = ref(false)
const modelLoading = ref(false)
const result = ref(null)
const modelPath = ref('imdb.onnx')
const modelLoaded = ref(false)

// 模型状态
const modelStatus = reactive({
  title: '模型未加载',
  type: 'warning',
  description: '请先点击"加载模型"按钮加载情感分析模型'
})

// 加载模型
const loadModel = async () => {
  if (!modelPath.value) {
    ElMessage.warning('请输入模型路径')
    return
  }

  modelLoading.value = true
  try {
    await sentimentApi.loadModel('models/' + (modelPath.value || 'imdb.onnx'))
    modelStatus.title = '模型已就绪'
    modelStatus.type = 'success'
    modelStatus.description = '模型加载成功，可以开始预测'
    modelLoaded.value = true
    ElMessage.success('模型加载成功')
  } catch (error) {
    modelStatus.title = '模型加载失败'
    modelStatus.type = 'error'
    modelStatus.description = error.message
    modelLoaded.value = false
    ElMessage.error('模型加载失败，请检查路径是否正确')
  } finally {
    modelLoading.value = false
  }
}

// 预测函数
const predict = async () => {
  if (!modelLoaded.value) {
    ElMessage.warning('请先加载模型')
    return
  }

  if (!inputText.value.trim()) {
    ElMessage.warning('请输入要分析的文本')
    return
  }

  loading.value = true
  try {
    const resp = await sentimentApi.predict(inputText.value.trim())
    result.value = {
      sentiment: (resp.confidence * 100).toFixed(2),
      score: (resp.prediction * 100).toFixed(2)
    }
  } catch (error) {
    ElMessage.error('预测失败：' + error.message)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.sentiment-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.input-card,
.result-card {
  height: 100%;
}

.model-status {
  margin-bottom: 20px;
}

.model-input {
  margin-bottom: 16px;
  display: flex;
  gap: 16px;
  align-items: center;
}

.model-input :deep(.el-input) {
  flex: 1;
}

.prediction-result {
  margin-top: 20px;
}

.result-text {
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 10px;
}
</style>