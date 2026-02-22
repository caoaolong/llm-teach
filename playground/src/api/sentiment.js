const BASE_URL = 'http://localhost:5000/sentiment'

export const sentimentApi = {
  // 加载模型
  async loadModel(modelPath) {
    try {
      const response = await fetch(`${BASE_URL}/load`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_path: modelPath
        })
      })
      if (!response.ok) throw new Error('模型加载失败')
      return await response.json()
    } catch (error) {
      throw new Error('模型加载失败：' + error.message)
    }
  },

  // 预测文本情感
  async predict(text) {
    try {
      const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })
      if (!response.ok) throw new Error('预测请求失败')
      return await response.json()
    } catch (error) {
      throw new Error('预测失败：' + error.message)
    }
  },
}