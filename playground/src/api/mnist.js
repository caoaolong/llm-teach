const BASE_URL = 'http://localhost:5000/mnist'

export const mnistApi = {
  /**
   * 加载模型
   * @param {string} modelPath - 模型路径
   * @returns {Promise}
   */
  async loadModel(modelPath) {
    const response = await fetch(`${BASE_URL}/load_model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_path: modelPath
      })
    })

    if (!response.ok) {
      throw new Error('模型加载失败')
    }

    const data = await response.json()
    if (data.error) {
      throw new Error(data.error)
    }

    return data
  },

  /**
   * 预测手写数字
   * @param {string} imageData - base64格式的图像数据
   * @returns {Promise}
   */
  async predict(imageData) {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageData
      })
    })

    if (!response.ok) {
      throw new Error('预测失败')
    }

    const data = await response.json()
    if (data.error) {
      throw new Error(data.error)
    }

    return data
  }
}