from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# onnxruntime session 全局变量
ort_session = None


def load_model(model_path):
    global ort_session
    try:
        ort_session = ort.InferenceSession(model_path)
        return True
    except Exception as e:
        print(f"加载ONNX模型失败: {str(e)}")
        return False


def softmax(x):
    x = x - np.max(x)  # 数值稳定性
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


@app.route('/mnist', methods=['POST'])
def predict():
    if ort_session is None:
        return jsonify({'error': '模型未加载'}), 400

    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # 转为灰度图，大小28x28，转numpy数组
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image).astype(np.float32) / 255.0

        # ONNX模型输入一般是NCHW格式，添加batch和channel维度
        input_tensor = image_array[np.newaxis, np.newaxis, :, :]

        # 获取模型输入名，方便调用
        input_name = ort_session.get_inputs()[0].name

        # 运行推理，结果是 list
        outputs = ort_session.run(None, {input_name: input_tensor})
        print(outputs)

        logits = outputs[0][0]  # 取 batch 0
        probs = softmax(logits)
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    try:
        data = request.json
        model_path = data['model_path']
        success = load_model(model_path)
        if success:
            return jsonify({'message': 'ONNX模型加载成功'})
        else:
            return jsonify({'error': 'ONNX模型加载失败'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
