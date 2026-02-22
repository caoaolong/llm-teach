import os

from flask import request, jsonify
import numpy as np
import onnxruntime as ort

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


def predict():
    if ort_session is None:
        return jsonify({'error': '模型未加载'}), 400

    try:
        data = request.json
        text = data['text']

        # 假设有一个函数 `preprocess_text` 可以将文本转换为模型所需的输入格式
        input_tensor = preprocess_text(text)

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


def preprocess_text(text):
    """
    将文本转换为模型所需的输入格式。
    这里只是一个示例实现，具体实现取决于模型的需求。
    """
    # 示例：简单地将文本转换为固定长度的向量
    max_len = 128
    input_ids = [ord(c) for c in text][:max_len]  # 简化的字符编码
    input_ids += [0] * (max_len - len(input_ids))  # 填充至最大长度
    return np.array([input_ids], dtype=np.int64)