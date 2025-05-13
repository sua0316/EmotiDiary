from flask import Flask, request, jsonify
import subprocess
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)

# CNN 모델 로드
cnn_model = load_model("cnn_emotion_model.h5")

# 감정 매핑 (CNN에서 사용)
emotion_mapping = {
    "fear": 0,
    "surprise": 1,
    "angry": 2,
    "sad": 3,
    "neutral": 4,
    "happy": 5,
    "disgust": 6
}

# Llama 모델 통신 함수
def ollama_query(prompt):
    """
    Llama 모델에 프롬프트를 보내고 결과를 반환하는 함수.
    """
    try:
        result = subprocess.run(
            ['/usr/local/bin/ollama', 'run', 'llama3.2:latest'],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print("Error:", result.stderr)  # 에러 메시지 출력
        return result.stdout.strip()
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {'/usr/local/bin/ollama'}")
        return None

# Llama 감정 예측 함수
def predict_llama_emotion(predict_sentence, max_retries=3):
    """
    한국어 문장의 감정을 Llama 모델로 예측하는 함수.
    예상치 못한 출력 형식이나 오류 발생 시 기본값 반환.
    """
    # 한국어 프롬프트 생성
    prompt = (
        "다음 문장의 감정을 분석해주세요. "
        "결과는 반드시 이 형식으로만 출력해야 합니다: [감정], [확률]. "
        "감정의 범주는 반드시 'fear', 'surprise', 'angry', 'sad', 'neutral', 'happy', 'disgust' 중 하나여야 합니다. "
        "확률은 반드시 0에서 100 사이의 숫자로, 퍼센트 기호(%)를 포함하지 않습니다. "
        "추가적인 설명이나 다른 형식의 출력은 절대 하지 마세요.\n"
        f"문장: {predict_sentence}\n"
    )

    valid_emotions = ['fear', 'surprise', 'angry', 'sad', 'neutral', 'happy', 'disgust']

    for attempt in range(max_retries):
        response = ollama_query(prompt)
        if response:
            try:
                # 응답 파싱
                response = response.strip()
                if "[" in response and "]" in response:
                    response = response.strip("[] ")
                emotion, confidence = map(str.strip, response.split(","))
                # 감정 값 확인
                if emotion not in valid_emotions:
                    continue  # 감정 값이 유효하지 않으면 재시도
                # 확률 값 확인 및 변환
                confidence = re.sub(r"[^\d.]", "", confidence)  # 숫자와 소수점만 남기기
                confidence = float(confidence) / 100  # 0~1 범위로 변환
                return emotion, confidence
            except (ValueError, IndexError, SyntaxError):
                continue  # 파싱 오류 발생 시 재시도

    # 모든 시도가 실패한 경우 기본값 반환
    return "neutral", 0.5

# CNN 예측 함수 정의
def predict_cnn_face(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(np.expand_dims(face_image, -1), 0)
    predictions = cnn_model.predict(face_image)
    predicted_label = np.argmax(predictions)
    emotion_labels_cnn = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return emotion_labels_cnn[predicted_label], predictions[0][predicted_label]

# 감정 결합 함수
def final_emotion(llama_result, cnn_result, llama_weight=0.7, cnn_weight=0.3):
    """
    Llama와 CNN 결과를 가중치 기반으로 결합하며, 감정이 같으면 확률을 합산.
    """
    # 결과 분리
    emotion_llama, prob_llama = llama_result
    emotion_cnn, prob_cnn = cnn_result

    # 가중치 적용한 확률 계산
    weighted_prob_llama = prob_llama * llama_weight
    weighted_prob_cnn = prob_cnn * cnn_weight

    # 감정이 같으면 확률 합산
    if emotion_llama == emotion_cnn:
        combined_emotion = emotion_llama
        combined_prob = weighted_prob_llama + weighted_prob_cnn
    else:
        # 감정이 다를 경우, 가중치 기반으로 높은 확률 선택
        if weighted_prob_llama >= weighted_prob_cnn:
            combined_emotion = emotion_llama
            combined_prob = weighted_prob_llama
        else:
            combined_emotion = emotion_cnn
            combined_prob = weighted_prob_cnn

    return {"emotion": combined_emotion, "confidence": combined_prob}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 텍스트 처리 (Llama 예측)
    text = data['text']
    emotion_text, prob_text = predict_llama_emotion(text)

    # 이미지 처리 (CNN 예측)
    image_data = base64.b64decode(data['image'])
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    emotion_face, prob_face = predict_cnn_face(image)

    # 감정 결과 결합
    final_result = final_emotion((emotion_text, prob_text), (emotion_face, prob_face))
    return jsonify({'emotion': final_result['emotion'], 'confidence': final_result['confidence']})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
