# EmotiDiary
**LLaMA와 CNN을 활용한 감정 분석 기반 일기 앱**

## 개요  
EmotiDiary는 사용자의 일기 텍스트와 셀카 이미지를 기반으로 감정을 분석하고 시각화하여, 정신 건강을 자연스럽게 관리할 수 있도록 돕는 AI 기반 감정 일기 앱입니다. 특히 MZ세대를 대상으로, 글쓰기 습관을 활용한 자기돌봄과 감정 인식 기능을 제공합니다.

- 텍스트 + 이미지 기반 감정 인식
- 감정 캘린더와 통계 시각화 기능 제공
- 이모지 기반 공유 기능으로 우울증 예방 기여

## 개발 동기  
현대 사회에서 심화되고 있는 정신 건강 문제, 특히 우울증에 대한 무감증을 해결하고자 시작된 프로젝트입니다. 기존 감정 분석 앱들이 단순히 텍스트 입력 또는 직접 이모티콘을 선택하게 하는 것과 달리, EmotiDiary는 **텍스트 + 이미지**를 모두 분석해 더 높은 정확도의 감정 인식을 목표로 했습니다.

## 주요 기능  
- **멀티모달 감정 분석**: 일기 텍스트 + 셀카 이미지를 입력으로 감정 예측  
- **감정 캘린더**: 하루 단위 이모지 기반 감정 기록  
- **감정 통계 분석**: 감정 변화 추이를 시각화하여 패턴 인식 가능  
- **감정 공유 기능**: 친구들과 감정 이모지를 공유하며 소통  
- **갤러리 뷰**: 저장된 일기 이미지들을 시각적으로 정리

## 시스템 구조  
- **텍스트 감정 분석**: LLaMA (Generative AI 기반 LLM)  
  → 감정 7개(fear, surprise, angry, sad, neutral, happy, disgust)에 대한 확률 출력  
- **이미지 감정 분석**: CNN 기반 얼굴 감정 인식 모델  
- **결합 모델**: 두 모델 결과가 동일할 경우, 확률을 합산하여 최종 감정 결정  
- **백엔드**: Firebase (이미지/텍스트 저장 및 불러오기)  
- **프론트엔드**: HCI 원칙 기반의 사용자 친화적 UI 설계

## 사용 기술 (Tech Stack)  
- Python  
- LLaMA (텍스트 감정 분석 모델)  
- TensorFlow/Keras (CNN 기반 얼굴 감정 모델)  
- OpenCV, pandas, NumPy  
- Flask (API 서버)  
- Firebase (서버 및 데이터 저장소)  
- matplotlib (시각화 도구)

## 스크린샷  
![figma](./figma1.png)
![app](./app1.png)
