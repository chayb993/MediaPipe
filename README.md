# MediaPipe 예제 모음

MediaPipe 라이브러리를 활용하여 카메라 및 동영상 스트림에서 손 검출, 얼굴 메시 검출, 셀피 배경 분리(Segmentation)를 수행하는 파이썬 실습 프로젝트입니다.

## 📁 프로젝트 파일 구조

* `hand_detector.py` - MediaPipe를 사용한 실시간 손 인식 및 랜드마크 시각화 코드
* `face_detector.py` - 실시간 카메라 기반 얼굴 메시(Face Mesh) 검출 코드
* `face_detector_youtube.py` - 유튜브 동영상 링크 주소를 파싱하여 얼굴 메시를 검출하는 코드
* `selfie_segmentation.py` - 인물과 배경을 분리하여 아웃포커싱(GaussianBlur) 효과를 주는 코드
* `requirements.txt` - 프로젝트 실행에 필요한 라이브러리 목록

## ⚙️ 설치 및 실행 방법

1. **필수 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **예제 코드 실행 (예시)**
   ```bash
   python selfie_segmentation.py
   ```
