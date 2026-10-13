#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 얼굴 메시(Face Mesh) 초기화
# =========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,        # 동영상 입력
    max_num_faces=2,                # 최대 인식할 얼굴 개수
    refine_landmarks=True,          # 눈동자 및 입술 주위 정밀 검출 활성화
    min_detection_confidence=0.5,   # 탐지 신뢰도
    min_tracking_confidence=0.5     # 추적 신뢰도
)

# =========================================
# 📸 카메라 또는 동영상 연결
# =========================================
# cap = cv2.VideoCapture(0) # 기본 카메라 사용시
cap = cv2.VideoCapture("face.mp4") # 동영상 파일 사용 시 (파일 이름에 맞게 수정하세요)

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상이 끝났거나 프레임을 읽지 못했습니다.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_mesh.process(image_rgb)

    # 👤 얼굴 랜드마크 표시
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 1. 얼굴 표면의 그물망(Mesh) 그리기 (그레이 색상 테두리)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            # 2. 눈, 눈썹, 입술 등 주요 윤곽선 그리기 (흰색 테두리)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            # 3. 눈동자(Iris) 그리기 (refine_landmarks=True 일 때 작동)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
            )

    # 화면에 맞게 크기 조정 (창이 잘리는 것 방지)
    h, w = image.shape[:2]
    if h > 720:
        image = cv2.resize(image, (int(w * 720 / h), 720))

    # 화면 표시
    cv2.imshow('👤 MediaPipe Face Mesh Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
