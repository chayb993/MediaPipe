#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
from yt_dlp import YoutubeDL

# =========================================
# 🧩 Mediapipe 얼굴 메시(Face Mesh) 초기화
# =========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================
# 📺 유튜브 동영상 링크 설정
# =========================================
youtube_url = "https://youtube.com" 

print("🔗 유튜브 영상 스트림 분석 중...")

# 💡 오류를 일으키던 'cookiesfrombrowser' 옵션을 완전히 제거했습니다.
ydl_opts = {
    'format': 'best[ext=mp4]/best',  # MP4 포맷 중 최고 화질
    'quiet': True,
}

try:
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_stream_url = info_dict.get('url')
except Exception as e:
    print(f"❌ 유튜브 영상을 불러오지 못했습니다. 원인:\n{e}")
    exit()

# =========================================
# 📸 비디오 스트림 연결
# =========================================
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ OpenCV가 비디오 스트림 주소를 열지 못했습니다.")
    exit()

print("📷 비디오 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상이 끝났거나 프레임을 읽지 못했습니다.")
        break

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_mesh.process(image_rgb)

    # 👤 얼굴 랜드마크 표시
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 1. 얼굴 표면 그물망
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            # 2. 눈, 코, 입 윤곽선
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            # 3. 눈동자
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
            )

    # 화면 크기 조정
    h, w = image.shape[:2]
    if h > 720:
        image = cv2.resize(image, (int(w * 720 / h), 720))

    # 화면 표시
    cv2.imshow('👤 MediaPipe Face Mesh Detector (YouTube)', image)

    # ESC 키로 종료
    if cv2.waitKey(10) & 0xFF == 27:
        print("👋 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()

