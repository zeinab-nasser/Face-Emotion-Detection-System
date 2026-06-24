ideo_transformer_factory=SmartEmotionTransformer)import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model.predict import make_prediction

st.set_page_config(page_title="Face Emotion Detection System", page_icon="🧠", layout="wide")

st.title("🧠 Face Emotion Detection System")
st.write("An end-to-end Deep Learning application using DeepFace & Streamlit.")

mode = st.sidebar.selectbox("Choose Mode", ["Upload Image", "Live Webcam (Beta)"])

# ─── Upload Image Mode ───────────────────────────────────────────────────────
if mode == "Upload Image":
    st.subheader("📸 Image Upload Mode")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=400)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing emotions... Please wait..."):
            result = make_prediction(opencv_image)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                dominant_emotion = result['dominant_emotion']
                all_emotions = result['emotion']

                st.success(f"**Dominant Emotion:** {dominant_emotion.upper()}")
                st.write("### Emotion Breakdown:")
                for emotion, score in all_emotions.items():
                    st.write(f"{emotion.capitalize()}: {score:.2f}%")
                    st.progress(min(float(score) / 100.0, 1.0))

# ─── Live Webcam Mode ────────────────────────────────────────────────────────
elif mode == "Live Webcam (Beta)":
    st.subheader("Real-time Smart Live Video Mode")
    st.write("The camera is analyzing emotions in real-time.")

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    from deepface import DeepFace

    # تحميل الـ model مرة واحدة بس عشان منهدرش memory في كل frame
    @st.cache_resource
    def load_deepface_model():
        DeepFace.build_model("Emotion")
        return True

    load_deepface_model()

    class SmartEmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.stable_faces = {}
            self.MAX_ABSENT_FRAMES = 15
            self.frame_count = 0
            # بنحلل كل 10 frames بدل كل frame عشان نوفر memory وCPU
            self.ANALYZE_EVERY_N_FRAMES = 10

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # زيادة عداد الغياب لكل الوجوه
            for face_id in self.stable_faces:
                self.stable_faces[face_id]['absent_count'] += 1

            # بنحلل بس كل ANALYZE_EVERY_N_FRAMES
            if self.frame_count % self.ANALYZE_EVERY_N_FRAMES == 0:
                try:
                    # تصغير الصورة قبل التحليل بيوفر memory كتير
                    small = cv2.resize(img, (320, 240))
                    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    analysis_list = DeepFace.analyze(
                        rgb_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )

                    # حساب نسبة التكبير عشان نرجع الـ bounding box للحجم الأصلي
                    scale_x = img.shape[1] / 320
                    scale_y = img.shape[0] / 240

                    for i, face_data in enumerate(analysis_list):
                        dominant_emotion = face_data['dominant_emotion']
                        confidence = face_data['emotion'][dominant_emotion]
                        region = face_data['region']

                        # تعديل الـ bounding box على أساس الحجم الأصلي
                        x = int(region['x'] * scale_x)
                        y = int(region['y'] * scale_y)
                        w = int(region['w'] * scale_x)
                        h = int(region['h'] * scale_y)

                        self.stable_faces[i] = {
                            'text': f"{dominant_emotion.upper()} ({confidence:.1f}%)",
                            'bbox': (x, y, w, h),
                            'absent_count': 0
                        }
                except Exception:
                    pass

            # تنظيف الوجوه اللي غابت
            self.stable_faces = {
                face_id: data for face_id, data in self.stable_faces.items()
                if data['absent_count'] < self.MAX_ABSENT_FRAMES
            }

            # رسم النتائج على الفريم
            for face_id, data in self.stable_faces.items():
                x, y, w, h = data['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, data['text'], (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            return img

    webrtc_streamer(key="smart-emotion-detection", video_transformer_factory=SmartEmotionTransformer)
