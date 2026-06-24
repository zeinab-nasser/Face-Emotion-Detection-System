import streamlit as st
import cv2
import numpy as np
from PIL import Image
# استيراد الدالة make_prediction من ملف predict.py
from model.predict import make_prediction

st.set_page_config(page_title="Face Emotion Detection System", page_icon="🧠", layout="wide")

st.title("🧠 Face Emotion Detection System")
st.write("An end-to-end Deep Learning application using DeepFace & Streamlit.")

mode = st.sidebar.selectbox("Choose Mode", ["Upload Image", "Live Webcam (Beta)"])

if mode == "Upload Image":
    st.subheader("📸 Image Upload Mode")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=400)
        
        # تحويل الصورة من PIL Image لـ OpenCV BGR format
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

# ----------------- تعديل المود للـ Live Webcam -----------------
elif mode == "Live Webcam (Beta)":
    st.subheader("🎥 Real-time Smart Live Video Mode")
    st.write("The camera is analyzing multiple emotions in real-time with automatic cleanup!")

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

    class SmartEmotionTransformer(VideoTransformerBase):
        def __init__(self):
            # قاموس لتخزين الوجوه المتسيفة مع عداد الغياب لكل وجه
            self.stable_faces = {}
            self.MAX_ABSENT_FRAMES = 15

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # زيادة عداد الغياب لكل الوجوه المتاحة في القاموس
            for face_id in self.stable_faces:
                self.stable_faces[face_id]['absent_count'] += 1
            
            try:
                # 1. تحليل الفريم بالكامل لمعرفة المشاعر لكل الوجوه المتاحة
                from deepface import DeepFace
                # بتحويل الفريم من BGR لـ RGB لأن DeepFace بيشتغل على RGB
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                analysis_list = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                
                # 2. تحديث قاموس الوجوه المتسيفة بالوجوه الجديدة اللي اتعرفت في الفريم الحالي
                for i, face_data in enumerate(analysis_list):
                    dominant_emotion = face_data['dominant_emotion']
                    confidence = face_data['emotion'][dominant_emotion]
                    region = face_data['region']
                    
                    self.stable_faces[i] = {
                        'text': f"{dominant_emotion.upper()} ({confidence:.1f}%)",
                        'bbox': (region['x'], region['y'], region['w'], region['h']),
                        'absent_count': 0  # موجود الآن في الكادر
                    }
            except Exception as e:
                pass
            
            # 3. تنظيف الوجوه اللي غابت عن الكادر لأكثر من الحد المسموح
            self.stable_faces = {
                face_id: data for face_id, data in self.stable_faces.items() 
                if data['absent_count'] < self.MAX_ABSENT_FRAMES
            }
            
            # 4. الرسم على الفريم الحالي: رسم المربعات والنصوص لكل الوجوه المتاحة
            for face_id, data in self.stable_faces.items():
                x, y, w, h = data['bbox']
                # رسم المربع حول الوجه
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # كتابة النص فوق المربع
                cv2.putText(img, data['text'], (x, y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            return img

    # تشغيل الـ WebRTC streamer مع الـ transformer الذكي
    webrtc_streamer(key="smart-emotion-detection", video_transformer_factory=SmartEmotionTransformer)