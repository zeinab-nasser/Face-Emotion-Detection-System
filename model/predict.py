import cv2
from deepface import DeepFace

def make_prediction(image_bgr):
    """
    بتاخد الصورة وتعمل التوقع وترجع النتيجة
    """
    try:
        predictions = DeepFace.analyze(img_path=image_bgr, actions=['emotion'], enforce_detection=False)
        return predictions[0]
    except Exception as e:
        return {"error": str(e)}