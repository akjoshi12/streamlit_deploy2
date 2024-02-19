# import streamlit as st
# from ultralytics import YOLO
# import pyttsx3
# import cv2

# # Function for YOLO prediction and TTS
# def process_video(video_file):
#     model = YOLO(r'/Users/atrijoshi/Downloads/best_1.pt')  # Load YOLO model
#     engine = pyttsx3.init('nsss')  # Initialize TTS engine

#     cap = cv2.VideoCapture(video_file)
#     cap = cv2.VideoCapture(video_file, apiPreference=cv2.CAP_ANY)
#     set1 = set()
#     counter = 0

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         results = model(frame)

#         if len(results) > 0:  # Check for detections
#             for obj in results:
#                 class_id = obj['class_id']
#                 class_name = obj['name']
#                 set1.add(class_name)

#                 if len(set1) > counter:
#                     class_name = str(class_name)
#                     engine.say(class_name)
#                     engine.runAndWait()
#                     counter += 1

#             # Display the frame with detected objects (optional)
#             # ... (code for visualization using OpenCV)

#     cap.release()

# # Streamlit app setup
# st.title("YOLO Object Detection with TTS Demo")

# # File uploader for video
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

# if uploaded_file is not None:
#     with st.spinner("Processing video..."):
#         process_video(uploaded_file)
#     st.success("Video processed successfully!")

# import streamlit as st
# from ultralytics import YOLO
# import pyttsx3
# import cv2
# import numpy as np

# # Function for YOLO prediction and TTS
# def process_video(video_file):
#     model = YOLO(r'/Users/atrijoshi/Downloads/best_1.pt')  # Load YOLO model
#     engine = pyttsx3.init()  # Initialize TTS engine

#     # Check if video_file is None
#     if video_file is None:
#         st.error("Please upload a video file.")
#         return

#     # Convert the file-like object to a byte stream
#     video_bytes = video_file.read()

#     # Convert the byte stream to a numpy array
#     nparr = np.frombuffer(video_bytes, np.uint8)

#     # Decode the numpy array to an OpenCV image
#     cap = cv2.VideoCapture()
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     set1 = set()
#     counter = 0

#     while frame is not None:  # Ensure frame is not None
#         results = model(frame)

#         if len(results) > 0:  # Check for detections
#             for obj in results:
#                 class_id = obj['class_id']
#                 class_name = obj['name']
#                 set1.add(class_name)

#                 if len(set1) > counter:
#                     class_name = str(class_name)
#                     engine.say(class_name)
#                     engine.runAndWait()
#                     counter += 1

#         # Read the next frame
#         success, frame = cap.read()

#     cap.release()

# # Streamlit app setup
# st.title("YOLO Object Detection with TTS Demo")

# # File uploader for video
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

# if uploaded_file is not None:
#     with st.spinner("Processing video..."):
#         process_video(uploaded_file)
#     st.success("Video processed successfully!")

# Works - Version2 

# Version3 

# import streamlit as st
# import torch
# import cv2
# import pyttsx3
# from ultralytics import YOLO

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")

# # Function to speak text
# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# # Streamlit app
# def main():
#     st.title("Object Detection with YOLOv8")

#     # Start webcam
#     cap = cv2.VideoCapture(0)

#     FRAME_WINDOW = st.image([])

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB (Streamlit works with RGB, OpenCV with BGR)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform inference
#         results = model(frame_rgb)

#         # Draw results on the frame
#         labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#         n = len(labels)
#         if n:
#             for i in range(n):
#                 row = cords[i]
#                 if row[4] >= 0.4:  # Confidence threshold
#                     x1, y1, x2, y2 = int(row[0]*frame.shape[1]), int(row[1]*frame.shape[0]), int(row[2]*frame.shape[1]), int(row[3]*frame.shape[0])
#                     cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     label = model.names[int(labels[i])]
#                     cv2.putText(frame_rgb, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                     speak(label)  # Speak detected object name

#         # Display the frame
#         FRAME_WINDOW.image(frame_rgb)

#         # Break loop with a stop button
#         if st.button('Stop'):
#             break

#     cap.release()

# if __name__ == '__main__':
#     main()


# Version 4


import streamlit as st
from ultralytics import YOLO
import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_objects(image_path):
    """
    Detects objects in an image using YOLOv8 and draws bounding boxes with labels.
    Speaks out a message when an object is detected.

    Args:
        image_path (str): Path to the image file.

    Returns:
        img: Image with bounding boxes and labels drawn.
    """

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Read image
    img = cv2.imread(image_path)

    # Run object detection
    results = model(img)

    # Check if there are any detections
    if len(results) == 0:
        print("No objects detected.")
        return img

    # Draw bounding boxes and labels
    for r in results:
        for b in r.boxes:
            class_id = int(b.cls.item())
            class_name = r.names[class_id]
            x_min, y_min, x_max, y_max = map(int, b.xyxy[0])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Speak out a message when an object is detected
            speak(f"There is a {class_name}")

    return img

# Streamlit app
st.title("Object Detection with Text-to-Speech")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Perform object detection when file is uploaded
    with st.spinner("Performing object detection..."):
        image_path = "temp_image.jpg"  # Save uploaded image to a temporary file
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        detected_image = detect_objects(image_path)

    # Display the detected image
    st.image(detected_image, caption="Detected Objects", use_column_width=True)
