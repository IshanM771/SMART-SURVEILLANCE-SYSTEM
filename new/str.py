import streamlit as st
import cv2
import numpy as np
import video # replace with your actual module

def recognize_faces_in_video(video_capture):
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Quit when the input video file ends
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = your_face_recognition_module.face_locations(rgb_small_frame)
            face_encodings = your_face_recognition_module.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = your_face_recognition_module.compare_faces(face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    name = "Known"

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        st.image(frame)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

st.title('Face Recognition App')

option = st.selectbox('Choose an option', ('Upload video file', 'Open camera'))

if option == 'Upload video file':
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        video_capture = cv2.VideoCapture(uploaded_file.name)
        recognize_faces_in_video(video_capture)
elif option == 'Open camera':
    video_capture = cv2.VideoCapture(0)
    recognize_faces_in_video(video_capture)