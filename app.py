import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    acne_image = image
    image = np.array(image.convert('RGB'))
    acne_image = np.array(acne_image.convert('RGB'))

    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x,y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = image[y:y+h, x:x+w]

    acne_image = cv2.cvtColor(acne_image, cv2.COLOR_BGR2RGB)
    acne_image = cv2.resize(acne_image, (224, 224))
    acne_image = img_to_array(acne_image)
    acne_image = preprocess_input(acne_image)
    acne_image = np.expand_dims(acne_image, axis=0)

    model = load_model("facemodel.model")
    (acne, withoutAcne) = model.predict(acne_image)[0]
    label = "Acne" if acne > withoutAcne else "No Acne"

    return image, faces, label

def about():
    st.write("We developed this application to help people suffering with acne.")

def resources():
    st.write("In order to learn more about acne and computer vision, read this: https://link.springer.com/article/10.1007/s10489-022-03774-z.")

def contact():
    st.write("Contact me at olagunju@ualberta.ca!")

def main():
    st.title("Acne Detection App")
    st.write("**This will be performed using the Haar Cascade Classifiers.**")

    activities = ["Home", "About", "Resources", "Contact"]
    choice = st.sidebar.selectbox("Select a choice from", activities)

    if choice == "Home":

        image_face = st.file_uploader("Upload image", type=['jpeg', 'jpg', 'png', 'webp'])

        if image_face is not None:
            image = Image.open(image_face)
            if st.button("Analyze"):
                result_img, result_faces, result_label = detect(image=image)
                st.image(result_img, use_column_width = True)
                st.success(f"{result_label}")

    elif choice == "About":
        about()
    elif choice == "Resources":
        resources()
    elif choice == "Contact":
        contact()

if __name__ == "__main__":
    main()