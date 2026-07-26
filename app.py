
import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np


model = keras.models.load_model(
    "histo_image_classifier_builtmodel.keras",
    compile=False
)


class_names = [
    "AMELOBLASTOMA",
    "AOT"
]


st.title("Oral Pathology AI Classifier")

st.write(
    "Upload a histopathology image for classification."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image"
    )


    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    prediction = model.predict(img_array)


    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction)


    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )
