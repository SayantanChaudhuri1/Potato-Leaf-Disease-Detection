import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("trained_plant_disease.keras")

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("POTATO LEAF DISEASE DETECTION 🌱")
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

if app_mode == 'Home':
    home_image = Image.open('leaf.jpg') 
    st.image(home_image, caption="Potato Plant (Healthy vs Diseased)", use_container_width=True)

    st.markdown("<h1 style='text-align: center;'>POTATO LEAF DISEASE DETECTION </h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This system uses a machine learning model to detect plant diseases in potato crops. 
    By uploading an image of a potato plant's leaf, the system will predict whether the plant is healthy or affected by a disease.
    
    The goal of this project is to assist farmers in identifying diseases early and taking preventive measures, thereby 
    ensuring better crop yield and promoting sustainable agriculture practices.

    ### How It Works:
    1. Upload an image of a potato plant.
    2. The model will predict whether it’s healthy or affected by a disease.
    3. The result will help you make informed decisions about plant care.
    
    ### Supported Diseases:
    - Early Blight
    - Late Blight
    - Healthy Plant

    ### Let's start by uploading an image in the 'Disease Recognition' page.
    """)

elif app_mode == 'Disease Recognition':
    st.header('POTATO LEAF DISEASE DETECTION ')

    test_image = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])

    if test_image:
        st.image(test_image, width=400, use_container_width=True)

        if st.button('Predict'):
            st.snow() 
            st.write('Prediction in progress...')
            result_index = model_prediction(test_image)  
            class_name = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']
            st.success(f'Model predicts it is: {class_name[result_index]}')  
