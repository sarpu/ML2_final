import streamlit as st
from tensorflow import keras
from keras.models import load_model
import keras.utils as image
import numpy as np
from pathlib import Path

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

size_ = (10,8)
BASE_PATH = Path('./')





#convert the predicted and actual test values back to their associated labels
def class_convert(classess):
    pred=[]
    for i in classess:
        if i ==0:
            pred.append('Cardboard')
        elif i==1:
            pred.append('Glass')
        elif i==2:
            pred.append('Metal')
        elif i==3:
            pred.append('Paper')
        elif i==4:
            pred.append('Plastic')
        elif i==5:
            pred.append('Trash')
    return pred


def make_prediction(model_name='MobileNetV2'):

    upload = st.session_state.file if st.session_state.file else st.session_state.camera
    model = load_model(BASE_PATH / 'models' / model_name)
    if upload is not None:
        custom_image = image.load_img(upload, target_size=(224, 224))
        img_array = image.img_to_array(custom_image)
        processed_img = keras.applications.mobilenet_v2.preprocess_input(img_array).astype(np.float32)
        swapped = np.moveaxis(processed_img, 0,1) 
        arr4d = np.expand_dims(swapped, 0)
        new_prediction= class_convert(np.argmax(model.predict(arr4d), axis = -1))
        st.write('Your item is: ', new_prediction[0])




def main(model_name='mobilenet_v2'):
    file_upload = st.file_uploader('Trash Image', key='file')
    camera = st.camera_input('Trash Image', key='camera')

    make_prediction()
   
    #model = load_model(BASE_PATH / 'models' / model_name)
    #if upload is not None:
    #    st.image(upload)
    #    custom_image = image.load_img(upload, target_size=(224, 224))
    #    img_array = image.img_to_array(custom_image)
    #    processed_img = keras.applications.mobilenet_v2.preprocess_input(img_array).astype(np.float32)
    #    swapped = np.moveaxis(processed_img, 0,1) 
    #    arr4d = np.expand_dims(swapped, 0)
    #    pred = model.predict(arr4d)
    #    print(pred)
    #    new_prediction= class_convert(np.argmax(pred, axis = -1))
    #    st.write('Your item is: ', new_prediction[0])


main()
