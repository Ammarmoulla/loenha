import os
import numpy as np
from pathlib import Path
import streamlit as st
from tensorflow.keras.models import load_model
from preprocess import map_proc, remove_diac
from dictionaries import arabic_characters, revers_classes


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "outputs")

#print("=============Load Model ================")
model = load_model(os.path.join(MODELS_DIR, "model_Lstm_Embd_20000.h5"))

def predict(arabic_text):
    X, _ = map_proc([arabic_text])
    predictions = model.predict(X).squeeze()
    predictions = predictions[1:]
    
    diac_arabic_text = ''
    for char, pred in zip(remove_diac(arabic_text), predictions):
        diac_arabic_text += char
        
        if char not in arabic_characters:
            continue
        
        if '<' in revers_classes[np.argmax(pred)]:
            continue

        diac_arabic_text += revers_classes[np.argmax(pred)]

    return diac_arabic_text


st.title('تطبيق معالجة النص')

input_text = st.text_area('أدخل النص هنا:')
if st.button('معالجة النص'):
    diac_arabic_text = predict(input_text)
    st.write('النص المعالج:')
    st.write(diac_arabic_text)
