import os
import numpy as np
from pathlib import Path
import streamlit as st
from arabic_support import support_arabic_text
from tensorflow.keras.models import load_model
from preprocess import map_proc, remove_diac
from dictionaries import arabic_characters, revers_classes

support_arabic_text(all=True)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "outputs")

model = load_model(os.path.join(MODELS_DIR, "model_Lstm_Embd_20000.h5"))

def predict(arabic_text):
    X, _ = map_proc([arabic_text])

    print(model.predict(X).shape)

    predictions = model.predict(X).squeeze()

    predictions = predictions[1:]

    print(predictions[0])

    diac_arabic_text = ''

    for char, pred in zip(remove_diac(arabic_text), predictions):
        diac_arabic_text += char
        
        if char not in arabic_characters:
            continue
        
        if '<' in revers_classes[np.argmax(pred)]:
            continue

        diac_arabic_text += revers_classes[np.argmax(pred)]

    return diac_arabic_text

st.markdown("<h1 style='text-align: center; color: black;'> 🥳 تطبيق تشكيل النص العربي 🥳</h1>", unsafe_allow_html=True)


input_text = st.text_area(
    label=":rocket:",
    label_visibility='collapsed',
    placeholder="انسخ والصق النص العربي هنا",
    height=150,
)


style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)

if st.button("تشكيل", type="primary"):

    diac_arabic_text = predict(input_text)

    st.write(f'<p style="font-size:30px; text-align: center;">{diac_arabic_text}</p>', unsafe_allow_html=True)

else:
    st.write(f'<p style="font-size:30px; text-align: center;">{input_text}</p>', unsafe_allow_html=True)

