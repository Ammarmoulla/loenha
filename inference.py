import os
from pathlib import Path
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import map_proc, remove_diac
from dictionaries import arabic_characters, revers_classes

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "outputs")


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


if __name__ == '__main__':
   
   parser = argparse.ArgumentParser(description='Process some URLs.')
   parser.add_argument('--model_path', type=str, help='The URL for type model in inference')
   parser.add_argument('--text', type=str, help='The Arabic  Sentence')

   args = parser.parse_args()
   arabic_text = args.text

   model = load_model(args.model_path)

   diac_arabic_text = predict(arabic_text)

   print(diac_arabic_text)
