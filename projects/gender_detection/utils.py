from tensorflow.keras.models import load_model
import numpy as np


def load_models():
    model = load_model('model_gender.h5')
    return model

def preprocess_pred(nama):
    name_length = 50
    nama = nama.lower()
    nama = nama.split(" ")
    nama = [(name + ' '*name_length)[:name_length] for name in nama]
    # Step 2: Encode Characters to Numbers
    nama = [[
            max(0.0, ord(char)-96.0) 
            for char in name] for name in nama]
    return nama

def predict(model, nama):

    pre_name = preprocess_pred(nama)
    # Predictions
    result = model.predict(np.asarray(
    pre_name)).squeeze(axis=1)
    panjang_nama = len(result)
    if panjang_nama%2==0 or panjang_nama==1:
        prob = sum(result)/len(result)
        if prob > 0.5:
            return result, result, "Boy"
        else:
            return result, result, "Girl"
    else:
        result_max = [ 1 if logit >= 0.5 else 0 for logit in result ]
        if list(result_max).count(1) >= (panjang_nama//2)+1:
            return result, result_max, "Boy"
        else:
            return result, result_max, "Girl"