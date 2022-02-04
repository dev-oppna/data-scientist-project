from termios import CINTR
from tensorflow.keras.models import load_model
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


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
            return result, result, "Laki-laki"
        else:
            return result, result, "Perempuan"
    else:
        result_max = [ 1 if logit >= 0.5 else 0 for logit in result ]
        if list(result_max).count(1) >= (panjang_nama//2)+1:
            return result, result_max, "Laki-laki"
        else:
            return result, result_max, "Perempuan"

def make_gauge(value, name):
    fig = go.Figure(go.Indicator(
        mode = "number+gauge",
        gauge = {'shape': "bullet", 'axis': {'range': [None, 100]}},
        value = value,
        domain = {'x': [0.1, 1], 'y': [0, 1]},
        title = {'text': name}))
    fig.update_layout(
        height = 200,
        width = 700,
        margin=dict(
        l=100,
        r=30,
        b=30,
        t=30,
    ),
    )
    return fig

def make_barplot(conf, names):
    genders = ["Laki-laki" if c >= 0.5 else "Perempuan" for c in conf]
    conf = [int(c*100) if c >= 0.5 else int((1-c)*100) for c in conf]
    data = pd.DataFrame(data = zip(names, genders, conf), columns=["Nama", "Gender", "Confidence"])
    data = data.reindex(index=data.index[::-1])
    fig = px.bar(data, x='Confidence', y='Nama', color='Gender', text="Confidence", range_x=[0,100])
    return fig