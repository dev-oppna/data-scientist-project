import streamlit as st
import streamlit_authenticator as stauth
from projects.gender_detection.utils import load_models, predict
from projects.name_correction.utils import get_name
from pyvis import network as net
import streamlit.components.v1 as components


load_models = st.cache(load_models, allow_output_mutation=True)

st.title('Data Scientist Project')

projects = "Home"

projects = st.sidebar.selectbox("Projects", ["Home", "Name Correction", "Gender Prediction"])

if projects == "Home":
    st.write('''Ini adalah platform projek untuk data scientist. Sebelum masuk production, ini adalah platform agar POC projek 
    bisa diperiksa secara bersama-sama''')

elif projects == "Name Correction":
    st.subheader("Name Correction")
    st.write('''Ini adalah algoritma pemilihan nama. Jika di data anda seorang user mempunyai banyak nama, dan anda mencoba 
    mencari nama yang paling relevan dari user tersebut. Masukkan nama-nama tersebut dengan dipisahkan tanda koma tanpa spasi 
    (Contoh: Untung,Untung Putra,Untung Putra Riadi).''')
    names_corr = st.text_input("Names", key="name_correction")

    if names_corr:
        names_corr, G = get_name(names_corr.split(","))
        st.write(f"Nama yang dipilih: {names_corr}")
        g4 = net.Network(height='400px', width='500px', notebook=True, heading='names_corr')
        g4.from_nx(G)
        nodes, edges, heading, height, width, options = g4.get_network_data()
        html = g4.template.render(height=height,
                                    width=width,
                                    nodes=nodes,
                                    edges=edges,
                                    heading=heading,
                                    options=options,
                                    physics_enabled=g4.options.physics.enabled,
                                    use_DOT=g4.use_DOT,
                                    dot_lang=g4.dot_lang,
                                    widget=g4.widget,
                                    bgcolor=g4.bgcolor,
                                    conf=g4.conf,
                                    tooltip_link=True)
        components.html(html, height = 1200,width=1000)

elif projects == "Gender Prediction":
    model = load_models()

    st.subheader("Gender Detection")
    st.write('''Ini adalah model pendeteksian gender dari nama. Model ini ditraining dari 15000 data nama gabungan dari dataset 
    nama yang ada di Indonesia maupun di US.''')
    name = st.text_input("Name", key="gender_detection")

    if name:
        result, result_max, gender = predict(model, name)
        st.write(gender)