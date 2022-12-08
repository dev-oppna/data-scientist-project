import streamlit as st
import streamlit_authenticator as stauth
from projects.gender_detection.utils import predict, make_gauge as mg, make_barplot
from projects.name_correction.utils import get_name, make_gauge, make_graph
from projects.merchant_categorization.utils import merchant_predict, merchant_clean
import streamlit.components.v1 as components
from statistics import mean


# load_models = st.cache(load_models, allow_output_mutation=True)

st.title('Data Scientist Project')

projects = "Home"

projects = st.sidebar.selectbox("Projects", ["Home", "Name Correction", "Gender Prediction","Merchant Categorization Prediction"])

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
        names_corr, G, conf_level = get_name(names_corr.split(","))
        st.write(f"Nama yang dipilih: {names_corr}")
        fig = make_gauge(conf_level)
        st.plotly_chart(fig, use_container_width=True)
        html = make_graph(G)
        components.html(html, height = 1200,width=1000)

elif projects == "Gender Prediction":
    # model = load_models()

    st.subheader("Gender Detection")
    st.write('''Ini adalah model pendeteksian gender dari nama. Model ini ditraining dari 1 juta data nama gabungan dari dataset 
    nama yang ada di Indonesia maupun di US.''')
    name = st.text_input("Name", key="gender_detection")

    if name:
        result, result_max, gender = predict(name)
        st.write(f"Gender dari {name} adalah: {gender}")
        conf = mean(result_max)
        if conf < 0.5:
            conf = 1 - conf
        fig = mg(int(conf*100), gender)
        st.plotly_chart(fig, use_container_width=True)
        fig_bar = make_barplot(result, name.split())
        st.plotly_chart(fig_bar, use_container_width=True)

elif projects == "Merchant Categorization Prediction":
    # model = load_models()
    st.subheader("Merchant Categorization Prediction")
    st.write('''Ini adalah model pendeteksian merchant categorization. Kamu dapat mendeteksi kategori dari nama merchant''')
    merchant_name = st.text_input("Merchant Name", key="merchant_detection")

    if merchant_name:
        merchant_name = merchant_clean(merchant_name)
        predict_merchant = merchant_predict(merchant_name)
        st.write(f'Category dari Merchant dengan nama {merchant_name} adalah: {predict_merchant}')


