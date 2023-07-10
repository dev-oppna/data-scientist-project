import streamlit as st
from projects.gender_detection.utils import predict, make_gauge as mg, make_barplot
from projects.sort_address.utils import sort_waybill, sort_waybill_addrress
from projects.name_correction.utils import get_name, make_gauge, make_graph
from projects.merchant_categorization.utils import merchant_predict, merchant_clean
from projects.address_verification.utils import get_status_address, extract_address, get_score, construct_address
from projects.look_a_like.utils import generate_df, fit_PU_estimator, predict_PU_prob, plot_bar, create_df_lift, lift_reach_plot, \
    get_precision_recall, prec_recall_plot, generate_df_all, download_button, get_figure, generate_opa_id
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit.components.v1 as components
from sqlalchemy import create_engine
import numpy as np
from plotly.subplots import make_subplots
from statistics import mean
import warnings
import pandas as pd
from datetime import datetime as dt
import random
import os
import psutil as p
import difflib



DB_URL = os.getenv("DB_URL")

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
@st.cache(suppress_st_warning=True)
def get_all_df():
    engine = create_engine(f'postgresql://{DB_URL}')
    all_df = pd.read_sql_query('select * from data',con=engine) # This makes the function take 2s to run
    all_df.top_spend_ecommerce_category = all_df.top_spend_ecommerce_category.fillna("No Category")
    return all_df

@st.cache(suppress_st_warning=True)
def download_df(df, name, size):
    name = name.replace(" ", "_").lower()+".csv"
    components.html(
        download_button(df, name, size),
        height=0,
    )

def convert_csv(df, size):
    object_to_download = df.loc[:,["opa_id"]].head(size).to_csv(index=False).encode('utf-8')
    return object_to_download

def convert_csv_general(df):
    object_to_download = df.to_csv(index=False).encode('utf-8')
    return object_to_download

all_df = get_all_df()
if "load_state" not in st.session_state:
    st.session_state.load_state = False
    st.session_state.model_trained = False
    st.session_state.set_address = False
    st.session_state.set_waybill = False
    st.session_state.address_url = None
    st.session_state.api_key_here = None
    st.session_state.size = None
    st.session_state.filename = None

# load_models = st.cache(load_models, allow_output_mutation=True)

st.title('Data Scientist Project')


projects = "Home"

projects = st.sidebar.selectbox("Projects", ["Home", "Name Correction", "Gender Prediction","Merchant Categorization Prediction", "Address Verification", "Sort waybill", "Look a like"])

if projects == "Home":
    st.session_state.load_state = False
    st.write('''Ini adalah platform projek untuk data scientist. Sebelum masuk production, ini adalah platform agar POC projek 
    bisa diperiksa secara bersama-sama''')
    if not st.session_state.set_address:
        with st.form("set_ngrok_url", clear_on_submit=False):
            address_url = st.text_input("Address extraction url")
            api_key_here = st.text_input("API Key")
            if st.form_submit_button("Store this urls"):
                st.session_state.set_address = True
                st.session_state.set_waybill = True
                st.session_state.address_url = address_url
                st.session_state.api_key_here = api_key_here
                st.experimental_rerun()

    else:
        st.write("Address extraction url:",st.session_state.address_url)
        st.write("API Key:",st.session_state.api_key_here)
        if st.button('Reset urls'):
            st.session_state.set_address = False
            st.experimental_rerun()


elif projects == "Name Correction":
    st.session_state.load_state = False
    st.header("Name Correction")
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
    st.session_state.load_state = False

    st.header("Gender Detection")
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

elif projects == "Address Verification":
    st.session_state.load_state = False
    st.header("Address Verification")
    st.write('''Ini adalah model untuk memverifikasi alamat free text, verifikasi ini bukan hanya melihat di mana alamat berada, tapi juga menskor kelengkapan alamat. Masukkan alamat dengan format alamat, district, city.''')
    if st.session_state.set_address:
        st.write("Your address extraction url:",st.session_state.address_url)
        st.write("Your address API Key:",st.session_state.api_key_here)
        with st.form("address_verification_form", clear_on_submit=False):
            address = st.text_input("Address", key="address_verification")
            col1, col2 = st.columns(2)
            poi = col1.text_input("What POI you want to search?", key="poi")
            radius = col2.number_input("Radius you wan to search (in meters)", key="radius", min_value=10, max_value=1000, format='%d')
            submit_address = st.form_submit_button("Score this address")
            if submit_address:
                address = address.split(",")
                district = address[-2]
                city = address[-1]
                street = " ".join(address[:-2])
                address = " ".join(address)
                data = extract_address(st.session_state.address_url, street)
                score = get_score(data)
                data['district'] = district
                data['city'] = city
                address = construct_address(data, district, city)
                address, lat, lon, num_of_alfa, categories, max_lanes, max_width, is_motorcycle, surface, confidence_score, min_distance_to_poi, max_distance_to_poi =  get_status_address(st.session_state.api_key_here, address, poi, radius, data)
                label = address["label"]
                confidence_score = round(confidence_score)
                label_extracted = extract_address(st.session_state.address_url, label.lower())
                data["road_address"] = data["road_address"] if difflib.SequenceMatcher(None, ''.join(filter(str.isalpha, data["road_address"])).lower(), ''.join(filter(str.isalpha, label_extracted["road_address"])).lower()).quick_ratio() < 0.6 or label_extracted["road_address"] == "" else label_extracted["road_address"]
                data["rt_address"] = data["rt_address"] if data["rt_address"] != "" else label_extracted["rt_address"]
                data["rw_address"] = data["rw_address"] if data["rw_address"] != "" else label_extracted["rw_address"]
                data["block_address"] = data["block_address"] if data["block_address"] != "" else label_extracted["block_address"]
                data['postal_code'] = address["postalCode"]
                data['lat'] = lat
                data['lon'] = lon
                data['score'] = score
                data['confidence'] = confidence_score
                df = pd.DataFrame([data])
                label = (construct_address(data, district, city) + f" {address['postalCode']} {address['county']} {address['countryName']}").lower()
                st.dataframe(df)
                st.write(f'''
                label: {label}\n
                confidence_score: {confidence_score}\n
                score: {score}\n
                lat: {lat}\n
                lon: {lon}\n
                num_of_{poi.replace(" ", "_")}_in_{radius}_m: {num_of_alfa}\n
                min_distance_to_{poi.replace(" ", "_")}: {min_distance_to_poi} meters\n
                max_distance_to_{poi.replace(" ", "_")}: {max_distance_to_poi} meters\n
                categories: {categories}\n
                lanes: {max_lanes}\n
                width: {max_width}\n
                is_motorcycle: {is_motorcycle}\n
                surface: {surface}''')
                if lat != "error":
                    st.map(df)
    else:
        st.warning('You need to set address url', icon="⚠️")

elif projects == "Sort waybill":
    st.session_state.load_state = False
    st.header("Sort waybill")
    st.write('''Ini adalah model untuk mensorting alamat berdasarkan alamat terdekat dengan TH.''')
    num_of_waybill = st.number_input("Number of waybill", key="num_of_waybull", min_value=1, max_value=100, format='%d')
    my_dict = {}
    if st.session_state.set_waybill and num_of_waybill:
        st.write("Your address API Key:",st.session_state.api_key_here)
        with st.form("sort_waybill_form", clear_on_submit=False):
            th = st.text_input("Alamat TH", key="address_th")
            col1, col2 = st.columns(2)
            latitude = col1.number_input("latitude", key="lat_th", format='%.9f')
            longitude = col2.number_input("longitude", key="lon_th", format='%.9f')

            for i in range(num_of_waybill):
                my_dict[i] = {}
                my_dict[i]["waybill"] = st.text_input(f"Waybill {i+1}", key=f"sort_waybill_{i}")
                my_dict[i]["recipient_address"] = st.text_input(f"Address {i+1}", key=f"sort_address_{i}")
                # col11, col21 = st.columns(2)
                # my_dict[i]["latitude"] = col11.number_input(f"latitude {i+1}", key=f"lat_{i}", format='%.9f')
                # my_dict[i]["longitude"]= col21.number_input(f"longitude {i+1}", key=f"lon_{i}", format='%.9f')
            submit_address = st.form_submit_button("Sort this address")
            if submit_address:
                th_data = [{"name": th, "recipient_address": latitude, "longitude": longitude}]
                # data = th_data + [x for w, x in my_dict.items()]
                data = [x for w, x in my_dict.items()]
                nodes = pd.DataFrame(data)
                # routes, distance, fig = sort_waybill(nodes)
                routes = sort_waybill_addrress(nodes, st.session_state.address_url)
                st.dataframe(routes)
                # st.write(f"Total distance: {distance}")
                # st.pyplot(fig)
        if submit_address:
            st.download_button(
                label="Download data as CSV",
                data= convert_csv_general(routes),
                file_name=f'routes.csv',
                mime='text/csv',
            )

    else:
        st.warning('You need to set api key and url', icon="⚠️")

elif projects == "Look a like":
    st.header("Look a like")
    st.write('''Ini adalah contoh aplikasi modeling untuk look a like segment.''')
    seed_file = st.file_uploader(label="Enter your seed", type=['csv'], key="files_seed")
    if seed_file is not None:
        df = pd.read_csv(seed_file)
        st.dataframe(df.head())
        if "phone_number" not in df.columns and "opa_id" not in df.columns:
            st.warning('You need to provide phone_number column in your data', icon="⚠️")
        else:
            if "phone_number" in df.columns:
                df['phone_number'] = df['phone_number'].astype('str')
                df['opa_id'] = df.phone_number.apply(generate_opa_id)
            if st.button('Look a like!') or st.session_state.load_state:
                st.session_state.load_state = True
                st.write("Checking...")
                # Ubah phone_number to opa_id

                # Cek berapa yang cross
                df_crossed = all_df.loc[all_df.opa_id.isin(df.opa_id)]
                df_crossed['label'] = 1
                len_all = len(all_df)
                len_df = len(df)
                len_crossed = len(df_crossed)
                perc_crossed = len_crossed/len_df
                st.write(f"Your data crossed {len_crossed}, or {perc_crossed*100:.2f}%")
                if perc_crossed > 0.2 or len_crossed > 1000:
                    df_bar = (df_crossed.loc[df_crossed.top_spend_ecommerce_category != "No Category", ['top_spend_ecommerce_category']].value_counts(normalize=True).rename_axis('Top Spend') * 100).round(2).reset_index(name='counts')[:10]
                    fig = plot_bar('Top Spend', df_bar)
                    st.plotly_chart(fig)
                    if not st.session_state.model_trained:
                        with st.spinner(text="Training your seed segment..."):
                            start = dt.now()
                            df_noncrossed = all_df.loc[~all_df.opa_id.isin(df.opa_id)]
                            df_noncrossed_sampled = df_noncrossed.sample(len_crossed)
                            df_noncrossed_sampled['label'] = 0
                            df_train = pd.concat([df_crossed, df_noncrossed_sampled], ignore_index=True)
                            df_train, attributes = generate_df(df_train)
                            label = [0,1,1,1]
                            df_train["label_adj"] = [x if x == 0 else random.choice(label) for x in df_train.label]
                            x_data = df_train.loc[:,attributes].values
                            y_labeled = df_train.loc[:,"label_adj"].values
                            y_positive = df_train.loc[:,"label"].values # original class
                            x_train, x_test, y_train, y_test = train_test_split(x_data, y_labeled, test_size=0.2, random_state=7, stratify=y_labeled)
                            list_probs = []
                            predicted = np.zeros(len(x_data))
                            learning_iterations = 4
                            for index in range(learning_iterations):
                                pu_estimator, probs1y1 = fit_PU_estimator(x_data, y_labeled, 0.2, KNeighborsClassifier(metric='cosine', 
                                                                                                                    n_neighbors=5, n_jobs=-1))
                                predicted += predict_PU_prob(x_data, pu_estimator, probs1y1)
                                list_probs.append(probs1y1)
                            threshold = np.mean(list_probs)
                            y_predict = [1 if x > 0.5 else 0 for x in (predicted/learning_iterations)]
                            y_probs = predicted/learning_iterations
                            max_ = max(y_probs)
                            range_ = max_ - 0.5
                            y_probs = [round(x, 2) if x < 0.5 else round(((x-0.5)/range_*0.5)+0.5, 2) for x in y_probs]
                            acc_score = round(accuracy_score(y_positive, y_predict)*100, 2)
                            rec_score = round(recall_score(y_positive, y_predict, average='binary')*100, 2)
                            precision_score = round(precision_score(y_positive, y_predict, average='binary')*100, 2)
                            minutes = round(((dt.now() - start).total_seconds()/60), 2)
                            precision, recall = get_precision_recall(y_positive, y_probs)
                            precision.reverse()
                            recall.reverse()
                        with st.spinner(text="Provisioning your pool..."):
                            x_data = generate_df_all(all_df).loc[:,attributes].values
                            y_probs = [x for x in predict_PU_prob(x_data, pu_estimator, threshold)]
                            max_ = max(y_probs)
                            range_ = max_ - 0.5
                            y_probs_adj = [round(x, 2) if x < 0.5 else round(((x-0.5)/range_*0.5)+0.5, 2) for x in y_probs]
                            all_opa = all_df.loc[:,['opa_id']]
                            all_opa['score'] = y_probs_adj
                            all_opa.sort_values(by="score", ascending=False, inplace=True)
                            df_lift = create_df_lift(all_df.label, y_probs_adj)
                        st.balloons()
                        st.success(f'Congratulations model already trained!')
                        st.session_state.model_trained = True
                        st.session_state.metadata = {"acc": acc_score, "rec": rec_score, "precision": precision_score, "minutes": minutes, "precision_list": precision, "recall_list": recall}
                        st.session_state.df = df_train
                        st.session_state.df_lift = df_lift
                        st.session_state.all_opa = all_opa
                    
                    st.info(f'Model already trained!')


                    precision = st.session_state.metadata.get("precision_list")
                    recall = st.session_state.metadata.get("recall_list")
                    fig__ = prec_recall_plot(precision, recall)
                    fig_subs = make_subplots(rows=1, cols=2, subplot_titles=('Precision-Recall',  'Uplift-Reach'))
                    df_lift = st.session_state.df_lift
                    fig_ = lift_reach_plot(df_lift)
                    fig_subs = get_figure(fig_subs, fig__, fig_)

                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    acc = st.session_state.metadata.get("acc")
                    rec = st.session_state.metadata.get("rec")
                    precision = st.session_state.metadata.get("precision")
                    minutes = st.session_state.metadata.get("minutes")
                    col1.metric("Accuracy", f"{acc}%")
                    col2.metric("Precision", f"{rec}%")
                    col3.metric("Recall", f"{precision}%")
                    col4.metric("Time", f"{minutes}m")
                    st.plotly_chart(fig_subs)
                    option = st.radio("Seed segment option", ('Include', 'Exclude', 'Optimize'), key="option_lal", help = '''**Include**: Combination of your seed and our pool
                    \n**Exclude**: The data is only from our pool excluding your seed
                    \n**Optimize**: All of our data and your crossed seed data''')
                    with st.form("look_a_like_form", clear_on_submit=False):
                        all_opa = st.session_state.all_opa
                        segment_name = st.text_input("Segment Name", key="segment_name")
                        max_size = int(len_all/2)
                        if st.session_state.option_lal == "Include":
                            all_opa = all_opa.loc[~all_opa.opa_id.isin(df.opa_id), ["opa_id"]].sample(max_size)
                            all_opa = pd.concat([all_opa, df.loc[:,["opa_id"]]], ignore_index=True).drop_duplicates('opa_id')
                            max_size = len(all_opa)
                        elif st.session_state.option_lal == "Exclude":
                            all_opa = all_opa.loc[~all_opa.opa_id.isin(df.opa_id), ["opa_id"]]
                            max_size = int(len(all_opa)/2)
                            all_opa = all_opa.sample(max_size)
                        else:
                            all_opa = all_opa.loc[:, ["opa_id"]].sample(max_size)
                        size = st.slider('Segment size', 1000, max_size)
                        submit = st.form_submit_button("Create Segment")
                        if submit:
                            # download_df(all_opa, segment_name, size)
                            st.session_state.size = size
                            st.session_state.filename = segment_name
                    if submit:
                        st.download_button(
                            label="Download data as CSV",
                            data= convert_csv(all_opa, st.session_state.size),
                            file_name=f'{st.session_state.filename}.csv',
                            mime='text/csv',
                        )
                else:
                    st.warning('Your data is not sufficient for build look a like model! Input another seed!', icon="⚠️")
    else:
        st.session_state.load_state = False
        st.session_state.model_trained = False
                    

elif projects == "Merchant Categorization Prediction":
    st.session_state.load_state = False
    st.header("Merchant Categorization Prediction")
    st.write('''Ini adalah model pendeteksian merchant categorization. Kamu dapat mendeteksi kategori dari nama merchant yang kamu cari''')
    merchant_name = st.text_input("Merchant Name", key="merchant_detection")

    if merchant_name:
        merchant_name = merchant_clean(merchant_name)
        predict_merchant_cat,merchant_cat_image = merchant_predict(merchant_name)
        st.write(f'Category dari Merchant dengan nama {merchant_name} adalah: {predict_merchant_cat}')
        st.image(merchant_cat_image)


