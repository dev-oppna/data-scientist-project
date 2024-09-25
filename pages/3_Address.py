import streamlit as st
from projects.poc.utils import transform_date, transform_addresses, get_address_verification_validation, masking_name, get_formatted
from projects.poc import DICT_CITY, LIST_CITY
from projects.utils import add_logo
import pandas as pd
import warnings
import json

warnings.filterwarnings("ignore")

st.set_page_config(
    layout="wide",
    page_title="Oppna - Your Data Ecosystem"
    )
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

add_logo("assets/logo.png")

if "load_state" not in st.session_state:
    st.session_state.load_state = False
    st.session_state.select_city = False
    st.session_state.aggregated_state = False
    st.session_state.detailed_state = False
    st.session_state.model_trained = False
    st.session_state.set_address = False
    st.session_state.set_waybill = False
    st.session_state.address_url = None
    st.session_state.api_key_here = None
    st.session_state.size = None
    st.session_state.filename = None

data = None
st.markdown("<span style='color:red'>**Demo Version**</span>", unsafe_allow_html=True)
st.header("Verify and validate user's address data in seconds")
st.caption('Disclaimer: This page is for demonstration purposes only, showcasing the Data Input and Output of our product. In a real scenario, our product can be accessed via API calls integrated with your system.')
st.write('''Enter User Information*''')
col01, col02, col03, col04 = st.columns(4)

phone = col01.text_input("Phone (Mandatory)", key="phone", placeholder="628XXXXXX/67762xxx")
address = col02.text_input("Free text address (Mandatory)", key="nik")
city = col03.selectbox("City", LIST_CITY, key="city")
district = col04.selectbox("District", DICT_CITY[city], key="district")
with st.form("input_pii", clear_on_submit=False):
    col11, col12 = st.columns(2)
    # city = col03.text_input("City name (Mandatory)", key="city")
    # district = col04.text_input("District name (Mandatory)", key="district")
    poi = col11.text_input("Point of interest (e.g. Alfamart, atm bca, etc.)", key="poi")
    radius = col12.number_input("POI radius in meter", key="radius", min_value=0, max_value=1000, format='%d')
    submit_data = st.form_submit_button("Search")
    if submit_data and phone and address and city and district:
        st.session_state.aggregated_state = True
        with st.spinner(text="Searching your data..."):
            detailed_data = get_address_verification_validation(phone=phone, address=address, district=district, city=city, poi=poi, radius=radius)
        if detailed_data.get("lon"):
            c1, c2, c3, c4 = st.columns(4)
            address_ = st.text_input("Formatted Address", key="address_", value=f"{get_formatted(detailed_data, detailed_data['district'], detailed_data['city'], detailed_data['province'])}", disabled=True)
            longlat = c1.text_input("Long Lat", key="longlat", value=f"{detailed_data['lon']},{detailed_data['lat']}", disabled=True)
            street = c2.text_input("Street", key="street", value=detailed_data["road_name"], disabled=True)
            rt = c3.text_input("RT", key="rt", value=detailed_data["rt"], disabled=True)
            rw = c4.text_input("RW", key="rw", value=detailed_data["rw"], disabled=True)
            district_ = c1.text_input("District", key="district_", value=detailed_data["district"], disabled=True)
            city_ = c2.text_input("City", key="city_", value=detailed_data["city"], disabled=True)
            province_ = c3.text_input("Province", key="province_", value=detailed_data["province"], disabled=True)
            confidence_level = c4.text_input("Confidence Level", key="confidence_level", value=detailed_data["confidence_level"], disabled=True)

            c1_, c2_ = st.columns(2)
            category = c1_.text_input("Category", key="category", value=detailed_data["category"], disabled=True)
            num_of_poi = c1_.text_input("POI", key="num_of_poi", value=str(detailed_data["num_of_poi"]), disabled=True)
            min_distance_to_poi = c1_.text_input("Min Distance to POI", key="min_distance_to_poi", value=str(detailed_data["min_distance_to_poi"]), disabled=True)

            width = c2_.text_input("Width", key="width", value=detailed_data["width"], disabled=True)
            lanes = c2_.text_input("Lane", key="lanes", value=detailed_data["lanes"], disabled=True)
            surface = c2_.text_input("Surface", key="surface", value=detailed_data["surface"], disabled=True)

            val_remarks = st.text_input("Validation Remark", key="validation_remark", value=detailed_data["validation_remark"], disabled=True)
            val_score = st.text_input("Validation Score", key="validation_score", value=str(detailed_data["validation_score"]), disabled=True)
        else:
            st.write("Not Found")
        
    elif submit_data:
        st.warning('Please fill the required field', icon="⚠️")   
st.caption('''How to use this platform:
- **Phone number**            : Enter your user “phone number” as submitted on this platform.\n
- **Free text address**   : Your user free text address including road name, rt, rw, number, etc.\n
- **District name**   : Your user address district name.\n
- **City name**   : Your user address city name.\n
- **POI**   : Point of Interest that you want to search in area of address.\n
- **Radius**   : Radius POI you want to search.\n
- **Click 'Search'** to get verification and validation about the user data that submitted. By clicking this, you confirm that data retrieval has received user consent.
''')