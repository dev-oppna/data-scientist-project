import streamlit as st
from projects.enrichment.utils import generate_opa_id, get_enrichment, MAPPING
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
st.header("Check your user behaviour")
st.caption('Disclaimer: This page is for demonstration purposes only, showcasing the Data Input and Output of our product. In a real scenario, our product can be accessed via API calls integrated with your system.')
st.write('''Enter User Information*''')

with st.form("input_pii", clear_on_submit=False):
    col1, col2 = st.columns(2)
    phone = col1.text_input("Phone (Mandatory)", key="phone", placeholder="628XXXXXX/67762xxx")
    name = col2.text_input("Name", key="name", placeholder="Your name")
    
    submit_data = st.form_submit_button("Search")
    if submit_data and phone:
        st.session_state.aggregated_state = True
        with st.spinner(text="Searching your data..."):
            detailed_data = get_enrichment(opa_id=generate_opa_id(phone), name=name)
        if detailed_data.get("code") == "01":
            dat_ = detailed_data["data"]
            i = 1
            for key, val in MAPPING.items():
                st.write(key)
                for v in val:
                    st.text_input(v.title(), key=v, value=str(dat_.get(v, "")), disabled=True)
                i += 1
        else:
            st.write("Not Found")
        
    elif submit_data:
        st.warning('Please fill the required field', icon="⚠️")   
st.caption('''How to use this platform:
- **Phone**            : Enter your user “phone number” as submitted on this platform.\n
- **Name**   : Your user name to check the similarity of the name.\n
- **Click 'Search'** to get information about the user data that submitted. By clicking this, you confirm that data retrieval has received user consent.
''')