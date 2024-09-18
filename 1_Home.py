import streamlit as st
from projects.poc.utils import transform_date, transform_addresses, get_detailed_retrieve_phone, masking_name
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
st.header("Welcome to Oppna Demo!")
st.write('''Experience how Oppna uses artificial intelligence to verify users and drive valuable business outcomes in Financial Services.
         

No matter what you’re looking to achieve, we’ve got you covered. Simply choose your use case to explore the tailored demo for your needs.''')

data = [
    {"Use Cases": "Responding False Negative on E-KYC",
     "Products": "User Verification (API)"},
     {"Use Cases": "Fraud Investigations",
     "Products": "User Verification (Ad-hoc)"},
     {"Use Cases": "Data Alternative for Underbanked / Unbanked (e.g additional information for Decision Engine)",
     "Products": "User Verification (API)\nAddress Verification (API)\nUser Enrichment (API)"},
     {"Use Cases": "Credit Analysis : Phone Connection Analysis / Location Based Insights",
     "Products": "User Verification (Ad-hoc)"},
     {"Use Cases": "Secured Loans Survey : Verifying Surveyor Report , Online Survey option",
     "Products": "Address Verification (API)"},
     {"Use Cases": "Centralized Public Data Checking (PEP, Court Cases, Sosial Media)",
     "Products": "Watchlist (API)"},
]

markdwn = """Use Cases | Products
:------: | :----------:
Responding False Negative on E-KYC | [User Verification (API)](/User_Verification)
Fraud Investigations | [User Verification (Ad-hoc)](/User_Verification)
Data Alternative for Underbanked / Unbanked (e.g additional information for Decision Engine) | [User Verification (API)](/User_Verification)<br/>[Address Verification (API)](/Address)<br/>[User Enrichment (API)](/Enrichment)
Credit Analysis : Phone Connection Analysis / Location Based Insights | [User Verification (Ad-hoc)](/User_Verification)
Secured Loans Survey : Verifying Surveyor Report , Online Survey option | [Address Verification (API)](/Address)
Centralized Public Data Checking (PEP, Court Cases, Sosial Media) | [Watchlist (API)](/Watchlist)
"""

# df = pd.DataFrame(data)
# st.dataframe(df, use_container_width=True)
st.markdown(markdwn, unsafe_allow_html=True)