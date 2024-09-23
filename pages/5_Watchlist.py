import streamlit as st
from projects.watchlist.utils import get_watchlist
from projects.utils import add_logo
import pandas as pd
import warnings

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
st.header("Check your user in multiple digital footprint")
st.caption('Disclaimer: This page is for demonstration purposes only, showcasing the Data Input and Output of our product. In a real scenario, our product can be accessed via API calls integrated with your system.')
st.write('''Enter User Information*''')

LIST_LOCATIONS = [
 '*',
 'Aceh',
 'Bandung',
 'Bangka Belitung',
 'Bantul',
 'Bengkulu',
 'Bogor',
 'Cibinong',
 'Cikarang',
 'Cirebon',
 'Gorontalo',
 'Gresik',
 'Jakarta Barat',
 'Jakarta Pusat',
 'Jakarta Selatan',
 'Jakarta Timur',
 'Jakarta Utara',
 'Jambi',
 'Kota Bekasi',
 'Kota Semarang',
 'Kota Surabaya',
 'Kupang',
 'Makassar',
 'Malang',
 'Manado',
 'Medan',
 'Padang',
 'Palangkaraya',
 'Palembang',
 'Pekanbaru',
 'Pematang Siantar',
 'Pontianak',
 'Purwakarta',
 'Sukabumi',
 'Tangerang',
 'Tasikmalaya']

location = st.selectbox("Location", LIST_LOCATIONS, key="location")

with st.form("input_pii", clear_on_submit=False):
    col01, col02 = st.columns(2)

    name = col01.text_input("Name (Mandatory)", key="name", placeholder="Your user name")
    phone = col01.text_input("Phone (Mandatory)", key="phone", placeholder="628XXXXXX/67762xxx")
    email = col01.text_input("Email", key="email", placeholder="test@gmail.com")

    nik = col02.text_input("Citizen ID", key="nik", placeholder="3171xxxx")
    dob = col02.text_input("DOB", key="dob", placeholder="1998-10-10")
    
    submit_data = st.form_submit_button("Search")
    if submit_data and phone and name:
        st.session_state.aggregated_state = True
        with st.spinner(text="Searching your data..."):
            detailed_data = get_watchlist(name=name, phone=phone, nik=nik, email=email, dob=dob, location=location)
        if detailed_data.get("code") == "01":
            data = detailed_data.get('data', {})
            state_court = data.get("state court", [])
            pep = data.get("political party member", {})
            social_media = data.get("social media", [])
            state_officials = data.get("state officials", {})
            kabinet = data.get("kabinet", [])
            
            st.write("State Court")
            if state_court != []:
                state_court = [y for x in state_court for y in x.get("data", [])]
            df_court = pd.DataFrame(state_court)
            st.dataframe(df_court, use_container_width=True)

            st.write("Political Exposed Person")
            pep = [] if pep == {} else [pep]
            state_officials = [] if state_officials == {} else [state_officials]
            df_pep = pd.DataFrame(kabinet+pep+state_officials)
            st.dataframe(df_pep, use_container_width=True)

            st.write("Social Media")
            df_socmed = pd.DataFrame(social_media)
            st.dataframe(df_socmed, use_container_width=True)
        else:
            st.write("Error")
        
    elif submit_data:
        st.warning('Please fill the required field', icon="⚠️")   
st.caption('''How to use this platform:
- **Name**   : Your user name to check the similarity of the name.\n
- **Phone number**            : Enter your user “phone number” as submitted on this platform.\n
- **Citizen ID (Optional)**   : If you choose to provide it, enter a valid NIK.\n
- **DOB (Optional)**   : If you choose to provide it, enter a valid date of birth informationwith format YYYY-MM-DD.\n
- **Email (Optional)**   : If you choose to provide it, enter a valid email.\n
- **Location**   : Choose location from list we provided.\n
- **Click 'Search'** to get information about your user from public data. By clicking this, you confirm that data retrieval has received user consent.
''')