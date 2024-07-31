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
st.markdown("<span style='color:red'>**Demo Version**</span>", unsafe_allow_html=True)
st.header("Enriching users data in seconds with AI")
st.caption('Disclaimer: This page is for demonstration purposes only, showcasing the Data Input and Output of our product. In a real scenario, our product can be accessed via API calls integrated with your system.')
st.write('''Enter User Information*''')
with st.form("input_pii", clear_on_submit=False):
    col01, col02, col03 = st.columns(3)
    name = col01.text_input("Full Name /Name /Nickname", key="name")
    phone = col02.text_input("Phone or Opa Id (Mandatory)", key="phone", placeholder="628XXXXXX/67762xxx")
    nik = col03.text_input("Citizen ID (Optional)", key="nik")
    submit_data = st.form_submit_button("Search")
    if submit_data and name and phone:
        st.session_state.aggregated_state = True
        with st.spinner(text="Searching your data..."):
            detailed_data = get_detailed_retrieve_phone(name=name, phone=phone)
        addresses_list = transform_addresses(detailed_data["addresses"])
        st.subheader("Details data")
        same_person = "NOT KNOWN"
        data_found = "TRUE" if detailed_data.get("opa") else "FALSE"
        opa = detailed_data.get("opa")
        if opa:
            same_person = "TRUE" if opa.get("name_similarity") > 0.7 else "FALSE"
        
        col1, col2 = st.columns(2)
        is_same_person = col1.text_input("Is Name Matched?", key="is_same_person", value=same_person, disabled=True)
        is_data_found = col2.text_input("Is Data Found?", key="is_data_found", value=data_found, disabled=True)

        df_connected_phone = pd.DataFrame(detailed_data["summary"])
        df_emails = pd.DataFrame(detailed_data["email"])
        df_nik = pd.DataFrame(detailed_data["nik"])
        df_pln = pd.DataFrame(detailed_data["pln"])
        address_shared = detailed_data["address"]
        list_addresses = []
        for address in addresses_list:
            address_id = address["address_id"]
            list_by_address_id = [json.loads(x) for x in list(set([json.dumps(x) for x in address_shared if x["address_id"] == address_id and x["opa_id"]!= opa["opa_id"]]))]
            names = ",".join([masking_name(x["name"]) for x in list_by_address_id])
            opas = ",".join([x["opa_id"] for x in list_by_address_id])
            names = names if names != "" else "-"
            opas = opas if opas != "" else "-"
            list_addresses.append({**address, **{"name": names, "opa_id": opas}})

        df_addresses = pd.DataFrame(list_addresses)

        if len(df_connected_phone.columns) > 0:
            df_connected_phone = df_connected_phone.loc[:, ["phone", "name", "connected_by", "id_connected", "opa_id"]]
            df_connected_phone["name"] = df_connected_phone.name.apply(masking_name)
            df_connected_phone.columns = ["Connected Phone", "Connected Name", "Connected By", "Connected ID", "Opa ID"]
            
        if len(df_emails.columns) > 0:
            df_emails = df_emails.loc[df_emails.opa_id != opa["opa_id"]]
            df_emails["name"] = df_emails.name.apply(masking_name)

        if len(df_nik.columns) > 0:
            df_nik = df_nik.loc[df_nik.opa_id != opa["opa_id"]]
            df_nik["name"] = df_nik.name.apply(masking_name)

        if len(df_pln.columns) > 0:
            df_pln = df_pln.loc[df_pln.opa_id != opa["opa_id"]]
            df_pln["name"] = df_pln.name.apply(masking_name)

        if len(df_addresses.columns) > 0:
            df_addresses = df_addresses.loc[:, ["address_id", "address_domicile", "name", "province_domicile", "city_domicile", "district_domicile", "created_date", "updated_date", "opa_id"]]
            df_addresses = df_addresses.sort_values("updated_date", ascending=False)
            df_addresses.columns = ["Connected ID", "Offline Presence", "Connected Name", "Province", "City", "District", "Created", "Updated", "Opa ID"]
            df_addresses["Created"] = df_addresses.Created.apply(transform_date)
            df_addresses["Updated"] = df_addresses.Updated.apply(transform_date)
        
        st.write("Is User has Connections?")
        st.dataframe(df_connected_phone, use_container_width=True)

        st.write("Is User Has Offline Presence ?")
        st.dataframe(df_addresses, use_container_width=True)

        # st.write("NIK Connected Details")
        # st.dataframe(df_nik, use_container_width=True)

        # st.write("Emails Connected Details")
        # st.dataframe(df_emails, use_container_width=True)

        # st.write("PLN Connected Details")
        # st.dataframe(df_pln, use_container_width=True)
        
    elif submit_data:
        st.warning('Please fill the required field', icon="⚠️")   
st.caption('''How to use this platform:
- **Name**                    : Enter your user “Name” as submitted on this platform.\n
- **Phone Number**            : Enter your user “phone number” as submitted on this platform. You can also use your OPA ID for this field.\n
- **Citizen ID (Optional)**   : If you choose to provide it, enter a valid NIK.\n
- **Click 'Search'** to find information about the user submitted. By clicking this, you confirm that data retrieval has received user consent.
''')