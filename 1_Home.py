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
st.header("Enriching users data in seconds with AI - User Verifications")
st.write('''By Phone opa''')
with st.form("input_pii", clear_on_submit=False):
    col01, col02, col03 = st.columns(3)
    name = col01.text_input("Full Name", key="name")
    phone = col02.text_input("Phone or Opa Id (Required)", key="phone", placeholder="628XXXXXX/67762xxx")
    nik = col03.text_input("NIK", key="nik")
    submit_data = st.form_submit_button("Search this data")
    if submit_data and name and phone:
        st.session_state.aggregated_state = True
        with st.spinner(text="Searching your data..."):
            detailed_data = get_detailed_retrieve_phone(name=name, phone=phone)
        addresses_list = transform_addresses(detailed_data["addresses"])
        st.subheader("Details data")
        same_person = "NOT KNOWN"
        data_found = "Available" if detailed_data.get("opa") else "Unavailable"
        opa = detailed_data.get("opa")
        if opa:
            same_person = "TRUE" if opa.get("name_similarity") > 0.7 else "FALSE"
        
        is_same_person = st.text_input("Is same person", key="is_same_person", value=same_person, disabled=True)

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
            df_addresses["created_date"] = df_addresses.created_date.apply(transform_date)
            df_addresses["updated_date"] = df_addresses.updated_date.apply(transform_date)
        
        st.write("Linked Phones Details")
        st.dataframe(df_connected_phone, use_container_width=True)

        st.write("Available Addresses Details (10 latest address used by the user)")
        st.dataframe(df_addresses.head(10), use_container_width=True)

        st.write("NIK Connected Details")
        st.dataframe(df_nik, use_container_width=True)

        st.write("Emails Connected Details")
        st.dataframe(df_emails, use_container_width=True)

        st.write("PLN Connected Details")
        st.dataframe(df_pln, use_container_width=True)
        
    elif submit_data:
        st.warning('Please fill the required field', icon="⚠️")   