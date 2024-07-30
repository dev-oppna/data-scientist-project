from pathlib import Path
import streamlit as st
import validators
import base64
import time


def stream_data(resp):
    for word in resp.split(" "):
        yield word + " "
        time.sleep(0.1)

def add_logo(logo_url: str, height: int = 120):
    """Add a logo (from logo_url) on the top of the navigation page of a multipage app.
    Taken from [the Streamlit forum](https://discuss.streamlit.io/t/put-logo-and-title-above-on-top-of-page-navigation-in-sidebar-of-multipage-app/28213/6)
    The url can either be a url to the image, or a local path to the image.

    Args:
        logo_url (str): URL/local path of the logo
    """

    if validators.url(logo_url) is True:
        logo = f"url({logo_url})"
    else:
        logo = f"url(data:image/png;base64,{base64.b64encode(Path(logo_url).read_bytes()).decode()})"

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarHeader"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                background-position: 25px 20px;
            }}
            [data-testid="stSidebarNav"] {{
                padding-top: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )