import streamlit as st
from navbar import navbar

from views.home import render_home
from views.kids import render_kids
from views.tv import render_tv

st.set_page_config(
    page_title="AIGen Write",
    page_icon="✍️",
    layout="wide"
)

# INITIAL PAGE
if "page" not in st.session_state:
    st.session_state.page = "Home"

# NAVBAR
navbar()

# ROUTING
if st.session_state.page == "Home":
    render_home()

elif st.session_state.page == "Kids":
    render_kids()

elif st.session_state.page == "TV":
    render_tv()