import streamlit as st
from components.hero import hero
from components.footer import footer

def render_home():
    # st.title("🏠 Home Page")
    hero()
    footer()
