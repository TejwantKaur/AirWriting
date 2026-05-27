# import streamlit as st


# def navbar():
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("Home", use_container_width=True):
#             st.session_state.page = "Home"
#             st.rerun()

#     with col2:
#         if st.button("Kids", use_container_width=True):
#             st.session_state.page = "Kids"
#             st.rerun()

#     with col3:
#         if st.button("TV", use_container_width=True):
#             st.session_state.page = "TV"
#             st.rerun()

import streamlit.components.v1 as components
from utils.load_css import load_css

import streamlit as st

def navbar():
    current_page = st.session_state.page

    if current_page == "Kids":
        css = load_css([
            "assets/kids_styles/navbar.css"
        ])
    else:
        css = load_css([
            "assets/home_styles/navbar.css"
        ])


    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

    
    col1, col2, col3, col4 = st.columns([4,1,1,1])

    with col1:
        st.markdown("""
        <h1 style='color:white; margin-top:-10px;'>
            ✍️ AIGen Write
        </h1>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("Home",
                    use_container_width=True,
                    type="primary" if current_page == "Home" else "secondary"
                    ):
            st.session_state.page = "Home"
            st.rerun()

    with col3:
        if st.button("Kids", 
                    use_container_width=True,
                    type="primary" if current_page == "Kids" else "secondary"
                    ):
            st.session_state.page = "Kids"
            st.rerun()

    with col4:
        if st.button("Smart TV", 
                    use_container_width=True,
                    type="primary" if current_page == "TV" else "secondary"
                    ):
            st.session_state.page = "TV"
            st.rerun()
