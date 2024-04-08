import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title

st.set_page_config(page_title = "yelim", page_icon="🍀", layout = "wide", initial_sidebar_state = "expanded")

st.markdown("# Main page")

st.sidebar.title("# Main page")
st.sidebar.markdown("어쩌구 저쩌구 페이지 설명")
st.sidebar.markdown("---")

st.sidebar.caption("Made by [yelim kim](https://github.com/yelim24/streamlit-main)")


