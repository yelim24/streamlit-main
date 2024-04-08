import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title

st.set_page_config(page_title = "yelim", page_icon="🍀", layout = "wide", initial_sidebar_state = "expanded")

st.sidebar.title("웹 타이틀🎨")
st.sidebar.markdown("어쩌구 저쩌구 웹 설명")
st.sidebar.caption("Made by [yelim kim](https://github.com/yelim24/streamlit-main)")

st.sidebar.markdown("---")

st.markdown("# Main page")
st.sidebar.markdown("# Main page")
# st.sidebar.write("어쩌구 저쩌구 페이지 짧은 설명")

# https://github.com/blackary/st_pages?tab=readme-ov-file

