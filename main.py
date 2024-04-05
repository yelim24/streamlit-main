import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title

st.set_page_config(page_title = "yelim", page_icon="🍀", layout = "wide", initial_sidebar_state = "expanded")

st.sidebar.title("Sophisticated Palette 🎨")

add_page_title()

show_pages(
    [
        Page("pages/yelim.py", "Main", "🙋‍♀️"),
    ]
)

st.sidebar.markdown("---")
show_pages(
    [
        Page("pages/ontology.py", "Ontology"),
        Page("pages/chatbot.py", "Chatbot"),
        Page("pages/emotion_classification.py"),
        Page("pages/zero_shot_classification.py")
    ]
)
st.sidebar.markdown("---")
