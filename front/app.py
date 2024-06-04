import streamlit as st
from pages.load_data import load_data_page

def main():
    st.set_page_config(page_title="Streamlit App", page_icon="📊", layout="centered")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Load Data"])

    if page == "Load Data":
        load_data_page()

if __name__ == "__main__":
    main()
