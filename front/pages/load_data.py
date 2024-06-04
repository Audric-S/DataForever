import streamlit as st
import pandas as pd

def load_data_page():
    st.title("Upload your CSV data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        
        # Affiche les trois premières lignes
        st.subheader("Premières 3 lignes:")
        st.dataframe(data.head(3))
        
        # Affiche les trois dernières lignes
        st.subheader("Dernières 3 lignes:")
        st.dataframe(data.tail(3))
    else:
        # Ne fait rien si aucun fichier n'est téléchargé, donc aucune info supplémentaire n'est affichée
        pass