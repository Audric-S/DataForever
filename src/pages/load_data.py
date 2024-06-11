import streamlit as st
import pandas as pd

def load_data_page():
    st.title("Upload your CSV data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            st.success('File successfully uploaded.')
            
            if data.empty:
                st.warning("Le fichier CSV est vide.")
            else:
                na_values = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "-", "--"]
                
                st.subheader("Premières 3 lignes:")
                st.dataframe(data.head(3))
                
                st.subheader("Dernières 3 lignes:")
                st.dataframe(data.tail(3))
                
                num_rows, num_cols = data.shape
                st.subheader("Informations sur le fichier CSV")
                st.write(f"Nombre de lignes: {num_rows}")
                st.write(f"Nombre de colonnes: {num_cols}")
                
                st.subheader("Noms des colonnes:")
                columns_df = pd.DataFrame(data.columns, columns=["Column Names"])
                st.table(columns_df)
                
                missing_values = data.isnull().sum()
                st.subheader("Valeurs manquantes par colonne:")
                st.write(missing_values)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV: {e}")
    else:
        st.info("Veuillez télécharger un fichier CSV.")

if __name__ == "__main__":
    load_data_page()
