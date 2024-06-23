import streamlit as st
import pandas as pd

def load_csv_file(uploader_label="Choose a CSV file", file_type="csv"):
    """
    Fonction pour charger un fichier CSV téléchargé via Streamlit.

    Args:
    - uploader_label (str): Libellé de l'uploader dans Streamlit.
    - file_type (str): Type de fichier à accepter (par défaut 'csv').

    Returns:
    - pd.DataFrame or None: Les données du fichier CSV ou None si aucun fichier n'est téléchargé.
    """
    uploaded_file = st.file_uploader(uploader_label, type=file_type)
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV: {e}")
    return None

def display_data_summary(data):
    """
    Fonction pour afficher un résumé des données CSV chargées.

    Args:
    - data (pd.DataFrame): Les données à afficher.

    Cette fonction affiche les premières lignes, les dernières lignes, les informations de base sur le CSV
    et les valeurs manquantes par colonne.
    """
    if data.empty:
        st.warning("Le fichier CSV est vide.")
    else:
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

def load_data_page():
    """
    Fonction principale pour afficher la page de chargement et d'analyse d'un fichier CSV.

    Cette fonction gère l'interface utilisateur en utilisant Streamlit et intègre les autres fonctions définies
    pour charger et afficher les données CSV.
    """
    st.title("Upload your CSV data")
    
    data = load_csv_file()
    
    if data is not None:
        st.session_state['data'] = data
        if 'data-clean' not in st.session_state:
            st.session_state['data-clean'] = data
        st.success('File successfully uploaded.')
        display_data_summary(data)
    else:
        st.info("Veuillez télécharger un fichier CSV.")

if __name__ == "__main__":
    load_data_page()
