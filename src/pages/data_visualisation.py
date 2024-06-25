import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

############################### Constante #####################################

visulasiation_options = ['HISTOGRAM', 'COURBE', 'BOITE_A_MOUSTACHE']


############################### Functions #####################################
def get_columns_for_mode(df, mode):
    """
        Fonction pour obtenir les colonnes à afficher en fonction du mode de visualisation.

        Args:
        - df (pd.DataFrame): Les données du dataset utilisé dans l'application.
        - mode (str): Le mode de visualisation sélectionné par l'utilisateur.

        Returns:
            - list or None: La liste des colonnes à afficher ou None si le mode n'est pas pris en charge.
    """
    if mode in ['HISTOGRAM', 'COURBE', 'BOITE_A_MOUSTACHE']:
        return df.select_dtypes(include='number').columns
    else:
        return None


def courbe(df, columns):
    """
        Fonction pour afficher une courbe en fonction de la colonne sélectionnée dans le dataset.

        Args:
        - df (pd.DataFrame): Les données du dataset utilisé dans l'application.
        - columns (list): La liste des colonnes du dataset pouvant être utilisées pour ce type de visualisation.

        Returns:
            - None
    """
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    chart = alt.Chart(df).mark_line().encode(
        x=column,
        y='count()',
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


def histogram(df, columns):
    """
        Fonction pour afficher un histogramme en fonction de la colonne sélectionnée dans le dataset.

        Args:
        - df (pd.DataFrame): Les données du dataset utilisé dans l'application.
        - columns (list): La liste des colonnes du dataset pouvant être utilisées pour ce type de visualisation.

        Returns:
            - None
    """
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=20)
    st.pyplot(fig)


def boite_a_moustache(df, columns):
    """
        Fonction pour afficher une boîte à moustaches en fonction de la colonne sélectionnée dans le dataset.

        Args:
        - df (pd.DataFrame): Les données du dataset utilisé dans l'application.
        - columns (list): La liste des colonnes du dataset pouvant être utilisées pour ce type de visualisation.

        Returns:
            - None
    """
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    chart = alt.Chart(df).mark_boxplot().encode(
        x=column,
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


def displayed_figure(df, selectedMode):
    """
        Fonction pour afficher la figure en fonction du mode de visualisation sélectionné et de la colonne choisie.

        Args:
        - df (pd.DataFrame): Les données du dataset utilisé dans l'application.
        - selectedMode (str): Le mode de visualisation sélectionné par l'utilisateur.

        Returns:
            - None
    """
    if selectedMode:
        if selectedMode == 'HISTOGRAM':
            columns = get_columns_for_mode(df, selectedMode)
            histogram(df, columns)
        elif selectedMode == 'COURBE':
            columns = get_columns_for_mode(df, selectedMode)
            courbe(df, columns)
        elif selectedMode == 'BOITE_A_MOUSTACHE':
            columns = get_columns_for_mode(df, selectedMode)
            boite_a_moustache(df, columns)
        else:
            st.write('Mode de visualisation non pris en charge')


############################## Using part #####################################

def data_visualization():
    """
        Fonction chargée de la visualisation des données

        Returns:
        - None
    """
    if 'data_clean' in st.session_state:
        df = st.session_state['data_clean']
        visualisation_mode = st.selectbox('Sélectionner le type de visualisation', visulasiation_options)
        displayed_figure(df, visualisation_mode)
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")
