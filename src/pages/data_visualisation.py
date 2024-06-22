import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

############################### Constante #####################################

visulasiation_options = ['HISTOGRAM', 'COURBE', 'BOITE_A_MOUSTACHE']


############################### Functions #####################################
def get_columns_for_mode(df, mode):
    if mode in ['HISTOGRAM', 'COURBE', 'BOITE_A_MOUSTACHE']:
        return df.select_dtypes(include='number').columns
    else:
        return None


def courbe(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    chart = alt.Chart(df).mark_line().encode(
        x=column,
        y='count()',
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

def histogram(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=20)
    st.pyplot(fig)

def boite_a_moustache(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    chart = alt.Chart(df).mark_boxplot().encode(
        x=column,
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


def displayed_figure(df, selectedMode):
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
    if 'data' in st.session_state:
        df = st.session_state['data']

        visualisation_mode = st.selectbox('Sélectionner le type de visualisation', visulasiation_options)

        displayed_figure(df, visualisation_mode)
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")
