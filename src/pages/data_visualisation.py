import streamlit as st
import matplotlib.pyplot as plt

############################### Constante #####################################

visulasiation_options = ['HISTOGRAM', 'COURBE', 'MODE']


############################### Functions #####################################

def get_columns_for_mode(df, mode):
    if mode in ['HISTOGRAM', 'COURBE']:
        return df.select_dtypes(include='number').columns
    elif mode == 'MODE':
        return df.select_dtypes(include='string').columns
    else:
        return None


def courbe(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    fig, ax = plt.subplots()
    ax.plot(df[column])
    st.pyplot(fig)



def histogram(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=20)
    st.pyplot(fig)


def mode(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    st.write(df[column].mode())

# def boite_a_moustache(df, columns):
#     column = st.selectbox('Sélectionner la colonne à afficher', columns)
#     fig, ax = plt.boxplot(df[column])
#     st.pyplot(fig)


def displayed_figure(df, selectedMode):
    if selectedMode:
        if selectedMode == 'HISTOGRAM':
            columns = get_columns_for_mode(df, selectedMode)
            histogram(df, columns)
        elif selectedMode == 'COURBE':
            columns = get_columns_for_mode(df, selectedMode)
            courbe(df, columns)
        elif selectedMode == 'MODE':
            columns = get_columns_for_mode(df, selectedMode)
            mode(df, columns)
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
