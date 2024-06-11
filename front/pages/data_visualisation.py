import streamlit as st
import pandas as pd

############################### Constante #####################################

visulasiation_options = ['HISTOGRAM', 'COURBE', 'MODE']

############################### Functions #####################################

def get_columns_for_mode(df, mode):
    if mode in ['HISTOGRAM', 'COURBE']:
        return df.select_dtypes(include='number').columns
    elif mode == 'MODE':
        return df.select_dtypes(include='object').columns
    else:
        return None

def courbe(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    st.line_chart(df[column])

def histogram(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    st.bar_chart(df[column])

def mode(df, columns):
    column = st.selectbox('Sélectionner la colonne à afficher', columns)
    st.write(df[column].mode())

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

st.title('Data Visualisation')

# Lire le fichier CSV
df = pd.read_csv('1000_cryptos.csv')

# Supprimer la première colonne
df = df.drop(columns=df.columns[0])

# Choose the mode of visualisation
visualisation_mode = st.selectbox('Sélectionner le type de visualisation', visulasiation_options)
displayed_figure(df, visualisation_mode)