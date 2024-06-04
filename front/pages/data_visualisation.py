import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Data Visualisation')

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [0, 10, 20, 10, 0]
})

# Afficher un histogramme pour chaque colonne
for column in df:
    st.subheader(f'Histogramme pour {column}')
    st.bar_chart(df[column])

# Afficher une boîte à moustaches pour chaque colonne
for column in df:
    st.subheader(f'Boîte à moustaches pour {column}')
    fig, ax = plt.subplots()
    ax.boxplot(df[column])
    st.pyplot(fig)
