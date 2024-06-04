import streamlit as st

st.title('Welcome on the data cleaning page')

st.subheader('Data cleaning', divider='grey')

method = st.selectbox(
    "How would you like to deal with missing values ?",
    ("Delete datas", "Datas replacement", "Sophisticated imputation"),
    index=None,
    placeholder="Select a method"
)

st.write("You selected:", method)

if method == "Datas replacement":
    replacement_options = ("Mean", "Median", "Mode")

    replacement = st.selectbox(
        "How would you like to replace the missing values ?",
        replacement_options,
        index=None,
        placeholder="Select a method"
    )

    st.write("You selected:", replacement)
elif method == "Sophisticated imputation":
    imputation_options = ("KNN", "Regression")

    imputation = st.selectbox(
        "How would you like to impute the missing values ?",
        imputation_options,
        index=None,
        placeholder="Select a method"
    )

    st.write("You selected:", imputation)

submit_button = st.button("Submit")

if submit_button:
    if method is None or (method == "Datas replacement" and replacement is None) or (method == "Sophisticated imputation" and imputation is None):
        st.error("Please select a valid option in all the select boxes")
    else:
        st.success("Datas has been cleaned successfully!")

st.subheader('Data normalizing', divider='grey')


normalizing = st.selectbox(
    "How would you like to normalize your datas ?",
    ("Min Max", "Z-score", "Other methods"),
    index=None,
    placeholder="Select a method"
)

st.write("You selected:", normalizing)

if st.button("Submit"):
    if method is None or (method == "Datas replacement" and replacement is None) or (method == "Sophisticated imputation" and imputation is None) or normalizing is None:
        st.error("Please select a valid option in all the select boxes")
    else:
        st.success("Submitted successfully!")
        # Add your data cleaning code here