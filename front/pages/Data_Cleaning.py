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

if method == "Delete datas":
    delete_options = ("Rows", "Columns", "Both")
    delete_choice = st.selectbox(
        "What would you like to delete ?",
        delete_options,
        index=None,
        placeholder="Select an option"
    )

    threshold = st.slider('Select the threshold percentage:', 0, 100, 50)
    st.write("You selected threshold:", threshold, "%")
    st.write("You selected to delete:", delete_choice)
elif method == "Datas replacement":
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

    # Define num_neighbors outside the if block and set it to None by default
    num_neighbors = None

    if imputation == "KNN":
        # Use st.number_input to get the user's input and update the value of num_neighbors
        num_neighbors = st.number_input("Choose the number of neighbors for KNN imputation:", value=None, placeholder="Type a number...", min_value=1)
        st.write("You selected:", num_neighbors, "neighbors")
    elif imputation == "Regression":
        regression_estimators = ("BayesianRidge", "LinearRegression", "Ridge",  "RandomForestRegressor", "SVR")

        selected_estimator = st.selectbox(
            "Which regression estimator would you like to use ?",
            regression_estimators,
            index=None,
            placeholder="Select an estimator"
        )

        st.write("You selected:", selected_estimator)

submit_button = st.button("Submit your cleaning")

if submit_button:
    if method is None or (method == "Delete datas" and (threshold is None or delete_choice is None)) or (method == "Datas replacement" and replacement is None) or (method == "Sophisticated imputation" and imputation is None) or (imputation == "KNN" and (num_neighbors is None or num_neighbors <= 0)) or (imputation == "Regression" and selected_estimator is None):
        st.error("Please select a valid option in all the select boxes and make sure the number of neighbors is positive")
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

if st.button("Submit your normalization"):
    if method is None or (method == "Delete datas" and (threshold is None or delete_choice is None)) or (method == "Datas replacement" and replacement is None) or (method == "Sophisticated imputation" and imputation is None) or normalizing is None or (imputation == "KNN" and (num_neighbors is None or num_neighbors <= 0)):
        st.error("Please select a valid option in all the select boxes and make sure the number of neighbors is positive")
    else:
        st.success("Datas has been normalized successfully!")
        # Add your data cleaning, normalization, and download code here
