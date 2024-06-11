import streamlit as st

def main():
    st.title('Welcome on the data cleaning page')
    st.subheader('Data cleaning', divider='grey')
    
    method = select_method()
    handle_method_selection(method)
    
    st.subheader('Data normalizing', divider='grey')
    normalizing = select_normalizing_method()
    
    st.write("You selected:", normalizing)
    
    if st.button("Submit your normalization"):
        if validate_inputs(method, normalizing):
            st.success("Datas has been normalized successfully!")
            # Add your data cleaning, normalization, and download code here

def select_method():
    method = st.selectbox(
        "How would you like to deal with missing values?",
        ("Delete datas", "Datas replacement", "Sophisticated imputation"),
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", method)
    return method

def handle_method_selection(method):
    if method == "Delete datas":
        handle_delete_datas()
    elif method == "Datas replacement":
        handle_datas_replacement()
    elif method == "Sophisticated imputation":
        handle_sophisticated_imputation()

def handle_delete_datas():
    delete_options = ("Rows", "Columns", "Both")
    delete_choice = st.selectbox(
        "What would you like to delete?",
        delete_options,
        index=None,
        placeholder="Select an option"
    )
    
    threshold = st.slider('Select the threshold percentage:', 0, 100, 50)
    st.write("You selected threshold:", threshold, "%")
    st.write("You selected to delete:", delete_choice)
    
def handle_datas_replacement():
    replacement_options = ("Mean", "Median", "Mode")
    replacement = st.selectbox(
        "How would you like to replace the missing values?",
        replacement_options,
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", replacement)
    
def handle_sophisticated_imputation():
    imputation_options = ("KNN", "Regression")
    imputation = st.selectbox(
        "How would you like to impute the missing values?",
        imputation_options,
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", imputation)
    
    if imputation == "KNN":
        num_neighbors = st.number_input("Choose the number of neighbors for KNN imputation:", value=None, placeholder="Type a number...", min_value=1)
        st.write("You selected:", num_neighbors, "neighbors")
    elif imputation == "Regression":
        regression_estimators = ("BayesianRidge", "LinearRegression", "Ridge", "RandomForestRegressor", "SVR")
        selected_estimator = st.selectbox(
            "Which regression estimator would you like to use?",
            regression_estimators,
            index=None,
            placeholder="Select an estimator"
        )
        st.write("You selected:", selected_estimator)

def select_normalizing_method():
    normalizing = st.selectbox(
        "How would you like to normalize your datas?",
        ("Min Max", "Z-score", "Other methods"),
        index=None,
        placeholder="Select a method"
    )
    return normalizing

def validate_inputs(method, normalizing):
    if method is None or \
       (method == "Delete datas" and (threshold is None or delete_choice is None)) or \
       (method == "Datas replacement" and replacement is None) or \
       (method == "Sophisticated imputation" and imputation is None) or \
       normalizing is None or \
       (imputation == "KNN" and (num_neighbors is None or num_neighbors <= 0)):
        st.error("Please select a valid option in all the select boxes and make sure the number of neighbors is positive")
        return False
    else:
        return True

def data_cleaning():
    if 'data' in st.session_state:
        df = st.session_state['data']
        main()
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")
