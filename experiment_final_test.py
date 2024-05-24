import streamlit as st
from streamlit import session_state
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load a csv file


def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file to work with:", type='csv')
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

# Function to choose a model
def choose_model():
    model_option = st.selectbox("Choose a model", [
        'KNN', 
        'Logistic Regression', 
        'Decision Tree', 
        'Bagging', 
        'Pasting', 
        'Random Forest', 
        'Adaptive Boosting (AdaBoost)', 
        'Gradient Boosting'
    ])
    
    if model_option == 'KNN':
        n_neighbors = st.number_input("Number of neighbors for KNN (default is 3)", min_value=1, value=3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_option == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_option == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_option == 'Bagging':
        base_estimator = DecisionTreeClassifier()
        model = BaggingClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=50, bootstrap=True)
    elif model_option == 'Pasting':
        base_estimator = DecisionTreeClassifier()
        model = BaggingClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=50, bootstrap=False)
    elif model_option == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_option == 'Adaptive Boosting (AdaBoost)':
        model = AdaBoostClassifier(n_estimators=50)
    elif model_option == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100)
    
    return model

# Function to clean data
def clean_data(data):
    st.write("Cleaning Data...")
    # Drop rows with missing values
    data = data.dropna()
    st.write("Dropped rows with missing values.")
    
    # Encode categorical variables
    if data.select_dtypes(include=['object']).shape[1] > 0:
        data = pd.get_dummies(data, drop_first=True)
        st.write("Encoded categorical variables.")
    else:
        st.write("No categorical variables to encode.")
    
    return data

# Function to display data summary
def data_summary(data):
    st.write("Data Summary")
    st.write(data.describe())
    st.write("Data Shape:", data.shape)

# Function to split and scale data
def split_and_scale_data(data, target_column):
    X = data.drop(columns=[target_column])  # Features (all columns except the target column)
    y = data[target_column]  # Target (the chosen column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.write("MAE", mean_absolute_error(y_pred, y_test))
    st.write("RMSE", mean_squared_error(y_pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test, y_test))

# Function for resampling options
def resample_data(X_train, y_train):
    resampling_method = st.selectbox("Choose a resampling method", ["None", "Oversampling (SMOTE)", "Undersampling", "SMOTEENN", "SMOTETomek"])
    
    if resampling_method == "Oversampling (SMOTE)":
        smote = SMOTE(random_state=42, sampling_strategy=1.0)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif resampling_method == "Undersampling":
        undersample = RandomUnderSampler(random_state=42)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
    elif resampling_method == "SMOTEENN":
        smote_enn = SMOTEENN(random_state=42)
        X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    elif resampling_method == "SMOTETomek":
        smote_tomek = SMOTETomek(random_state=42)
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    
    return X_train, y_train

# Function for Grid Search
def grid_search(model, X_train, y_train, cv):
    default_grid = {
        "n_estimators": [5, 10, 20, 50],
        "max_leaf_nodes": [25, 50, 100, None],
        "max_depth": [5, 15, 25]
    }
    grid = st.text_area("Enter Grid Search Parameters (default used if empty)", value=str(default_grid))
    grid = eval(grid)
    
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search

# Function for Random Search
def random_search(model, X_train, y_train, cv):
    default_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=20, stop=200, num=10)],
        "max_leaf_nodes": [int(x) for x in np.linspace(start=50, stop=300, num=10)],
        "max_depth": [int(x) for x in np.linspace(5, 66, num=11)]
    }
    grid = st.text_area("Enter Random Search Parameters (default used if empty)", value=str(default_grid))
    grid = eval(grid)
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=100, cv=cv, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search

# Streamlit App
st.title('Machine Learning Model Training App')

data = load_data()

if data is not None:
    st.write("Loaded Data:")
    st.write(data)
    
    if st.button("Clean Data"):
        data = clean_data(data)
        st.write("Cleaned Data:")
        st.write(data)

    data_summary(data)

    # Option to select target column, defaulting to the last column
    target_column = st.selectbox("Select the target column", data.columns, index=len(data.columns) - 1)

    # Determine the type of the target column
    target_type = data[target_column].dtype
    if target_type == 'bool':
        st.write("Target column is of type: Boolean")
    elif target_type == 'int64' or target_type == 'float64':
        st.write("Target column is of type: Numerical")
    else:
        st.write("Target column is of type: Categorical")

    X_train, X_test, y_train, y_test = split_and_scale_data(data, target_column)
    model = choose_model()

    if st.button("Train Model"):
        if data is not None:
            X_train, X_test, y_train, y_test = split_and_scale_data(data, target_column)
            model.fit(X_train, y_train)
            st.write(f"Trained {model.__class__.__name__} model.")
            evaluate_model(model, X_test, y_test)
        else:
            st.write("Please load and clean the data before training the model.")