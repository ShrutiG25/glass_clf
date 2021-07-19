import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.ensemble import RandomForestClassifier
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()
def prediction(model, Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe):
    glass_type = model.predict([[Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

st.title('Glass-Type Classification')
st.sidebar.title('Toolbar')

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass-Type Raw Dataset")
    st.dataframe(glass_df)

import seaborn as sns
import matplotlib.pyplot as plt
multi = st.sidebar.multiselect("Select the Charts you want", ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))

if "Line Chart" in multi:
    st.subheader("Line Chart")
    st.line_chart(glass_df)

if "Area Chart" in multi:
    st.subheader("Area Chart")
    st.area_chart(glass_df)

#avoid warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation Heatmap'in multi:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8,4))
    sns.heatmap(glass_df.corr(), annot = True)
    st.pyplot()

if 'Count Plot' in multi:
    st.subheader("Count Plot")
    plt.figure(figsize=(8,4))
    sns.countplot(glass_df["GlassType"])
    st.pyplot()

if "Pie Chart" in multi:
    st.subheader("Pie Chart")
    count = glass_df["GlassType"].value_counts()
    plt.figure(figsize=(8,4))
    plt.pie(count, labels = count.index, autopct = "%1.2f%%")
    st.pyplot()

if "Box Plot" in multi:
    st.subheader("Box Plot")
    sel = st.sidebar.selectbox(label = "Choose the column", options = ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize=(8,4))
    sns.boxplot(glass_df[sel])
    st.pyplot()

st.sidebar.subheader("Choose the values")
Ri = st.sidebar.slider("Ri", float(glass_df["RI"].min()), float(glass_df["RI"].max()))
Na = st.sidebar.slider("Na", float(glass_df["Na"].min()), float(glass_df["Na"].max()))
Mg =  st.sidebar.slider("Mg", float(glass_df["Mg"].min()), float(glass_df["Mg"].max()))
Al =  st.sidebar.slider("Al", float(glass_df["Al"].min()), float(glass_df["Al"].max()))
Si =  st.sidebar.slider("Si", float(glass_df["Si"].min()), float(glass_df["Si"].max()))
K =  st.sidebar.slider("K", float(glass_df["K"].min()), float(glass_df["K"].max()))
Ca =  st.sidebar.slider("Ca", float(glass_df["Ca"].min()), float(glass_df["Ca"].max()))
Ba =  st.sidebar.slider("Ba", float(glass_df["Ba"].min()), float(glass_df["Ba"].max()))
Fe =  st.sidebar.slider("Fe", float(glass_df["Fe"].min()), float(glass_df["Fe"].max()))

st.sidebar.subheader("Choose a Classifier")
classify = st.sidebar.selectbox("Classificier", options = ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier"))

from sklearn.metrics import plot_confusion_matrix
if classify == "Support Vector Machine":
    st.sidebar.subheader("Select the Hyperparamater")
    c_val = st.sidebar.number_input("C (Error rate", 1,100, step = 1)
    gamma_val = st.sidebar.number_input("Gamma", 1,100, step = 1)
    kernel_val = st.sidebar.radio("Kernel", options = ("linear", "rbf", "poly"))

    if st.sidebar.button("Classify"):
        st.subheader("Support Vector Classification")
        svc_m = SVC(C = c_val, gamma = gamma_val, kernel = kernel_val)
        svc_m.fit(X_train, y_train)
        x_test_score = svc_m.score(X_test, y_test)

        pred = svc_m.predict(X_test)
        glass_type = prediction(svc_m,Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        st.write("The Predicted Glass-Type is : ", glass_type)
        st.write("The accuracy score is :", x_test_score)
        plot_confusion_matrix(svc_m,X_test, y_test)
        st.pyplot()
if classify =="Random Forest Classifier":
    st.sidebar.subheader("Select the Hyperparamater")
    n_est = st.sidebar.number_input("N_Estimators", 100,5000, step = 10)
    max_depth1= st.sidebar.number_input("Maximum Depth", 1,100, step = 2)

    if st.sidebar.button("Classify"):
        st.sidebar.subheader("Random Forest Classifier")
        rfc1 = RandomForestClassifier(n_estimators=n_est, max_depth = max_depth1, n_jobs = -1)
        rfc1.fit(X_train, y_train)
        rfc_score_val = rfc1.score(X_test, y_test)

        pred1 = rfc1.predict(X_test)
        glass_type1 = prediction(rfc1, Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        st.write("The Predicted Glass-Type is : ", glass_type1)
        st.write("The accuracy score is :", rfc_score_val)
        plot_confusion_matrix(rfc1,X_test, y_test)
        st.pyplot()






