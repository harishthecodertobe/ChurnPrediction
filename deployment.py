import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('my_model.h5')

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
with open('onehot_encoder_geo.pkl','rb') as f:
    onehot_encoder_geo=pickle.load(f)
##Streamlit App
st.title("Customer Churn Prediction")
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.slider('Balance',0,2500000)
credit_score=st.slider('Credit Score')
estimated_salary=st.slider('Estimated Salary',0,200000)
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

##Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

input_scaled=scaler.transform(input_data)
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]

if st.button("Predict Churn"):
    if prediction_proba>0.5:
        st.error(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1-prediction_proba:.2f}")