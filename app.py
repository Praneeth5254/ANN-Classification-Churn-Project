import pandas as pd
import numpy as np
import os
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


#load the trained model
model = tf.keras.models.load_model('model.h5')


##load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
          label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
        one_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)


#streamlit app
st.title("Customer churn prediction")

#user input
geography  = st.selectbox('Geography', one_encoder_geo.categories_[0])
gender  = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit card', [0,1])
is_active_member = st.selectbox('Is Active member', [0,1])


#prepare the data

#Example usage

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography' : [geography],
    'Gender' : [gender],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

#one hot encode 'geography'
geo_encoded = one_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= one_encoder_geo.get_feature_names_out(['Geography']))


#combine one hot encoded data
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis= 1)

# drop original geography column to match training features
input_data.drop('Geography', axis=1, inplace=True)

# encode Gender using LabelEncoder
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

#scale the input data
input_scaled_data = scaler.transform(input_data)

#predicting the churn
prediction = model.predict(input_scaled_data)
prediction_proba = prediction[0][0]


st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
        st.write("This customer is likely to churn")
else:
        st.write("This customer is not likely to churn")