import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import streamlit as st 
import pickle 

model=tf.keras.models.load_model('model.h5')
with open('LabelEncoder_gender.pkl','rb') as file:
    LabelEncoder_gender=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
with open('ohe_geo.pkl','rb') as file:
    ohe_geo=pickle.load(file)

st.title("Customer Estimated salary Prediction ")
geography=st.selectbox('Geography',ohe_geo.categories_[0])
gender= st.selectbox('Gender',LabelEncoder_gender.classes_)
age=st.slider('Age',18,99)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_credit_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender':[LabelEncoder_gender.transform([gender])[0]],
        'Age':[age],
        
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_credit_card],
        'IsActiveMember':[is_active_member],
        'Exited':[exited]
    }
)
# Transform the new data (must be in a 2D format like [['France']])
geo_encoded = ohe_geo.transform([[geography]]).toarray()

# DataFrame with the feature names
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)#concatinating the geographical ohe data to imput data

scaled_input=scaler.transform(input_data)

#prediction time it is hehehehe
pred=model.predict(scaled_input)
original_value=pred[0][0]

st.write(f'Expected salary would be:{original_value*100090.239881:.2f}')
