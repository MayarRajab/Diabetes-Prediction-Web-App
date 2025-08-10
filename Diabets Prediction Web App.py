# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 19:25:43 2025

@author: click
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open("C:/Users/click/Documents/ProjectML/Deployement/trained_model.sav", 'rb'))


#creating a funcation as prediction

def diabets_prediction(input_data):
    

    # change input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped =input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction) 
    
    if(prediction[0]==0) :
       return('The person is not diabetic')
    else:
      return('The person is  diabetic')

    
def main():
    # getting the title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Naumber of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI= st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age Of The Person')
    
    #code  for predication
    diagnosis = ''
    
    #creating  a button  for  predication
    if st.button('Diabetes Test Result'):
      diagnosis = diabets_prediction([
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
        DiabetesPedigreeFunction, Age
    ])
      st.success(diagnosis)
  
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
