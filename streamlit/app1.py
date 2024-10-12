import streamlit as st
import pandas as pd
import joblib
#Loading model
try:
    model=joblib.load('xgb_model.pkl')
    st.write("Model Loaded Successfully!")
except Exception as e:
    st.write(f"Error occured in loading model {e}")
    st.stop()
#Defining category mapping of income
income_prediction={
    1:'Income >50K',
    0:'Income <=50K'
}
country_mapping = {
    'United-States': 0,
    'Canada': 1,
    'India': 2,
    'Mexico': 3,
    # Add other countries as needed
}
#Main streamlit app definiton

def main():
    st.title("Income Category Predictor")
    age=st.number_input('age',min_value=0,max_value=100)
    fnlwgt=st.number_input('Final Weight',min_value=0.0,max_value=430000.0,value=185000.0)
    education_num=st.number_input('Education Qualification(In Number)',min_value=0,max_value=17)
    capital_gain=st.number_input('Capital Gain($)',min_value=0,max_value=100000,value=1090)
    capital_loss=st.number_input('Capital Loss($)',min_value=0,max_value=4400,value=85)
    hours_per_week=st.number_input("Working Hours(Per Week):",min_value=0,value=42)

    #Inputs Boxes For Categorical variables
    workclass=st.selectbox('Work Class',['Private','Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked'])
    marital_status=st.selectbox('Marital Status',['Widowed','Divorced','Separated','Married-civ-spouse','Married-spouse-absent','Married-AF-spouse','Never-married'])
    occupation=st.selectbox('Occupation',['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'])
    relationship=st.selectbox('Relationship',['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'])
    native_country=st.text_input('Enter your Native Country')
    
    native_country_encoded=country_mapping.get(native_country,-1)
    #Creating a datframe with user inputs
    data=pd.DataFrame({
        'age':[age],
        'fnlwgt':[fnlwgt],
        'education.num':[education_num],
        'capital.gain':[capital_gain],
        'capital.loss':[capital_loss],
        'hours.per.week':[hours_per_week],

        'workclass':[{'Federal-gov':0,'Local-gov':1,'Never-worked':2,'Private':3,'Self-emp-inc':4,'Self-emo-not-inc':5,'State-gov':6,'Without-pay':7}[workclass]],
        'marital.status':[{'Widowed':6,'Divorced':0,'Separated':5,'Married-civ-spouse':2,'Married-spouse-absent':3,'Married-AF-spouse':1,'Never-married':0}[marital_status]],
        'occupation':[{'Prof-specialty':9,'Craft-repair':2,'Exec-managerial':3,'Adm-clerical':0,'Sales':11,'Other-service':7,'Machine-op-inspct':6,'Transport-moving':13,'Handlers-cleaners':5,'Farming-fishing':4,'Tech-support':12,'Protective-serv':10,'Priv-house-serv':8,'Armed-Forces':1}[occupation]],
        'relationship':[{'Husband':0,'Not-in-family':1,'Own-child':3,'Unmarried':4,'Wife':5,'Other-relative':2}[relationship]],
        'native.country':[native_country_encoded]
    })

    st.write("Input data for prediction")
    st.write(data)


    # Create a button to trigger prediction
    if st.button('Predict'):
        try:
            # Make prediction
            prediction = model.predict(data)
            probabilities = model.predict_proba(data)

            # Get predicted class and confidence level
            predicted_class = prediction[0]
            confidence = probabilities[0][predicted_class]

            income = income_prediction.get(prediction[0], 'Unknown Income Category')
            st.write(f'Predicted Income Category: {income}')
            st.write(f'Confidence Level: {confidence:.2f}')
        except Exception as e:
            st.write(f"Error during prediction: {e}")
if __name__=='__main__':
    main()  
