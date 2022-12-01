# deploying my model to streamlit cloud
import numpy as np
import streamlit as st
import pandas as pd 
import joblib

model = joblib.load("nn_model.joblib")

def data_processing(df):
    """
    Function that converts string data into numeric data for the model.
    """
    df.Sex = df.Sex.map({'Male':0, 'Female':1})
    df.MeetCountry = df.MeetCountry.map(country_list)
    df.Equipment = df.Equipment.map({'Yes': 1, 'No': 0})
    
    return df

def get():
    #AGE OF CONTESTANT
    age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=0
            )
    
    # SEX OF CONTESTANT
    sex = st.selectbox(
            "Sex",
            options=['Male',"Female"],
            help="Select your sex from the list."
            )

    body_weight = st.slider(
            "Bodyweight (in Kg)",
            min_value=20,
            max_value=200,
            value= 50
            )

    # MEET COUNTRY OF POWERLIFTING EVENT
    country = st.selectbox(
            "Meet Country",
            options=['Germany', 'Ukraine', 'USA', 'Japan', 'Slovenia', 'Wales', 'UK',
            'England', 'Australia', 'New Zealand', 'Ireland', 'Canada',
            'Netherlands', 'Scotland', 'Lithuania', 'Indonesia', 'Tahiti',
            'Denmark', 'USSR', 'Sweden', 'Russia', 'Poland', 'Italy',
            'Belgium', 'France', 'Kazakhstan', 'South Africa', 'Finland',
            'Nauru', 'Slovakia', 'Norway', 'Papua New Guinea', 'Spain',
            'Czechia', 'Hungary', 'Austria', 'China', 'Morocco', 'Egypt',
            'Algeria', 'Brazil', 'India', 'Iceland', 'Belarus', 'Switzerland',
            'Luxembourg', 'Bulgaria', 'Estonia', 'Serbia', 'Argentina',
            'Colombia', 'Taiwan', 'Mongolia', 'Peru', 'Ivory Coast',
            'Puerto Rico', 'Uruguay', 'Turkey', 'South Korea', 'Philippines',
            'N.Ireland', 'Georgia', 'Samoa', 'New Caledonia', 'Singapore',
            'Fiji', 'Azerbaijan', 'Ecuador', 'Croatia', 'Uzbekistan', 'Iran',
            'Hong Kong', 'Mexico', 'Nicaragua', 'Aruba', 'Guyana',
            'US Virgin Islands', 'Venezuela', 'Cayman Islands', 'Guatemala',
            'Costa Rica', 'El Salvador', 'Thailand', 'Latvia'],
            help="Select which country the powerlifting meet was held in.",
            )

    # EQUIPMENT USAGE
    equipment = st.selectbox(
            "Are you going to use Equipment?",
            options=["Yes","No"],
            help="Select if you are going to use equipment."
            )

    wilks = st.slider(
            "Wilks Coefficient",
            min_value=20.00,
            max_value=800.00,
            step=0.01,
            value=330.00
            )

    glossbrenner = st.slider(
            "Glossbrenner Coefficient",
            min_value=20.00,
            max_value=700.00,
            step=0.01,
            value=300.00
            )

    mcculloch = st.slider(
            "McCulloch Coefficient",
            min_value=20.00,
            max_value=850.00,
            step=0.01,
            value=350.00
            )

    features = {
            'Age': age,
            'AgeClass': age,
            'Sex': sex,
            'BodyweightKg': body_weight,
            'MeetCountry': country,
            'Equipment': equipment,
            'Wilks': wilks,
            'Glossbrenner': glossbrenner,
            'McCulloch': mcculloch
            }
    
    df = pd.DataFrame(features,index=[0])

    return df
    

result = -1
country_list = {'Germany': 0,'Ukraine': 1,'USA': 2,'Japan': 3,'Slovenia': 4,'Wales': 5,'UK': 6,'England': 7,'Australia': 8,'New Zealand': 9,'Ireland': 10,'Canada': 11,'Netherlands': 12,'Scotland': 13,'Lithuania': 14,'Indonesia': 15,'Tahiti': 16,'Denmark': 17,'USSR': 18,'Sweden': 19,'Russia': 20,'Poland': 21,'Italy': 22,'Belgium': 23,'France': 24,'Kazakhstan': 25,'South Africa': 26,'Finland': 27,'Nauru': 28,'Slovakia': 29,'Norway': 30,'Papua New Guinea': 31,'Spain': 32,'Czechia': 33,'Hungary': 34,'Austria': 35,'China': 36,'Morocco': 37,'Egypt': 38,'Algeria': 39,'Brazil': 40,'India': 41,'Iceland': 42,'Belarus': 43,'Switzerland': 44,'Luxembourg': 45,'Bulgaria': 46,'Estonia': 47,'Serbia': 48,'Argentina': 49,'Colombia': 50,'Taiwan': 51,'Mongolia': 52,'Peru': 53,'Ivory Coast': 54,'Puerto Rico': 55,'Uruguay': 56,'Turkey': 57,'South Korea': 58,'Philippines': 59,'N.Ireland': 60,'Georgia': 61,'Samoa': 62,'New Caledonia': 63,'Singapore': 64,'Fiji': 65,'Azerbaijan': 66,'Ecuador': 67,'Croatia': 68,'Uzbekistan': 69,'Iran': 70,'Hong Kong': 71,'Mexico': 72,'Nicaragua': 73,'Aruba': 74,'Guyana': 75,'US Virgin Islands': 76,'Venezuela': 77,'Cayman Islands': 78,'Guatemala': 79,'Costa Rica': 80,'El Salvador': 81,'Thailand': 82,'Latvia': 83}

st.set_page_config(layout="wide")

st.markdown(""" 
# Powerlifting Performance Predictor
### Predict your performance at a meet!
### Use the parameters below!
### Possibile outcomes: EXCELLENT, GOOD, OKAY, BAD 
# 
""")
col1,col2 = st.columns(2)

with col1:
    df = get()

    df = data_processing(df=df)

# #     maxI = max(index[0])

#         i_old = np.where(index[0] == maxI)
#         i = i_old[0][0]
    
    r = model.predict(df)
    max_index = max(r[0])

    result = np.where(r[0] == max_index)[0][0]
    print('RESULT == ', result)
    print('df == ', df)




with col2:
    if result == -1:
        st.markdown("""
            ### ENTER PARAMETERS""")
    elif result == 0:
        st.markdown("""
            ### Performance: **_EXCELLENT_**""")
    elif result == 1:
        st.markdown("""
            ### Performance: **_GOOD_**""")
    elif result == 2:
        st.markdown("""
            ### Performance: **_OKAY_**""")
    elif result == 3:
        st.markdown("""
            ### Performance: **_BAD_**""")

