import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import numpy as np
import hiplot as hip


st.title("Starbucks Drinks: Are they good for you?")

starbucks = pd.read_csv("/Users/sarahbradford/Downloads/starbucks_drinks.csv")
starbucks.head()

starbucks.describe()

starbucks.isna().sum()

starbucks.dropna(axis=0, inplace=True)

dv_enc = LabelEncoder()
for i in starbucks.columns:
    if starbucks[i].dtype=='object':
        starbucks[i]=dv_enc.fit_transform(starbucks[i])

#hiplot
starbucks_hip = hip.Experiment.from_dataframe(starbucks).display()
starbucks_hip


starbucks.describe()
st.subheader('Starbucks Nutrients Correlation Map')
plt.subplots(figsize=(25,20))
heatmap = sns.heatmap(starbucks.corr(), cmap='Greens', annot=True)
heatmap_fig = px.imshow(starbucks.corr(), text_auto=True, color_continuous_scale='Greens')
st.plotly_chart(heatmap_fig)
st.subheader("The highest correlated relationships:")
st.write("Cholestoral & Calories 0.94")
st.write("Sodium & Saturated Fat 0.92")
st.write("Sugars & Calories 0.91")
st.write("Sugars & Total Carbs 0.77")
st.write("Sodium & Transfat 0.71")


st.title("What are you putting in your body when you consume starbucks?")
starbucks_data=[]
choice1 = x_variable = st.sidebar.selectbox("Select A Drink", starbucks.columns['Beverage'])
choice2 = y_variable = st.sidebar.selectbox("Select A Nutrient", starbucks.columns)
st.bar_chart(data = starbucks, x= choice1, y = choice2)




sorted_starbucks = starbucks.sort_values(by='Beverage_category')
classic_espresso = sorted_starbucks[0:58]
classic_espresso = classic_espresso.drop(['Beverage_category'], axis=1)
#classic_espresso = pd.DataFrame(classic_espresso)
#beverage = [str(item) for item in classic_espresso['Beverage'].tolist()]
#beverage = 
#beverage_choice = st.selectbox('Choose your beverage:', beverage)
