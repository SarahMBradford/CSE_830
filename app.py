#!/usr/bin/env python
# coding: utf-8
# Sarah Bradford
# CSE 830
# 10/16/2023


# Importing Libraries
# In[21]:
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
import graphviz as graph
import hiplot as hip
from PIL import Image
# Website Build
st.subheader("Sarah Bradford")

st.title("Welcome to Starbucks, what would you like to know?")
# # <div class="alert alert-success"> **Starbucks Drinks: Is it good for you?** ☕️</div>
image = Image.open('starbucks_pic.jpg')
incoffee_image = image.resize((600, 400))
st.image(incoffee_image)
st.subheader("Introduction")
st.write(" Everyone loves a good “pick-me-up”, especially the one that has a line full of people all hours throughout the day. Starbucks has been at the top of the coffeehouse connoisseur chain serving the most delicious drinks while delivering phenomenal customer service. Although, Starbucks is a favorite amongst many types of people, the discussion of its nutritional values has not been raised enough. The data science project: Starbucks: Is it good for you?: intends to explore the nutriuental values on 241 of their most popular drink combinations containing caffeine. This is an important project, because we are living in a world that is fast paced and often relies on caffeine to get through the day everyday. According to the FDA, caffeine as well as all other nutrients should be consumed in moderation or there will be side effects. ")

# # <div class="alert alert-success"> EDA </div>

# **Before diving into the project, we want to load the starbucks drink menu into a pandas dataframe and then use .head() to view the columns we are working with.**

# In[22]:
starbucks = pd.read_csv("starbucks_drinks.csv") # read in the csv file
starbucks.head()
starbucks_values = pd.read_csv("starbucks_drinks.csv") # read in the csv file
# In[23]:
starbucks.info()
# In[24]:
#Dropping Daily Value Percentage Columns from Both Data Set
starbucks = starbucks.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)
starbucks_values = starbucks_values.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)
# **By the looks of it we have our traditional nutritional label in a dataframe with the most popular drinks on the menu. Next, we will use .describe() to see if there are any missing values present in the dataset and need some data cleaning.**

# In[25]: 
# Description of the Starbucks Drink dataset
starbucks.describe()
# **It looks like this is a clean dataset with 241 drinks to choose from, but the daily value percentage and caffeine categories are missing so we will need to do some converting to add them back to the dataframe. So, we must take a closer look using the .isna().**
# In[26]:
# Dropping the rows with missing values from datasets
starbucks.isna().sum()
# In[27]:
starbucks.dropna(axis=0, inplace=True)
starbucks_values.dropna(axis=0, inplace=True)
# Changing the object data types to numeric
# In[28]:
dv_enc = LabelEncoder()
for i in starbucks_values.columns:
    if starbucks_values[i].dtype=='object':
        starbucks_values[i]=dv_enc.fit_transform(starbucks_values[i])

# In[29]:
starbucks['Beverage_category'].unique() # looking at unique drink names
# In[ ]:
# In[30]:
starbucks_fig = plt.figure(figsize=(10,8)); ax = starbucks_fig.gca()
starbucks.hist(color='green', bins=30, ax=ax)
plt.suptitle('Starbucks Drinks: Nutruitional Values', y=1.03)

# In[ ]:
# Give users the option to view the statistics and the dataframe
col1,col2 = st.columns(2,gap='small')
starbucks_statistics= col1.checkbox('Display the description of the Starbucks Drink dataset')
if starbucks_statistics==True:
    st.table(starbucks.describe())
    st.markdown('<p class="font_subtext">Table 1: Description of the statistics in the Starbucks Drink dataset', unsafe_allow_html=True)

starbucks_show = col2.checkbox('Display the Starbucks Drink dataframe')
if starbucks_show==True:
    st.table(starbucks)
    st.markdown('Starbucks Drink Dataframe', unsafe_allow_html=True)
#st.pyplot_chart(starbucks_fig)   
st.write("Now, that you've gotten acquainted with the Starbucks Drinks dataset, it is time to explore some trends! Begin your journey by clicking on the EDA tab.")
# In[31]:
# **All columns will now be used. Now, I want to see what nutrients are correlated with one another.**
# 
# In[13]:
sorted_starbucks = starbucks.sort_values(by='Beverage_category')
#sorted_starbucks.to_csv('sorted_starbucks.csv', index=False)
#sorted_starbucks.to_excel('sorted_starbucks.xlsx', index=False)
# In[14]:
tab0,tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(["Data Preprocessing","EDA", "Classic Espresso Drinks", "Coffee", "Coffee Frappuccinos", "Creme Frappuccinos", "Lightly Blended Frappucinos", "Shaken Iced Drinks", "Signature Expressos", "Smoothies", "Tazo Teas", "Build Nutritional Label", "References"])
with tab0: # Data Preprocessing Step Walkthrough
    st.subheader("Data Preprocessing")
    st.write("The first step in any data science project is to preprocess the data. This process includes cleaning the data, removing missing values, and converting the data types to numeric or the same data type. The Starbucks Drinks dataset was not too complex, but the daily value percentage columns were not necessary for this project and some of the caffeine rows were missing, so they were dropped using the .dropna fuction to remove them. Next, for the purpose of visualizing a correlation map a label encoder was used to temporarily change the data types of the categorical object columns to numerical values. This had to be done because the correlation map only works with numerical values. Then the original starbucks dataset was used with their unique values, so that users could see their drinks and the prep that goes into them.")
    st.write("**Data Preprocessing Steps:**")
    st.write("1. Starbucks.info to see the data types and missing values")
    st.write("2. Starbucks.describe to see the statistics of the dataset")
    st.write("3. Starbucks.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1) to drop the daily value percentage columns")
    st.write("4. Starbucks.isna().sum() to see the missing values")
    st.write("5. Starbucks.dropna(axis=0, inplace=True) to drop the missing values")
    st.write("6. LabelEncoder() to change the object data types to numeric")
    st.write("7. Starbucks['Beverage_category'].unique() to see the unique drink names")
    st.write("8. Starbucks.sort_values(by='Beverage_category') to sort the drinks by category")
    st.write("9. Starbucks.to_csv('sorted_starbucks.csv', index=False) to save the sorted dataset")
    st.write("10. Each Beverage_category was saved as a new dataframe to be used for the project.")

with tab1:
    st.subheader("FDA Recommendations & Nutrient Correlation Map")
    st.write("The United States Food and Drug Administration has recommendations of nutrients and how they should be consumed responsibly. From this, I began performing exploratory data analysis to see what how much of nutrients was included in these popular beverages, what type of beverage, and what nutrients were correlated with one another. Through my EDA efforts, I found Cholestoral & Calories have a correlation of 0.94, Sodium & Saturated Fat have a correlation of 0.92, Sugars & Calories have a correlation of 0.91, Sugars & Total Carbs have a correlation of  0.77 and Sodium & Transfat have a correlation of  0.71. All of these nutrients were at the top of the list for what the FDA recommends people consume in moderation. So, I decided to explore the relationship between all nutrients along with caffeine to see what people a consuming.")
    st.subheader('Starbucks Nutrient Correlation Map')
    plt.subplots(figsize=(25,20))
    heatmap = sns.heatmap(starbucks_values.corr(), cmap='Greens', annot=True)
    heatmap_fig = px.imshow(starbucks_values.corr(), text_auto=True, color_continuous_scale='Greens')
    st.plotly_chart(heatmap_fig)

    st.write("**The highest correlated relationships:**")
    st.write("*Cholestoral & Calories 0.94*")
    st.write("*Sodium & Saturated Fat 0.92*")
    st.write("*Sugars & Calories 0.91*")
    st.write("*Sugars & Total Carbs 0.77*")
    FDA_Recs = pd.read_csv("FDA_Recs.csv")
    st.write("Given the FDA Recommendations below, it is time to explore the highest correlated relationships in each Beverage Category. ")
    st.table(FDA_Recs)
    # Glimpse of Nutriutional Values
    na_image = Image.open("na.jpeg")
    na_image = na_image.resize((600, 600))
    st.image(na_image)
    
with tab1:
    #EDA
    st.subheader("EDA")
    st.write("Data Visualizations for the Most Popular Starbucks Drinks")
    cho_col1, sod_col2, cal_col3, trans_col4 = st.columns(4, gap='large')
    cho = cho_col1.checkbox('Cholesterol (mg) & Calories')
    if cho==True:
        plt.figure(figsize=(25,8))
        scat1 = sns.scatterplot(data=starbucks, x="Cholesterol (mg)", y="Calories", hue="Beverage_category", palette="Greens")
        plt.title("Scatterplot of Cholesterol (mg) & Calories Based on Beverage Category")
        st.pyplot(scat1.figure)
        plt.figure(figsize=(25,8))
        bar1 = sns.barplot(data=starbucks, x="Cholesterol (mg)", y="Calories", hue="Beverage_category", palette="Greens")
        plt.title("Barplot of Cholesterol (mg) & Calories Based on Beverage Category")
        st.pyplot(bar1.figure)
        plt.figure(figsize=(25,8))
        violin1 = sns.violinplot(data=starbucks, x="Cholesterol (mg)", y="Calories", hue="Beverage_category", palette="Greens")
        plt.title("Violinplot of Cholesterol (mg) & Calories Based on Beverage Category")
        st.pyplot(violin1.figure)
     
       
    sod = sod_col2.checkbox('Sodium (mg) & Saturated Fat (g)')
    if sod==True:
        plt.figure(figsize=(25,8))
        scat2 = sns.scatterplot(data=starbucks, x="Sodium (mg)", y="Saturated Fat (g)", hue="Beverage_category", palette="Greens")
        plt.title("Scatterplot of Sodium (mg) & Saturated Fat (g) Based on Beverage Category")
        st.pyplot(scat2.figure)
        plt.figure(figsize=(25,8))
        bar2 = sns.barplot(data=starbucks, x="Sodium (mg)", y="Saturated Fat (g)", hue="Beverage_category", palette="Greens")
        plt.title("Barplot of Sodium (mg) & Saturated Fat (g) Based on Beverage Category")
        st.pyplot(bar2.figure)
        plt.figure(figsize=(25,8))
        violin2 = sns.violinplot(data=starbucks, x="Sodium (mg)", y="Saturated Fat (g)", hue="Beverage_category", palette="Greens")
        plt.title("Violinplot of Sodium (mg) & Saturated Fat (g) Based on Beverage Category")
        st.pyplot(violin2.figure)
    
      
    cal_col3 = cal_col3.checkbox('Calories & Sugars (g)')
    if cal_col3==True:
        plt.figure(figsize=(25,8))
        scat3 = sns.scatterplot(data=starbucks, x="Calories", y="Sugars (g)", hue="Beverage_category", palette="Greens")
        plt.title("Scatterplot of Calories & Sugars (g) Based on Beverage Category")
        st.pyplot(scat3.figure)
        plt.figure(figsize=(25,8))
        bar3 = sns.barplot(data=starbucks, x="Calories", y="Sugars (g)", hue="Beverage_category", palette="Greens")
        plt.title("Barplot of Calories & Sugars (g) Based on Beverage Category")
        st.pyplot(bar3.figure)
        plt.figure(figsize=(25,8))
        violin3 = sns.violinplot(data=starbucks, x="Calories", y="Sugars (g)", hue="Beverage_category", palette="Greens")
        plt.title("Violinplot of Calories & Sugars (g) Based on Beverage Category")
        st.pyplot(violin3.figure)
    
      
    trans_col4 = trans_col4.checkbox("Sodium (mg) and Trans Fat (g)")
    if trans_col4 == True:
        plt.figure(figsize=(25,8))
        scat4 = sns.scatterplot(data=starbucks, x="Sodium (mg)", y="Trans Fat (g) ",hue="Beverage_category")
        plt.title("Scatterplot of Sodium (mg) & Trans fat (g) Based on Beverage Category")
        st.pyplot(scat4.figure)
        plt.figure(figsize=(25,8))
        bar4 = sns.barplot(data=starbucks, x="Sodium (mg)", y="Trans Fat (g) ",hue="Beverage_category")
        plt.title("Barplot of Sodium (mg) & Trans fat (g) Based on Beverage Category")
        st.pyplot(bar4.figure)
        plt.figure(figsize=(25,8))
        violin4 = sns.violinplot(data=starbucks, x="Sodium (mg)", y="Trans Fat (g) ",hue="Beverage_category")
        plt.title("Violinplot of Sodium (mg) & Trans fat (g) Based on Beverage Category")
        st.pyplot(violin4.figure)
        

with tab1: 
    st.subheader("Top 10 Beverages with the Highest Caffeine Content")
    filtered_starbucks = starbucks[(~starbucks['Caffeine (mg)'].str.lower().str.contains('varies')) & (starbucks['Caffeine (mg)'].str.isnumeric())]
    filtered_starbucks['Caffeine (mg)'] = pd.to_numeric(filtered_starbucks['Caffeine (mg)'])
    top_caf = filtered_starbucks.groupby(['Beverage_category', 'Beverage', 'Beverage_prep'])['Caffeine (mg)'].max().sort_values(ascending=False).head(10)
    st.write(top_caf)
    
    st.subheader("Top 10 Beverages with the Highest Calories")
    top_cal = starbucks.groupby(['Beverage_category', 'Beverage', 'Beverage_prep'])['Calories'].max().sort_values(ascending=False).head(10)
    st.write(top_cal)
    
    st.subheader("Top 10 Beverages with the Highest Cholesterol")
    top_chol = starbucks.groupby(['Beverage_category', 'Beverage', 'Beverage_prep'])['Cholesterol (mg)'].max().sort_values(ascending=False).head(10)
    st.write(top_chol)
    
    st.subheader("Top 10 Beverages with the Highest Sodium")
    top_sod = starbucks.groupby(['Beverage_category', 'Beverage', 'Beverage_prep'])['Sodium (mg)'].max().sort_values(ascending=False).head(10)
    st.write(top_sod)
    
    st.subheader("Summary of EDA")
    st.write("After completing exploratory data analysis, it is safe to conclude coffee by itself is not as harmful when considering caffeine content as any of Starbucks' other beverages. The added ingredients that transform coffee into the various categories cause a spike in all nutritional values.")

# In[32]:
with tab2:
    st.header("Espresso Drinks")
    espresso_image = Image.open("starbucks-mocha-drinks-2-2.jpg")
    espresso_image = espresso_image.resize((450, 400))
    st.image(espresso_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    classic_espresso = sorted_starbucks[0:58]
    classic_espresso_fig = plt.figure(figsize=(10,8)); ax = classic_espresso_fig.gca()
    classic_espresso.hist(color='green', bins=30, ax=ax)
    plt.suptitle('Classic Espresso: Nutruitional Values', y=1.03)
    plt.tight_layout()
    
    single = alt.selection_point(on='mouseover', nearest=True)
esp_cal_chol = (
    alt.Chart(classic_espresso)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']  # corrected variable name
    )
    .interactive()
    .add_selection(single)
)
    
esp_sod_sat = (
    alt.Chart(classic_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
esp_sug_cal = (
    alt.Chart(classic_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']  # corrected variable name
    )
    .interactive()
    .add_selection(single)
)
    
esp_sug_carb = (
    alt.Chart(classic_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab2:
    esp_col1,esp_col2,esp_col3,esp_col4 = st.columns(4, gap='large')
esp1 = esp_col1.checkbox('Display the scatterplot of the First Highest Correlated Relationship')
if esp1==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Calories and Cholestoral")
    st.altair_chart(esp_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Espresso Drinks with Calories and Cholestoral', unsafe_allow_html=True)

esp2 = esp_col2.checkbox('Display the scatterplot of the Second Highest Correlated Relationship')
if esp2==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sodium and Saturated Fat")
    st.altair_chart(esp_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Espresso Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

esp3 = esp_col3.checkbox('Display the scatterplot of the Third Highest Correlated Relationship')
if esp3==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sugars and Calories")
    st.altair_chart(esp_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Espresso Drinks with Sugars and Calories', unsafe_allow_html=True) 

esp4 = esp_col4.checkbox('Display the Scatterplot of the Fourth Highest Correlated Relationships')
if esp4==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(esp_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Espresso Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 


with tab3:
    st.header("Coffee Drinks")
    coffee_image = Image.open("best-starbucks-iced-coffee-drinks.jpg")
    coffee_image = coffee_image.resize((450, 400))
    st.image(coffee_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    coffee = sorted_starbucks[58:62]
    single = alt.selection_point(on='mouseover', nearest=True)
coffee_cal_chol = (
    alt.Chart(coffee)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
coffee_sod_sat = (
    alt.Chart(coffee)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
coffee_sug_cal = (
    alt.Chart(coffee)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
coffee_sug_carb = (
    alt.Chart(coffee)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab3:
    coffee_col1,coffee_col2,coffee_col3,coffee_col4 = st.columns(4, gap='large')
coffee1 = coffee_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Coffee Drinks')
if coffee1==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Calories and Cholestoral")
    st.altair_chart(coffee_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Coffee Drinks with Calories and Cholestoral', unsafe_allow_html=True)

coffee2 = coffee_col2.checkbox('Display the scatterplot of the Second Highest Correlated Relationship for Coffee Drinks')
if coffee2==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Sodium and Saturated Fat")
    st.altair_chart(coffee_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Coffee Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

coffee3 = coffee_col3.checkbox('Display the scatterplot of the Third Highest Correlated Relationship for Coffee Drinks')
if coffee3==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Sugars and Calories")
    st.altair_chart(coffee_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Coffee Drinks with Sugars and Calories', unsafe_allow_html=True) 

      

# In[33]:
with tab4:
    st.header("Coffee Frappuccinos")
    frap_image = Image.open("frap.jpg")
    frap_image = frap_image.resize((450, 400))
    st.image(frap_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    frappuccino_coffee = sorted_starbucks[62:98]
    single = alt.selection_point(on='mouseover', nearest=True)
coff_cal_chol = (
    alt.Chart(frappuccino_coffee)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
coff_sod_sat = (
    alt.Chart(frappuccino_coffee)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
coff_sug_cal = (
    alt.Chart(frappuccino_coffee)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
coff_sug_carb = (
    alt.Chart(frappuccino_coffee)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab4:
    coff_col1,coff_col2,coff_col3,coff_col4 = st.columns(4, gap='large')
coff1 = coff_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship ')
if coff1==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccio Drinks with Calories and Cholestoral")
    st.altair_chart(coff_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Coffee Frappuccio Drinks with Calories and Cholestoral', unsafe_allow_html=True)

coff2 = coff_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship')
if coff2==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccio Drinks with Sodium and Saturated Fat")
    st.altair_chart(coff_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Coffee Frappuccio Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

coff3 = coff_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship')
if coff3==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccio Drinks with Sugars and Calories")
    st.altair_chart(coff_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Coffee Frappuccio Drinks with Sugars and Calories', unsafe_allow_html=True) 

coff4 = coff_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationships')
if coff4==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccio Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(coff_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Coffee Frappuccio Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
    
####
with tab5:
    st.header("Creme Frappuccinos")
    creme_image = Image.open("vanilla.jpg")
    creme_image = creme_image.resize((450, 400))
    st.image(creme_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    frappuccino_creme = sorted_starbucks[98:111]
    single = alt.selection_point(on='mouseover', nearest=True)
c_cal_chol = (
    alt.Chart(frappuccino_creme)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
c_sod_sat = (
    alt.Chart(frappuccino_creme)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
c_sug_cal = (
    alt.Chart(frappuccino_creme)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
c_sug_carb = (
    alt.Chart(frappuccino_creme)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab5:
    c_col1,c_col2,c_col3,c_col4 = st.columns(4, gap='large')
c1 = c_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Creme Frappuccio Drinks')
if c1==True:
    st.subheader("Relationships between Caffienated Creme Frappuccio Drinks with Calories and Cholestoral")
    st.altair_chart(c_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Creme Frappuccio Drinks with Calories and Cholestoral', unsafe_allow_html=True)

c2 = c_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Creme Frappuccio Drinks')
if c2==True:
    st.subheader("Relationships between Caffienated Creme Frappuccio Drinks with Sodium and Saturated Fat")
    st.altair_chart(c_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Creme Frappuccio Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

c3 = c_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Creme Frappuccio Drinks')
if c3==True:
    st.subheader("Relationships between Caffienated Creme Frappuccio Drinks with Sugars and Calories")
    st.altair_chart(c_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Creme Frappuccio Drinks with Sugars and Calories', unsafe_allow_html=True) 

c4 = c_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Creme Frappuccio Drinks')
if c4==True:
    st.subheader("Relationships between Caffienated Creme Frappuccio Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(c_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Creme Frappuccio Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 


##
with tab6:
    st.header("Lightly Blended Frappuccinos")
    light_image = Image.open("Light_Frappuccino-1.jpg")
    light_image = light_image.resize((450, 400))
    st.image(light_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    frappuccino_lightblend = sorted_starbucks[111:123]
    single = alt.selection_point(on='mouseover', nearest=True)
light_cal_chol = (
    alt.Chart(frappuccino_lightblend)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
light_sod_sat = (
    alt.Chart(frappuccino_lightblend)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
light_sug_cal = (
    alt.Chart(frappuccino_lightblend)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
light_sug_carb = (
    alt.Chart(frappuccino_lightblend)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab6:
    light_col1,light_col2,light_col3,light_col4 = st.columns(4, gap='large')
light1 = light_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Lightly Blended Frappuccio Drinks')
if light1==True:
    st.subheader("Relationships between Caffienated Lightly Blended Frappuccio Drinks with Calories and Cholestoral")
    st.altair_chart(light_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Lightly Blended Frappuccio Drinks with Calories and Cholestoral', unsafe_allow_html=True)

light2 = light_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Lightly Blended Frappuccio Drinks')
if light2==True:
    st.subheader("Relationships between Caffienated Lightly Blended Frappuccio Drinks with Sodium and Saturated Fat")
    st.altair_chart(light_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Lightly Blended Frappuccio Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

light3 = light_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Lightly Blended Frappuccio Drinks')
if light3==True:
    st.subheader("Relationships between Caffienated Lightly Blended Frappuccio Drinks with Sugars and Calories")
    st.altair_chart(light_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Lightly Blended Frappuccio Drinks with Sugars and Calories', unsafe_allow_html=True) 

light4 = light_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Lightly Blended Frappuccio Drinks')
if light4==True:
    st.subheader("Relationships between Caffienated Lightly Blended Frappuccio Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(light_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Lightly Blended Frappuccio Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
###


##

with tab7:
    st.header("Shaken Iced Drinks")
    shaken_image = Image.open("shaken.jpg")
    shaken_image = shaken_image.resize((450, 400))
    st.image(shaken_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    shaken_iced = sorted_starbucks[123:140]
    single = alt.selection_point(on='mouseover', nearest=True)
shaken_cal_chol = (
    alt.Chart(shaken_iced)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
shaken_sod_sat = (
    alt.Chart(shaken_iced)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
shaken_sug_cal = (
    alt.Chart(shaken_iced)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
shaken_sug_carb = (
    alt.Chart(shaken_iced)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab7:
    shaken_col1,shaken_col2, shaken_col3,shaken_col4 = st.columns(4, gap='large')
shaken1 = shaken_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Shaken Iced Drinks')
if shaken1==True:
    st.subheader("Relationships between Caffienated Shaken Iced Drinks with Calories and Cholestoral")
    st.altair_chart(shaken_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Shaken Iced Drinks with Calories and Cholestoral', unsafe_allow_html=True)

shaken2 = shaken_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Shaken Iced Drinks')
if shaken2==True:
    st.subheader("Relationships between Caffienated Shaken Iced Drinks with Sodium and Saturated Fat")
    st.altair_chart(shaken_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Shaken Iced Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

shaken3 = shaken_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Shaken Iced Drinks')
if shaken3==True:
    st.subheader("Relationships between Caffienated Shaken Iced Drinks with Sugars and Calories")
    st.altair_chart(shaken_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Shaken Iced Drinks with Sugars and Calories', unsafe_allow_html=True) 

shaken4 = shaken_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Shaken Iced Drinks')
if shaken4==True:
    st.subheader("Relationships between Caffienated Shaken Iced Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(shaken_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Shaken Iced Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 




###
with tab8:
    st.header("Signature Espresso Drinks")
    sig_image = Image.open("blonde_espresso_20180109-tease.jpg")
    sig_image = sig_image.resize((450, 400))
    st.image(sig_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    signature_espresso = sorted_starbucks[140:180]
    single = alt.selection_point(on='mouseover', nearest=True)
sig_cal_chol = (
    alt.Chart(signature_espresso)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
sig_sod_sat = (
    alt.Chart(signature_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
sig_sug_cal = (
    alt.Chart(signature_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
sig_sug_carb = (
    alt.Chart(signature_espresso)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab8:
    sig_col1,sig_col2,sig_col3,sig_col4 = st.columns(4, gap='large')
sig1 = sig_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Signature Espresso Drinks')
if sig1==True:
    st.subheader("Relationships between Caffienated Signature Espresso Drinks with Calories and Cholestoral")
    st.altair_chart(sig_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Signature Espresso Drinks with Calories and Cholestoral', unsafe_allow_html=True)

sig2 = sig_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Signature Espresso Frappuccio Drinks')
if sig2==True:
    st.subheader("Relationships between Caffienated Signature Espresso Drinks with Sodium and Saturated Fat")
    st.altair_chart(sig_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Signature Espresso Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

sig3 = sig_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Signature Espresso Drinks')
if sig3==True:
    st.subheader("Relationships between Caffienated Signature Espresso Drinks with Sugars and Calories")
    st.altair_chart(sig_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Signature Espresso Drinks with Sugars and Calories', unsafe_allow_html=True) 

sig4 = sig_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Signature Espresso Drinks')
if sig4==True:
    st.subheader("Relationships between Caffienated Signature Espresso Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(sig_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Signature Espresso Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 




###
with tab9:
    st.header("Smoothies")
    smoothie_image = Image.open("Does-Starbucks-Have-Smoothies-2.png")
    smoothie_image = smoothie_image.resize((450, 400))
    st.image(smoothie_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    smoothies = sorted_starbucks[180:189]
    single = alt.selection_point(on='mouseover', nearest=True)
sm_cal_chol = (
    alt.Chart(smoothies)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
sm_sod_sat = (
    alt.Chart(smoothies)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
sm_sug_cal = (
    alt.Chart(smoothies)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
sm_sug_carb = (
    alt.Chart(smoothies)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab9:
    sm_col1,sm_col2,sm_col3,sm_col4 = st.columns(4, gap='large')
sm1 = sm_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Smoothies')
if sm1==True:
    st.subheader("Relationships between Caffienated Smoothies with Calories and Cholestoral")
    st.altair_chart(sm_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Smoothies with Calories and Cholestoral', unsafe_allow_html=True)

sm2 = sm_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Smoothies')
if sm2==True:
    st.subheader("Relationships between Caffienated Smoothies with Sodium and Saturated Fat")
    st.altair_chart(sm_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Smoothies with Sodium and Saturated Fat', unsafe_allow_html=True) 

sm3 = sm_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Smoothies')
if sm3==True:
    st.subheader("Relationships between Caffienated Smoothies with Sugars and Calories")
    st.altair_chart(sm_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Smoothies with Sugars and Calories', unsafe_allow_html=True) 

sm4 = sm_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Smoothies')
if sm4==True:
    st.subheader("Relationships between Caffienated Smoothies with Sugars and Total Carbohydrates")
    st.altair_chart(sm_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Smoothies with Sugars and Total Carbohydrates', unsafe_allow_html=True) 



###
with tab10:
    st.header("Tazo Teas")
    tea_image = Image.open("starbucks-tazo.top.jpg")
    tea_image = tea_image.resize((450, 400))
    st.image(tea_image)
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
    tazo_teas = sorted_starbucks[189:241]
    single = alt.selection_point(on='mouseover', nearest=True)
t_cal_chol = (
    alt.Chart(tazo_teas)
    .mark_circle(size=200)
    .encode(
        x='Calories:Q',
        y='Cholesterol (mg):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Calories:Q', 'Cholesterol (mg):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
t_sod_sat = (
    alt.Chart(tazo_teas)
    .mark_circle(size=200)
    .encode(
        x='Sodium (mg):Q',
        y='Saturated Fat (g):Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sodium (mg):Q', 'Saturated Fat (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
) 
t_sug_cal = (
    alt.Chart(tazo_teas)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Calories:Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars (g):Q', 'Calories:Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)
    
t_sug_carb = (
    alt.Chart(tazo_teas)
    .mark_circle(size=200)
    .encode(
        x='Sugars (g):Q',
        y='Total Carbohydrates (g) :Q',
        size='Caffeine (mg):Q',
        color=alt.condition(single, 'Beverage:N', alt.value('lightgray')),
        tooltip=['Sugars(g):Q', 'Total Carbohydrates (g):Q', 'Caffeine (mg):Q', 'Beverage:N']
    )
    .interactive()
    .add_selection(single)
)

with tab10:
    t_col1,t_col2,t_col3,t_col4 = st.columns(4, gap='large')
t1 = t_col1.checkbox(' Display the scatterplot of the First Highest Correlated Relationship for Tazo Teas')
if t1==True:
    st.subheader("Relationships between Caffienated Tazo Teas with Calories and Cholestoral")
    st.altair_chart(t_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Tazo Teas with Calories and Cholestoral', unsafe_allow_html=True)

t2 = t_col2.checkbox(' Display the scatterplot of the Second Highest Correlated Relationship for Tazo Teas')
if t2==True:
    st.subheader("Relationships between Caffienated Tazo Teas with Sodium and Saturated Fat")
    st.altair_chart(t_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Tazo Teas with Sodium and Saturated Fat', unsafe_allow_html=True) 

t3 = t_col3.checkbox(' Display the scatterplot of the Third Highest Correlated Relationship for Tazo Teas')
if t3==True:
    st.subheader("Relationships between Caffienated Tazo Teas with Sugars and Calories")
    st.altair_chart(t_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Tazo Teas with Sugars and Calories', unsafe_allow_html=True) 

t4 = t_col4.checkbox(' Display the Scatterplot of the Fourth Highest Correlated Relationship for Tazo Teas')
if t4==True:
    st.subheader("Relationships between Caffienated Tazo Teas with Sugars and Total Carbohydrates")
    st.altair_chart(t_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Tazo Teas with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
###

### Seperation of Beverage Preparation
# In[ ]:
# In[Build Your Drink]:
with tab11:
    st.subheader("Build Your Favorite Drink's Nutritional Label")
    st.write("Now that you have finished exploring the different drinks and the four highest correlated relationships, it is time to see the nutritional make-up of your favorite drink. Use the sidecar select box to favorite beverage and beverage preparation to see the nutritional label for your drink.")
    st.write("***Special Notes***")
    st.write("  **Please Read Before Selecting Your Drink**")
    st.write("*Note: If you do not see your favorite drink, it is because it is not included in the dataset. Please select a different drink.*")
    st.write("*Note: If you do not see your favorite beverage preparation, it is because it is not included in the dataset. Please select a different beverage preparation.*")
    st.write("*Note: When selecting your drink's preperation, if you choose 'Tall', 'Grande', or 'Venti' you will see the nutritional label for original recipe for the drink in a 12 oz, 16 oz, or 20 oz cup, respectively.*")
 
with tab11:    
    st.title("Your Turn to Build Your Drink's Nutritional Label")
    # User can use sidebar for selecting a beverage and preparation
    selected_beverage = st.sidebar.selectbox("Select a beverage", starbucks['Beverage'].unique())
    selected_prep = st.sidebar.selectbox("Select prep", starbucks['Beverage_prep'].unique())

    # select multiple nutrients using multiselect
    nutrient_columns = starbucks.columns[3:18]
    selected_nutrients = st.sidebar.multiselect("Select nutrients", nutrient_columns)

    if selected_beverage and selected_prep and selected_nutrients:
    # Filter data based on the selected beverage and preparation
        filtered_data = starbucks[(starbucks['Beverage'] == selected_beverage) & (starbucks['Beverage_prep'] == selected_prep)]

    # radial chart for the selected drink's nutrients
        radial_chart = alt.Chart(filtered_data).transform_fold(
            selected_nutrients,
            as_=['Nutrient', 'Value']
        ).mark_bar().encode(
            alt.X('Nutrient:N', title='Nutrient', scale=alt.Scale(domain=selected_nutrients)),
            alt.Y('Value:Q', title='Value'),
            color=alt.Color('Nutrient:N', scale=alt.Scale(scheme='yellowgreen'))
        ).properties(width=400)
        st.subheader("Nutritional Label for Your Drink")

        st.write(radial_chart)
        st.write("Hover over the bars to see the nutritional value of each nutrient in your drink.")
    # pie chart for the selected drink's nutrients
        pie_chart = alt.Chart(filtered_data).transform_fold(
            selected_nutrients,
            as_=['Nutrient', 'Value']
        ).mark_arc().encode(
            theta='Value:Q',
            color=alt.Color('Nutrient:N', scale=alt.Scale(scheme='yellowgreen')),
            tooltip=['Nutrient:N', 'Value:Q']
        ).properties(width=400, height=400)
        st.write("Hover over the pie chart to see the nutritional value of each nutrient in your drink.")
        st.write(pie_chart)
    st.subheader("Conclusion")
    st.write("After reviewing the nutritional label for your drink, multiply the values by the number of times you visit a Starbucks. With this information, you can compare your drink’s nutrition values to the FDA recommendations. Now, ask yourself should you be in line for Starbucks every day or multiple times a day despite your lack of energy?")
    st.write("In conclusion, there are numerous ways to build a beverage at Starbucks to keep customers energized throughout the day, while enjoying a delicious treat. However, it is important to be aware of the nutritional value of your drink and how it compares to the FDA recommendations. With this information, you can make an informed decision about your drink and how often you should be in line at Starbucks.")
with tab12:
    st.subheader("References")  
    st.write("1. https://www.kaggle.com/starbucks/starbucks-menu")
    st.write("2. https://www.starbucks.com/menu/drinks")
    st.write("3. https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels") 
