#!/usr/bin/env python
# coding: utf-8

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

st.title("Welcome to Starbucks, is the energy from your caffeinated beverages really worth it?")
# # <div class="alert alert-success"> **Starbucks Drinks: Are they good for you?** ☕️</div>
image = Image.open('starbucks_pic.jpg')
incoffee_image = image.resize((600, 400))
st.image(incoffee_image)
st.subheader("Introduction")
st.write(" Everyone loves a good “pick-me-up”, especially the one that has a line full of people all hours throughout the day. Starbucks has been at the top of the coffeehouse connoisseur chain serving the most delicious drinks while delivering phenomenal customer service. Although, Starbucks is a favorite amongst many types of people, the discussion of its nutritional values has not been raised enough. The data science project: Starbucks: Are the drinks healthy for you?, intends to explore the nutriuental values on 241 of their most popular drink combinations containing caffiene. This is an important project, because we are living in a world that is fast paced and often relies on caffiene to get through the day everyday. According to the FDA, caffiene as well as all other nutrients should be consumed in moderation or there will be side effects. ")

# # <div class="alert alert-success"> EDA </div>

# **Before diving into the project, we want to load the starbucks drink menu into a pandas dataframe and then use .head() to view the columns we are working with.**

# In[22]:
starbucks = pd.read_csv("starbucks_drinks.csv")
starbucks.head()
starbucks_values = pd.read_csv("starbucks_drinks.csv")
# In[23]:

# In[24]:

starbucks = starbucks.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)
starbucks_values = starbucks_values.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)

starbucks_numeric = ['Total Fat (g)', 'Caffeine (mg)', 'Total Carbohydrates (g) ']
starbucks[starbucks_numeric] = starbucks[starbucks_numeric].apply(pd.to_numeric, errors='coerce')
# **By the looks of it we have our traditional nutritional label in a dataframe with the most popular drinks on the menu. Next, we will use .describe() to see if there are any missing values present in the dataset and need some data cleaning.**

# In[ ]:

# In[25]:

# **It looks like this is a clean dataset with 242 drinks to choose from, but the daily value percentage and caffeine categories are missing so we will need to do some converting to add them back to the dataframe. So, we must take a closer look using the .isna().**
# In[26]:
starbucks.isna().sum()
# In[27]:
starbucks.dropna(axis=0, inplace=True)
starbucks_values.dropna(axis=0, inplace=True)

# In[28]:
dv_enc = LabelEncoder()
for i in starbucks_values.columns:
    if starbucks_values[i].dtype=='object':
        starbucks_values[i]=dv_enc.fit_transform(starbucks_values[i])

# In[29]:
starbucks['Beverage_category'].unique()
# In[ ]:


# In[30]:
starbucks_fig = plt.figure(figsize=(10,8)); ax = starbucks_fig.gca()
starbucks.hist(color='green', bins=30, ax=ax)
plt.suptitle('Starbucks Drinks: Nutruitional Values', y=1.03)
#st.pyplot_chart(starbucks_fig)
# In[ ]:
col1,col2 = st.columns(2,gap='small')
starbucks_statistics= col1.checkbox('Display the description of the Starbucks Drink dataset')
if starbucks_statistics==True:
    st.table(starbucks.describe())
    st.markdown('<p class="font_subtext">Table 1: Description of the statistics in the Starbucks Drink dataset', unsafe_allow_html=True)

starbucks_show = col2.checkbox('Display the Starbucks Drink dataframe')
if starbucks_show==True:
    st.table(starbucks)
    st.markdown('Starbucks Drink Dataframe', unsafe_allow_html=True)
    
st.write("Now, that you've gotten acquainted with the Starbucks Drinks dataset, it is time to explore some trends! Begin your journey by clicking on the Nutrition Correlation Map tab.")
# In[31]:
# **All columns will now be used. Now, I want to see what nutrients are correlated with one another.**
#
# In[13]:
sorted_starbucks = starbucks.sort_values(by='Beverage_category')
#sorted_starbucks.to_csv('sorted_starbucks.csv', index=False)
#sorted_starbucks.to_excel('sorted_starbucks.xlsx', index=False)
# In[14]:
tab0,tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(["EDA","Nutrient Correlation Map", "Classic Espresso Drinks", "Coffee", "Coffee Frappuccinos", "Creme Frappuccinos", "Lightly Blended Frappucinos", "Shaken Iced Drinks", "Signature Expressos", "Smoothies", "Tazo Teas", "Future Works"])
with tab0:
    st.subheader("Data Visualizations for the Most Popular Starbucks Drinks")
    st.subheader('Average Caffiene for Each Beverage Category')
    sns.set_theme(style="ticks")
    plot = sns.catplot(data=starbucks, kind="bar", x="Beverage_category", y="Caffeine (mg)", hue="Beverage_prep", palette="Greens", height= 13, aspect= 1.7)
    plot.set(xlabel='Beverage Category', ylabel='Average Caffeine')
    st.pyplot(plot)
    
    st.subheader("Pairplot of all Nutrients Based on Beverage Category")
    pairplot = sns.pairplot(data=starbucks, hue="Beverage_category")
    st.pyplot(pairplot)
    st.write("Although it is nice to see all nutrients that go into the build of a delicious Starbucks drink, let's take a closer look at the caffeine that is used.")
    
    st.subheader("Violin Plot of Caffeinated Beverages by Category and Prep")
    fig, ax = plt.subplots(figsize=(25, 7))
    sns.violinplot(data=starbucks,  x='Beverage_category', y='Caffeine (mg)', hue="Beverage_prep", palette='Greens', ax=ax)
    ax.set_title("Violin Plot of Caffeinated Beverages by Category and Prep")
    st.pyplot(fig)
    
    st.subheader("Interactive Stacked Bar Plot")
    st.write("Select and hover over your favorite beverage category to see how much caffeine is in your beverage!")
    click = alt.selection_point(encodings=['color'])
    beverage_hist = alt.Chart(starbucks).mark_bar().encode(
    x='Beverage_category',
    y='Caffeine (mg)',
    color=alt.condition(click, 'Caffeine (mg)', alt.value('lightgray'))
).add_params(
    click
)
    st.altair_chart(beverage_hist, use_container_width=True)
    st.write("Now that we have done a bit of exploration it is clear that caffeine is a major part of Starbucks' beverage make up. This was intresting, because even their smoothies and creme drinks contain caffeine. Next, we want to know how these drinks and their other nutrients combined can make an unhealthy or healthy drink.")

with tab1: 
    st.write("The United States Food and Drug Administration has recommendations of nutrients and how they should be consumed responsibly. From this, I began performing exploratory data analysis to see what how much of nutrients was included in these popular beverages, what type of beverage, and what nutrients were correlated with one another. Through my EDA efforts, I found Cholestoral & Calories have a correlation of 0.94, Sodium & Saturated Fat have a correlation of 0.92, Sugars & Calories have a correlation of 0.91, Sugars & Total Carbs have a correlation of  0.77 and Sodium & Transfat have a correlation of  0.71. All of these nutrients were at the top of the list for what the FDA recommends people consume in moderation. So, I decided to explore the relationship between all nutrients along with caffiene to see what people a consuming.")
    st.subheader('Starbucks Nutrient Correlation Map')
    plt.subplots(figsize=(25,20))
    heatmap = sns.heatmap(starbucks_values.corr(), cmap='Greens', annot=True)
    heatmap_fig = px.imshow(starbucks_values.corr(), text_auto=True, color_continuous_scale='Greens')
    st.plotly_chart(heatmap_fig)
    st.subheader("The highest correlated relationships:")
    st.write("Cholestoral & Calories 0.94")
    st.write("Sodium & Saturated Fat 0.92")
    st.write("Sugars & Calories 0.91")
    st.write("Sugars & Total Carbs 0.77")
    FDA_Recs = pd.read_csv("FDA_Recs.csv")
    st.write("Given the FDA Recommendations below, it is time to explore the highest correlated relationships in each Beverage Category. ")
    st.table(FDA_Recs)

    

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
    .add_params(single)
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
    .add_params(single)
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
    .add_params(single)
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
    .add_params(single)
)
with tab2:
    click = alt.selection_point(encodings=['color'])
    beverage_hist = alt.Chart(classic_espresso).mark_bar().encode(
    x='Beverage',
    y='Caffeine (mg)',
    color=alt.condition(click, 'Caffeine (mg)', alt.value('lightgray'))
    ).add_params(
    click
    ).properties(
      title='Caffeine in Espresso Drinks'  
    )
    title='Caffeine in Espresso Drinks'
    st.altair_chart(beverage_hist, use_container_width=True)
    
with tab2:
    esp_col1,esp_col2,esp_col3,esp_col4 = st.columns(4, gap='large')
esp1 = esp_col1.checkbox('Display the scatterplot of the First Highest Correlated Relationship for Espresso Drinks')
if esp1==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Calories and Cholestoral")
    st.altair_chart(esp_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Espresso Drinks with Calories and Cholestoral', unsafe_allow_html=True)

esp2 = esp_col2.checkbox('Display the scatterplot of the Second Highest Correlated Relationshipfor Espresso Drinks')
if esp2==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sodium and Saturated Fat")
    st.altair_chart(esp_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Espresso Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

esp3 = esp_col3.checkbox('Display the scatterplot of the Third Highest Correlated Relationship for Espresso Drinks')
if esp3==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sugars and Calories")
    st.altair_chart(esp_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Espresso Drinks with Sugars and Calories', unsafe_allow_html=True) 

esp4 = esp_col4.checkbox('Display the Scatterplot of the Fourth Highest Correlated Relationships for Espresso Drinks')
if esp4==True:
    st.subheader("Relationships between Caffienated Espresso Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(esp_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Espresso Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
    

# In[33]:
with tab3:
    st.header("Coffee")
    coffee_image = Image.open("starbucks-coffee-cup-is-seen-inside-a-starbucks-coffee-shop-news-photo-947784930-1536936500.jpg")
    coffee_image = coffee_image.resize((450, 400))
    st.image(coffee_image)
    coffee = sorted_starbucks[58:62]
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
 
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
    .add_params(single)
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
    .add_params(single)
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
    .add_params(single)
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
    .add_params(single)
)
with tab3:
    click = alt.selection_point(encodings=['color'])
    coffee_hist = alt.Chart(coffee).mark_bar().encode(
    x='Beverage',
    y='Caffeine (mg)',
    color=alt.condition(click, 'Caffeine (mg)', alt.value('lightgray'))
    ).add_params(
    click
    ).properties(
      title='Caffeine in Coffee Drinks'  
    )
    title='Caffeine in Coffee Drinks'
    st.altair_chart(coffee_hist, use_container_width=True)
    
with tab3:
    coffee_col1,coffee_col2, coffee_col3,coffee_col4 = st.columns(4, gap='large')
coffee1 = coffee_col1.checkbox('Display the scatterplot of the First Highest Correlated Relationship for Coffee Drinks')
if coffee1==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Calories and Cholestoral")
    st.altair_chart(coffee_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Coffee Drinks with Calories and Cholestoral', unsafe_allow_html=True)

coffee2 = coffee_col2.checkbox('Display the scatterplot of the Second Highest Correlated Relationshipfor Coffee Drinks')
if coffee2==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Sodium and Saturated Fat")
    st.altair_chart(coffee_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Coffee Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

coffee3 = coffee_col3.checkbox('Display the scatterplot of the Third Highest Correlated Relationship for Coffee Drinks')
if coffee3==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Sugars and Calories")
    st.altair_chart(coffee_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Coffee Drinks with Sugars and Calories', unsafe_allow_html=True) 

coffee4 = coffee_col4.checkbox('Display the Scatterplot of the Fourth Highest Correlated Relationships for Coffee Drinks')
if coffee4==True:
    st.subheader("Relationships between Caffienated Coffee Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(coffee_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Coffee Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
 
with tab4:
    st.header("Coffee Frappuccinos")
    coffeefrap_image = Image.open("GUEST_5befc930-6f2c-40d8-9d9e-513ad264cf0f.jpeg")
    coffeefrap_image = coffee_image.resize((450, 400))
    st.image(coffeefrap_image)
    frappuccino_coffee = sorted_starbucks[62:98]
    st.write("Hover over the circles to uncover where your favorite caffienated drinks lie on the scatterplot. First, all beverages will appear colored by type, once a circle is selected it will uncover a drink's nutrient values.")
 
    single = alt.selection_point(on='mouseover', nearest=True)
coffeefrap_cal_chol = (
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
    .add_params(single)
)
    
coffeefrap_sod_sat = (
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
    .add_params(single)
) 
coffeefrap_sug_cal = (
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
    .add_params(single)
)
    
coffeefrap_sug_carb = (
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
    .add_params(single)
)
with tab4:
    click = alt.selection_point(encodings=['color'])
    coffeefrap_hist = alt.Chart(frappuccino_coffee).mark_bar().encode(
    x='Beverage',
    y='Caffeine (mg)',
    color=alt.condition(click, 'Caffeine (mg)', alt.value('lightgray'))
    ).add_params(
    click
    ).properties(
      title='Caffeine in Coffee Frappuccino Drinks'  
    )
    title='Caffeine in Coffee Frappuccino Drinks'
    st.altair_chart(coffeefrap_hist, use_container_width=True)
    
with tab4:
    coffeefrap_col1,coffeefrap_col2, coffeefrap_col3,coffeefrap_col4 = st.columns(4, gap='large')
coffeefrap1 = coffeefrap_col1.checkbox('Display the scatterplot of the First Highest Correlated Relationship for Coffee Frappuccino Drinks')
if coffeefrap1==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccino Drinks with Calories and Cholestoral")
    st.altair_chart(coffeefrap_cal_chol, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 1: Relationships between Caffienated Coffee Frappuccino Drinks with Calories and Cholestoral', unsafe_allow_html=True)

coffeefrap2 = coffeefrap_col2.checkbox('Display the scatterplot of the Second Highest Correlated Relationship for Coffee Frappuccino Drinks')
if coffeefrap2==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccino Drinks with Sodium and Saturated Fat")
    st.altair_chart(coffeefrap_sod_sat, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 2: Caffienated Coffee Frappuccino Drinks with Sodium and Saturated Fat', unsafe_allow_html=True) 

coffeefrap3 = coffeefrap_col3.checkbox('Display the scatterplot of the Third Highest Correlated Relationship for Coffee Frappuccino Drinks')
if coffeefrap3==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccino Drinks with Sugars and Calories")
    st.altair_chart(coffeefrap_sug_cal, use_container_width=True)
    st.markdown('<p class="font_subtext">Figure 3: Caffienated Coffee Frappuccino Drinks with Sugars and Calories', unsafe_allow_html=True) 

coffeefrap4 = coffeefrap_col4.checkbox('Display the Scatterplot of the Fourth Highest Correlated Relationships for Coffee Frappuccino Drinks')
if coffeefrap4==True:
    st.subheader("Relationships between Caffienated Coffee Frappuccino Drinks with Sugars and Total Carbohydrates")
    st.altair_chart(coffeefrap_sug_carb, use_container_width=True) 
    st.markdown('<p class="font_subtext">Figure 4: Caffienated Coffee Frappuccino Drinks with Sugars and Total Carbohydrates', unsafe_allow_html=True) 
 
    
    
    
    
    
    
with tab11:
    st.subheader("Future Works")
    st.write("Upon finishing this web application, a Starbucks beverage builder will be available to all users")
    
     
   

# In[34]:
sns.scatterplot(data=starbucks, x="Calories", y="Cholesterol (mg)",hue="Beverage_category")
plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
plt.title("Relationship Between Calories & Cholesterol (mg) Based on Beverage Category")

# In[36]:

# In[41]:

# In[36]:
sns.scatterplot(data=starbucks, x="Sodium (mg)", y="Saturated Fat (g)",hue="Beverage_category")
plt.legend(bbox_to_anchor =(1.5, 1.15), ncol = 2)
plt.title("Relationship Between Sodium (mg) & Saturated Fat (g) Based on Beverage Category")
# In[39]:
sns.scatterplot(data=starbucks, x="Calories", y="Sugars (g)",hue="Beverage_category")
plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2) 
plt.title("Relationship Between Calories & Sugars (g) Based on Beverage Category")
# In[33]:
sns.scatterplot(data=starbucks, x="Sodium (mg)", y="Trans Fat (g) ",hue="Beverage_category")
plt.legend(bbox_to_anchor =(1.5, 1.15), ncol = 2)
plt.title("Relationship Between Sodium (mg) & Trans Fat (g) Based on Beverage Category")
# In[ ]:
sns.scatterplot(data=starbucks, x="Sodium (mg)", y="Trans Fat (g) ",hue="Beverage_category")
plt.legend(bbox_to_anchor =(1.5, 1.15), ncol = 2)
plt.title("Relationship Between Sodium (mg) & Trans Fat (g) Based on Beverage Category")
# In[21]:
# # <div class="alert alert-success"> Seperation of Beverage Categories </div>

# In[13]:
sorted_starbucks = starbucks.sort_values(by='Beverage_category')
#sorted_starbucks.to_csv('sorted_starbucks.csv', index=False)
#sorted_starbucks.to_excel('sorted_starbucks.xlsx', index=False)
# In[14]:

# In[15]:
classic_espresso.drop(['Beverage_category'], axis=1)
# In[17]:
sns.lmplot(x='Sugars (g)', y='Calories', hue='Caffeine (mg)', data=classic_espresso)
# In[ ]:
sns.lmplot(x='Cholesterol (mg)', y='Calories', hue='Caffeine (mg)', data=classic_espresso)


# In[ ]:


#beverage = classic_espresso['Beverage']
#beverage_choice = st.sidebar.selectbox('Choose your beverage:', beverage)


# In[ ]:

# In[ ]:


# In[ ]:
frappuccino_creme = sorted_starbucks[98:111]

# In[ ]:
frappuccino_lightblend = sorted_starbucks[111:123]

# In[ ]:
shaken_iced = sorted_starbucks[123:140]

# In[ ]:
signature_espresso = sorted_starbucks[140:180]

# In[ ]:
smoothies = sorted_starbucks[180:189]

# In[ ]:
tazo_teas = sorted_starbucks[189:241]
