#!/usr/bin/env python
# coding: utf-8

# In[21]:


import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# # <div class="alert alert-success"> **Starbucks Drinks: Are they good for you?** ☕️</div>

# # Step 1

# # <div class="alert alert-success"> EDA </div>

# **Before diving into the project, we want to load the starbucks drink menu into a pandas dataframe and then use .head() to view the columns we are working with.**

# In[22]:

st.title("Starbucks Drinks: Are they good for you?")
starbucks = pd.read_csv("/Users/sarahbradford/Downloads/starbucks_drinks.csv")
starbucks.head()
starbucks_values = pd.read_csv("/Users/sarahbradford/Downloads/starbucks_drinks.csv")


# In[23]:


starbucks.info()


# In[24]:


starbucks = starbucks.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)
starbucks_values = starbucks_values.drop(['Vitamin A (% DV) ', 'Vitamin C (% DV)', 'Calcium (% DV) ', 'Iron (% DV) '],axis=1)


# **By the looks of it we have our traditional nutritional label in a dataframe with the most popular drinks on the menu. Next, we will use .describe() to see if there are any missing values present in the dataset and need some data cleaning.**

# In[ ]:





# In[25]:


starbucks.describe()


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
plt.tight_layout()


# In[ ]:





# In[31]:


starbucks.describe()


# **All columns will now be used. Now, I want to see what nutrients are correlated with one another.**

# In[32]:


plt.subplots(figsize=(25,20))
sns.heatmap(starbucks_values.corr(), cmap='Greens', annot=True)
plt.show()


# In[33]:


starbucks.info()


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

# In[34]:


sns.scatterplot(data=starbucks, x="Calories", y="Cholesterol (mg)",hue="Beverage_category")
plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
plt.title("Relationship Between Calories & Cholesterol (mg) Based on Beverage Category")


# In[35]:


interval = alt.selection_interval(encodings=['Beverage_category'])

alt.Chart(starbucks).mark_point().encode(
    x='Calories:Q',
    y='Cholesterol (mg):Q',
    color=alt.condition(interval, 'Beverage_category', alt.value('lightgreen'))
).interactive().add_params(
    interval
)


# In[36]:


single = alt.selection_point(on='mouseover', nearest=True)
alt.Chart(starbucks).mark_circle(size=200).encode(
    x='Calories:Q',
    y='Cholesterol (mg):Q',
    color=alt.condition(single, 'Beverage_category', alt.value('lightgray'))
).interactive().add_params(
    single
)


# In[41]:


interval = alt.selection_interval(encodings=['x'])

alt.Chart(starbucks).mark_point().encode(
    x='Calories:Q',
    y='Cholesterol (mg):Q',
    color=alt.condition(interval, 'Beverage_category', alt.value('lightgray'))
).add_params(
    interval
)


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


classic_espresso = sorted_starbucks[0:58]
classic_espresso_fig = plt.figure(figsize=(10,8)); ax = classic_espresso_fig.gca()
classic_espresso.hist(color='green', bins=30, ax=ax)
plt.suptitle('Classic Espresso: Nutruitional Values', y=1.03)
plt.tight_layout()


# In[15]:


classic_espresso.drop(['Beverage_category'], axis=1)


# In[17]:


sns.lmplot(x='Sugars (g)', y='Calories', hue='Caffeine (mg)', data=classic_espresso)


# In[ ]:


sns.lmplot(x='Cholesterol (mg)', y='Calories', hue='Caffeine (mg)', data=classic_espresso)


# In[ ]:


beverage = classic_espresso['Beverage']
beverage_choice = st.sidebar.selectbox('Choose your beverage:', beverage)


# In[ ]:


coffee = sorted_starbucks[58:62]


# In[ ]:


frappuccino_coffee = sorted_starbucks[62:98]


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


# starbucks_fig = plt.figure(figsize=(10,8)); ax = starbucks_fig.gca()
# starbucks.hist(color='green', bins=30, ax=ax)
# plt.suptitle('Starbucks Drinks: Nutruitional Values', y=1.03)
# plt.tight_layout()

# In[ ]:





# In[ ]:





# <div class="alert alert-success"> According to the FDA, the daily recommended values for Cholesterol is 300mg, Sodium is 2300mg, Sugars are 50g, Carbohydrates are 275g, Saturated fat is 20g, and Fat is 70g. Additionally, the FDA recommends staying under 400mg of Caffiene a day. These values should not be exceeded, if they are a person is at a higher risk of health issues.  </div>

# In[ ]:


sns.boxplot(x='Cholesterol (mg)', y='Calories', hue='Caffeine (mg)', data=starbucks)


# In[ ]:


sns.lmplot(x='Cholesterol (mg)', y='Calories', hue='Caffeine (mg)', data=starbucks)


# In[ ]:


starbucks['Beverage_category'].unique()


# In[ ]:


caffeine = starbucks[['Beverage_category','Caffeine (mg)']]
caffeine.columns = ['Drinks', 'Caffeine (mg)']


# In[ ]:


sns.countplot(y='Drinks', data=caffeine)


# In[ ]:


#drinks_ = starbucks.groupby("Beverage_category")


# In[ ]:
st.title("What are you putting in your body when you consume starbucks?")
starbucks_data=[]
choice1 = x_variable = st.sidebar.selectbox("Select A Drink", starbucks.columns['Beverage'])
choice2 = y_variable = st.sidebar.selectbox("Select A Nutrient", starbucks.columns)
st.bar_chart(data = starbucks, x= choice1, y = choice2)

sorted_starbucks = starbucks.sort_values(by='Beverage_category')
classic_espresso = sorted_starbucks[0:58]
classic_espresso = classic_espresso.drop(['Beverage_category'], axis=1)
#classic_espresso = pd.DataFrame(classic_espresso)
beverage = classic_espresso['Beverage'].tolist()
#beverage = 
beverage_choice = st.selectbox('Choose your beverage:', beverage)


