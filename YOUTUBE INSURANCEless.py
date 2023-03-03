#!/usr/bin/env python
# coding: utf-8

# ## Hospital Insurance Charges

# In[1]:


#Import Libraries to read and manipulate data
import numpy as np
import pandas as pd

# Libaries for data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor 
from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR


# In[3]:


df= pd.read_csv("insurance.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[ ]:





# In[7]:


df.describe().T


# In[8]:


df["region"].unique()


# In[9]:


df.region


# In[10]:


df["sex"].unique() , df["smoker"].unique()


# In[11]:


cat_col = list(df.select_dtypes("object").columns)
cat_col

for column in cat_col:
    print(column,df[column].unique())
    print("__" * 10)


# In[12]:


cat_coll = list(df.select_dtypes("object").columns)
cat_coll


# In[13]:


categ_col = df.select_dtypes(include = ["object","category"]).columns

num_cols = [col for col in df.columns if col not in categ_col]
print(categ_col)


# In[14]:


print(num_cols)


# In[15]:


numm = list(df.select_dtypes(["int64" ,"float64"]).columns)
numm


# In[16]:


#label encoding

from sklearn.preprocessing import LabelEncoder,OneHotEncoder


le = LabelEncoder()
for i in categ_col:
    df[i] = le.fit_transform(df[i])


# In[17]:


df.head()


# In[18]:


categ_col


# In[19]:


cat_col = list(df.select_dtypes("object").columns)
cat_col

for column in cat_col:
    print(column,df[column].unique())
    print("__" * 10)


# In[20]:


#cat_column

for column in cat_col:
    print(column,df[column].unique())
    print("-" * 20)


# ### Modelling

# In[21]:


fig, ax = plt.subplots(figsize =  (14,7))

ax = sns.heatmap(df.corr(),annot = True);


# In[44]:


fig,ax = plt.subplots(figsize = (16,7))
ax = sns.heatmap(df.corr(),annot = True);
plt.savefig("htm")


# In[23]:


figsize = plt.figure(26)
sns.heatmap(df.corr(),annot = True);


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR


# In[26]:


#1.We encoded the categorical variables
#2 Splitting the data into features and target

x = df.drop("charges",axis = 1)
y = df["charges"]


# In[27]:


#3.Splitinto train(and validation) and test data x.shape,y.shape

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[28]:


#default regression score is r2 score,for classification is accuracy score


# In[55]:


#putting the models in a dictionary

models = {"Linear": LinearRegression(),
          "XGBoost": XGBRegressor(),
         "Catboost": CatBoostRegressor(),
         "DecisionTr": DecisionTreeRegressor(),
         "RandomFr": RandomForestRegressor(),
          
          "gradintb": GradientBoostingRegressor(),
          "KNN":KNeighborsRegressor(),
          "Ridge": Ridge(),
          "SVR": SVR(),
          
         }



#setting up a function to fit and score the model
def fit_and_score(models,X_train,X_test,y_train,y_test):
    """
    Fits and evaluates given machine learning models
    models ; a dictionary of different scikit learn machine learning models
    X_train : training data(no labels)
    y_train: training labels
    """
    
    #setting up a random seed
    np.random.seed(42)
    
    #making a dictionary to keep model scores
    
    model_scores = {}
    
    #looping through models
    
    for name,model in models.items():
        #fitting the model to the data
        
        model.fit(X_train,y_train)
        #evaluating the model and appending its score to model_scores
        
    
        
        model_scores[name] = model.score(X_test,y_test)
        
        #model_scores["model"].append(model_scores)
    return model_scores


# In[56]:


scores = fit_and_score(models= models,
                       X_train = X_train,
                       X_test = X_test,
                       y_train = y_train,
                       y_test = y_test )

scores


# In[58]:


model_compare = pd.DataFrame(scores, index = ["r2_score"])
model_compare.T.plot.bar();
plt.xticks(rotation = 45)
plt.savefig("mod2")


# In[32]:


xgb_model = XGBRegressor()
xgb_model.fit(X_train,y_train)


# In[33]:


xgb_model.score(X_test, y_test)


# In[34]:


#making predictions

y_pred = xgb_model.predict(X_test)
y_pred[:10]


# ------------------------------------------------

# R2        or coefficient of determination-compares my models prediction to the mean

# In[35]:


mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print(mse),print (rmse)


# In[36]:


mae = mean_absolute_error(y_test,y_pred)
mae


# In[37]:


df2 = pd.DataFrame(data = {"actual values": y_test,"predictions": y_pred})
df2


# ### HYPER PARAMETER TUNING

# In[46]:


fig, ax = plt.subplots(figsize = (14,7))

x = np.arange(0,len(df2),1)

ax.scatter(x,df2["actual values"],c ="b",label = "Actual Values")
ax.scatter(x,df2["predictions"],c = "r",label = "Predictions")

ax.legend(loc=(1,0.5));
plt.savefig("hpr")


# In[39]:


xgb_model.feature_importances_

#fii = pd.DataFrame(data ={"features":"features","feature_importance": importance_})
#,column= "importance")
#fii


# In[40]:


#creating a functiom to visualize important features


# In[41]:


def plot_importance(columns,importances, n = 20):
    df3 = (pd.DataFrame({"features": columns,
                     "feature_importances": importances}).sort_values("feature_importances",ascending = False).reset_index(drop = False))
                                                                      
    #plotting  the dataFrame
    fig,ax = plt.subplots(figsize = (10,8))
    ax = sns.barplot(x = "feature_importances", y = "features",data = df3[:n],orient = "h") 
                                                                      
    plt.ylabel("features") 
    plt.xlabel("feature_importance")                                                                  
                                                                      


# In[47]:



plot_importance(X_train.columns,xgb_model.feature_importances_)
plt.savefig("fea")


# In[ ]:




