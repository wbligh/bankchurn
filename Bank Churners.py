

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



'''
        
            Prliminary EDA
      
      1) Data prepocessing
      2) Univariate analysis & distribution review
      3) Bivariate & Correlation analysis with target & Feature importance
      4) Models
            RF, SVM, XGBoost, Ridge, Lasso, KNN
            
       
        
'''
#%%
#   1)  Data Preprocessing

#       Remove unwanted columns from dataset


df = pd.read_csv('Data/BankChurners.csv')

dropcols =['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

df = df.drop(columns = dropcols)
df_preview = df.head(10)
#       First let's check for NULL values

df.isnull().sum()




# %%    

#    2) Univariate analysis & distribution review


# We can visualise our data first in order to understand if further preprocessing is required

#   First let's visualise our target variable to understand how this is distributed
plt.figure()
sns.countplot(df['Attrition_Flag'])

#   We can see that the dataset is quite unbalanced and there is a significant minority class (Attrited Customer)



#   Have a quick glance at some of the main categorical variables to understand balance in the dataset
fig, ax = plt.subplots(3,2, figsize= (8,12))
sns.countplot(data = df, x = 'Income_Category',order = ['Less than $40K','$40K - $60K','$60K - $80K','$80K - $120K','$120K +','Unknown'], ax = ax[0,0],hue = "Attrition_Flag")
sns.countplot(data = df, x = 'Gender', ax = ax[0,1],hue = "Attrition_Flag")
sns.countplot(data = df, x = 'Education_Level', ax = ax[1,0],hue = "Attrition_Flag")
sns.countplot(data = df, x = 'Marital_Status', ax = ax[1,1],hue = "Attrition_Flag")
sns.countplot(data = df, x = 'Education_Level', ax = ax[2,0],hue = "Attrition_Flag")
sns.countplot(data = df, x = 'Card_Category', ax = ax[2,1],hue = "Attrition_Flag")
fig.show()


fig, ax = plt.subplots(3,2, figsize= (8,12))
sns.histplot(data = df, x = 'Customer_Age', ax = ax[0,0],hue = "Attrition_Flag")
sns.histplot(data = df,x = 'Months_on_book', ax = ax[0,1],hue = "Attrition_Flag")
sns.histplot(data = df,x = 'Credit_Limit', ax = ax[1,0],hue = "Attrition_Flag")
sns.histplot(data = df,x = 'Total_Revolving_Bal', ax = ax[1,1],hue = "Attrition_Flag")
sns.histplot(data = df,x = 'Total_Trans_Amt', ax = ax[2,0],hue = "Attrition_Flag")
sns.histplot(data = df,x ='Avg_Utilization_Ratio', ax = ax[2,1],hue = "Attrition_Flag")
fig.show()

# %%

numerics = ['int16','int32','int64','float16','float32','float64']
df_num_corr = df.select_dtypes(include = numerics).corr()

sns.heatmap(df_num_corr)


# %%


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#   Let's scale our data, and encode our non-numeric data, ready for modelling.

df.select_dtypes(include ='object').nunique()

df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer':0, 'Attrited Customer': 1})


#   There aren't too many categories for each variable so I feel comfortable to one hot encode the lot.
cat_cols = list(df.select_dtypes(include= 'object').columns)

for col in cat_cols:
    df = pd.concat([df.drop(columns = col), pd.get_dummies(df[col],prefix = str(col))], axis = 1)

# We can then scale our data before putting it into the 

Y = df['Attrition_Flag']
X = df.drop(columns = 'Attrition_Flag')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()

X_train[(X_train.select_dtypes(include = numerics).columns)] = scaler.fit_transform(X_train[(X_train.select_dtypes(include = numerics).columns)] )
X_test[(X_test.select_dtypes(include = numerics).columns)] = scaler.transform(X_test[(X_test.select_dtypes(include = numerics).columns)] )


# We also want to resample the training data so that our model has more balanced view of the data

sampler =  SMOTE(random_state =0)

X_train_res, y_train_res = sampler.fit_resample(X_train, y_train.ravel())

y_train_res = pd.Series(y_train_res)





#%%    

# We can now spin up the models and run 10-fold CV to understand how the models perform

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

models = [LogisticRegression, RandomForestClassifier, KNeighborsClassifier]

for model in models:
    clf = model()
    clf.fit(X_train_res, y_train_res)
    print(str(model.__name__), str(clf.score(X_test, y_test)))
    




for model in models:
    clf = model()
    clf.fit(X_train, y_train)
    print(str(model.__name__), str(clf.score(X_test, y_test)))





