#Importing data
import pandas as pd
#import matplotlib as plt
import os
import numpy as np
#from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
print (os.getcwd())
#import xgboost as xgb
print (os.getcwd())
#Importng Data
d=pd.read_csv("TEDSA_2015_PUF.csv")
df=pd.DataFrame(data=d)
df_1 = df.copy(deep=True)

# #Deleting no response data points
df_1['RACE']=df_1['RACE'].replace(-9, np.nan)
df_1['MARSTAT']=df_1['MARSTAT'].replace(-9, np.nan)
df_1['EDUC']=df_1['EDUC'].replace(-9, np.nan)
df_1['EMPLOY']=df_1['EMPLOY'].replace(-9, np.nan)
df_1['LIVARAG']=df_1['LIVARAG'].replace(-9, np.nan)
df_1['NOPRIOR']=df_1['NOPRIOR'].replace(-9, np.nan)
df_1['SUB1']=df_1['SUB1'].replace(-9,np.nan)
df_1['GENDER']=df_1['GENDER'].replace(-9,np.nan)
df_1 = df_1.dropna(how='any',axis=0)
print(df_1.shape)

#Replace age with midpoint of age
df_1['AGE'] = df_1['AGE'].replace(2, (14+12)/2)
df_1['AGE'] = df_1['AGE'].replace(3, (17+15)/2)
df_1['AGE'] = df_1['AGE'].replace(4, (18+20)/2)
df_1['AGE'] = df_1['AGE'].replace(5, (21+24)/2)
df_1['AGE'] = df_1['AGE'].replace(6, (25+29)/2)
df_1['AGE'] = df_1['AGE'].replace(7, (30+34)/2)
df_1['AGE'] = df_1['AGE'].replace(8, (35+39)/2)
df_1['AGE'] = df_1['AGE'].replace(9,(40+44)/2)
df_1['AGE'] = df_1['AGE'].replace(10, (45+49)/2)
df_1['AGE'] = df_1['AGE'].replace(11, (50+54)/2)
df_1['AGE'] = df_1['AGE'].replace(12, (55+80)/2)

#data transformation response variable. grouping non-opioids into one class and opioids into another.
for i in range(2,6):
    df_1['SUB1']=df_1['SUB1'].replace(i,0)
for i in range(6,8):
    df_1['SUB1']=df_1['SUB1'].replace(i,1)
for i in range(8,21):
    df_1['SUB1']=df_1['SUB1'].replace(i,0)
df_1['SUB1'].value_counts()
#hist_SUB1=df_1['SUB1'].hist()

opioid_data= pd.concat([df_1['SUB1'],df_1['MARSTAT'],df_1['EDUC'],df_1['EMPLOY'],df_1['NOPRIOR'],
df_1['LIVARAG'],df_1['AGE'],df_1['GENDER']],axis=1)

#Coding Categorical Variables to strings for later use
mapping_NOPRIOR = {0: 'NO PRIOR TREATMENT EPISODES', 1 :'ONE PRIOR TREATMENT EPISODES',2: 'TWO PRIOR TREATMENT EPISODES',
3: 'THREE PRIOR TREATMENT EPISODES',4: 'FOUR PRIOR TREATMENT EPISODES',5: 'FIVE OR MORE PRIOR TREATMENT EPISODES'}
opioid_data['NOPRIOR']=opioid_data['NOPRIOR'].map(mapping_NOPRIOR)

mapping_MARSTAT={1 :'NEVER MARRIED',2: 'NOW MARRIED',3: 'SEPARATED',4: 'DIVORCED OR WIDOWED'}
opioid_data['MARSTAT']=opioid_data['MARSTAT'].map(mapping_MARSTAT)

mapping_EMPLOY={1 :'FULL TIME',2: 'PART TIME',3: 'UNEMPLOYED',4: 'NOT IN LABOR FORCE'}
opioid_data['EMPLOY']=opioid_data['EMPLOY'].map(mapping_EMPLOY)

mapping_LIVARAG={1 :'HOMELESS',2: 'DEPENDENT LIVING',3: 'INDEPENDENT LIVING'}
opioid_data['LIVARAG']=opioid_data['LIVARAG'].map(mapping_LIVARAG)

mapping_EDUC={1 :'EIGHT YEARS OR LESS',2: 'SOME HIGH SCHOOL',3: 'HIGH SCHOOL GRADUATE',4: 'SOME COLLEGE',5: 'COLLEGE GRAD OR HIGHER'}
opioid_data['EDUC']=opioid_data['EDUC'].map(mapping_EDUC)

mapping_GENDER={1 :'MALE',2: 'FEMALE'}
opioid_data['GENDER']=opioid_data['GENDER'].map(mapping_GENDER)

#One-hot encoding
X= pd.get_dummies(opioid_data, columns = ['MARSTAT','EMPLOY','NOPRIOR','LIVARAG','GENDER','EDUC'])
X= X.drop(['SUB1'], axis=1)
Y=opioid_data.SUB1

#Training Logistic Regression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3, random_state=1, stratify=Y)
clf=LogisticRegression(C=1.0, random_state=1,class_weight= 'balanced')

clf.fit(X_train,Y_train)

clf = clf.fit(X_train, Y_train)
print(clf.predict_proba(X_test.iloc[:1,:]))

user_input = {'EDUC' : 'EIGHT YEARS OR LESS', 'AGE' : 37, 'MARSTAT' : 'NOW MARRIED', 'EMPLOY' : 'PART TIME', 'NOPRIOR': 'FOUR PRIOR TREATMENT EPISODES', 'LIVARAG':'DEPENDENT LIVING', 'GENDER':'MALE'}
def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(25)
    # set the numerical input as they are
    enc_input[0] = data['AGE']
    #enc_input[1] = data['AGE']
    ##################### MARTSTAT#########################
    # get the array of MARSTAT categories
    MARSTAT = opioid_data.MARSTAT.unique()
    # redefine the the user input to match the column name
    redefinded_user_input = 'MARSTAT_'+data['MARSTAT']
    # search for the index in columns name list
    MARSTAT_column_index = X.columns.tolist().index(redefinded_user_input)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[MARSTAT_column_index] = 1

    ##################### EMPLOY ####################
    # get the array of fuel type
    EMPLOY= opioid_data.EMPLOY.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'EMPLOY_'+data['EMPLOY']
    # search for the index in columns name list
    EMPLOY_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[EMPLOY_column_index] = 1

    ##################### NOPRIOR ####################
    # get the array of fuel type
    NOPRIOR= opioid_data.NOPRIOR.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'NOPRIOR_' + data['NOPRIOR']
    # search for the index in columns name list
    NOPRIOR_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[NOPRIOR_column_index] = 1

    ##################### LIVARAG ####################
    # get the array of fuel type
    LIVARAG = opioid_data.LIVARAG.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'LIVARAG_' + data['LIVARAG']
    # search for the index in columns name list
    LIVARAG_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[LIVARAG_column_index] = 1

    ##################### GENDER ####################
    # get the array of fuel type
    GENDER = opioid_data.GENDER.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'GENDER_' + data['GENDER']
    # search for the index in columns name list
    GENDER_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[GENDER_column_index] = 1

    ##################### EDUC ####################
    # get the array of fuel type
    EDUC = opioid_data.EDUC.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'EDUC_' + data['EDUC']
    # search for the index in columns name list
    EDUC_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[EDUC_column_index] = 1
    return enc_input
print(input_to_one_hot(user_input))
opioid_pred = clf.predict_proba([input_to_one_hot(user_input)])
print(opioid_pred [0][1])
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
clf= joblib.load('model.pkl')