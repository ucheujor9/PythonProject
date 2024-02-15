# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

#import the dataset
fitness = pd.read_csv("C:/Users/user/Downloads/fitness_class_2212.csv")

#Checks
print(fitness.head())
print(fitness.info())
print(fitness.describe())
print(fitness.shape)
print(fitness.columns)
print(fitness.dtypes)
print(fitness.isna().sum())
print(fitness.duplicated().sum())

#clean the data
print(fitness['months_as_member'].min())

fitness['weight'].round(2)
print(fitness['weight'].min())
fitness.loc[fitness['weight']>40, 'weight']=40
print(fitness['weight'].min())
print(fitness['weight'].mean())
fitness['weight'].fillna(fitness['weight'].mean(), inplace = True)

fitness['days_before'] = fitness['days_before'].str.replace(" days", "")
fitness['days_before'] = fitness['days_before'].astype(int)
print(fitness['days_before'].min())

print(fitness['day_of_week'].unique())
fitness['day_of_week'] = fitness['day_of_week'].str.strip('.')
fitness['day_of_week'].replace({'Monday':'Mon', 'Tuesday':'Tue', 'Wednesday':'Wed', 'Thursday':'Thu', 'Friday':'Fri', 'Saturday':'Sat', 'Sunday':'Sun'}, inplace=True)
print(fitness['day_of_week'].unique())

print(fitness['time'].unique())

print(fitness['category'].unique())
print(sum(fitness['category']=='-'))
fitness['category'].replace('-', 'Unknown', inplace=True)
print(fitness['category'].unique())

print(fitness['attended'].unique())

fitness[['day_of_week', 'time', 'category']]=fitness[['day_of_week', 'time', 'category']].astype('category')
print(fitness.dtypes)
print(fitness.isna().sum())
print(fitness.describe())
print(fitness.shape)
print(fitness.info())

#Visualize the number of attendance against the duration of membership
#This shows that people who recently joined the program are highest in attendance.
sns.countplot(x='attended', data = fitness)
plt.xlabel('Category of Attendance')
plt.ylabel('Count')
plt.title("Graph 1 - The Count of Attendance")
plt.show()

#Visualize the distribution of Number of months of members
sns.distplot(fitness['months_as_member'], kde = True)
plt.xlabel('Duration of membership (in months)')
plt.ylabel('Count')
plt.title("Graph 2A - The Distribution of Number of months as a member")
plt.show()

sns.histplot(fitness['months_as_member'])
plt.xlabel('Duration of membership (in months)')
plt.ylabel('Count')
plt.title("Graph 2B - The Distribution of Number of months as a member")
plt.show()

#Visualize the relationship between attendance and months as member
sns.boxplot(x='attended', y='months_as_member', data=fitness)
plt.title('Graph 3 - Relationship B/W Attendance & Membership duration')
plt.xlabel('Attendance')
plt.ylabel('Duration of membership (in months)')
plt.show()

#Normalize the column(months_as_member)
fitness['log_months_as_member']=np.log(fitness['months_as_member'])

#Visualize log-transformed data
sns.boxplot(x='attended', y='log_months_as_member', data=fitness)
plt.title('Graph 3B - Relationship B/W Attendance & Membership duration')
plt.xlabel('Attendance')
plt.ylabel('Duration of membership (in months)')
plt.show()

sns.histplot(fitness['log_months_as_member'])
plt.xlabel('Log-transformed duration of membership (in months)')
plt.ylabel('Count')
plt.title("Graph 3C - The log-dist. of No of membership duration")
plt.show()

#filter data to be used for prediction
fitness_for_use = fitness.drop(['booking_id', 'months_as_member'], axis = 1)
print(fitness_for_use.shape)

#feature engineering
le = LabelEncoder()
fitness_for_use['time_enc']=le.fit_transform(fitness_for_use['time'])
fitness_for_use['category_enc']=le.fit_transform(fitness_for_use['category'])
fitness_for_use['day_of_week_enc']=le.fit_transform(fitness_for_use['day_of_week'])
print(fitness_for_use.columns)

X = fitness_for_use.drop(['attended', 'time', 'category', 'day_of_week'], axis = 1)
y = fitness_for_use['attended']

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=2)

#Instantiate models
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_score = accuracy_score(y_test, knn_pred)
print(knn_score)

con_mat = confusion_matrix(y_test, knn_pred)
df_con_mat = pd.DataFrame(con_mat)
print(df_con_mat)

#data=np.array([["True Negateve", "False Positive"], ["False Negative", "True Positive"]])

sns.heatmap(df_con_mat, annot = True, fmt= 'd', cmap='Blues', linewidths=0.5)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Map 1 - Attendance KNN Model Result')
plt.show()


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_score = accuracy_score(y_test, dt_pred)
print(dt_score)

con_mat2 = confusion_matrix(y_test, dt_pred)
df_con_mat2 = pd.DataFrame(con_mat2)
print(df_con_mat2)

sns.heatmap(df_con_mat2, annot=True, fmt= 'd', cmap='Oranges', linewidths=0.5)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Map 2 - Attendance DT Model Result')
plt.show()

#Using the KNN model would give us a more accurate prediction with 230 True Positives.