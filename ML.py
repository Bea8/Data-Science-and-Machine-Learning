#Logistic Regression with Python
#I have tried to predict a classification: will graduate or will drop out.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
from sklearn.preprocessing import StandardScaler



data_score = pd.read_csv("score_board.csv")

#Exploratory Data Analysis: patterns and visualisaions

#Accepted
sns.set_style('whitegrid')
sns.countplot(x='accepted',data=data_score,palette='RdBu_r')



data_grad = pd.read_csv("graduates.csv")

#Graduated
sns.set_style('whitegrid')
sns.countplot(x='graduated',data=data_grad,palette='RdBu_r')



#combine data sets into one full data set
train_unfiltered = pd.merge(data_score, data_grad, how='left')
train = train_unfiltered.drop(train_unfiltered[train_unfiltered["accepted"] == False].index, inplace=False).reset_index(drop=True)
train.head()



#We can use seaborn to create a simple heatmap to see if/where we are missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')



#There are no missing values in the train data set

#Features: id, year are unlikely to yield any useful information and therefore will be dropped entirely. Feature accepted can be omitted as well.

#train dataset column types information
train.info()



#Data Cleaning: dropping columns
train = train.drop(['id','year','accepted'], axis = 1, inplace=False)
train.head()



#Converting booleans to numerical data
train['graduated'] = train['graduated'].astype('int')

train.head()



#Calculating avarage values
#Analysing how having a score above avarage affects a student's potential for graduation
def cat(*args):
    for i in args:
        mean = train[i].mean()
        df = (train.loc[(train['graduated'] == 1) & (train[i] > mean)])
        above_avg = df[i].count()

        total_grads = sum(train['graduated']==1)
        percent = round((above_avg/total_grads)*100)
        print(str(percent) +  "% " + "of gradutes had a " + str(i) + " score " + "higher than avarage of " + str(round(mean, 2)))

cat('gpa', 'maths_exam', 'art_exam', 'language_exam', 'essay_score', 'interview_score')



#Almost 80% of graduates had essay and interview scores above avarage. We can use these finding to engineer new features.

# Proportion of graduates with essay score above avarage(True)
es_mean = (train['essay_score'].mean())
esm_above = (train['essay_score'] > es_mean)

df = pd.concat([train['graduated'], esm_above], axis=1)
sns.set_style('whitegrid')
sns.countplot(x='graduated',hue='essay_score',data=df,palette='RdBu_r')



# Proportion of graduates with interview score above avarage(True)
is_mean = (train['interview_score'].mean())
ism_above = (train['interview_score'] > is_mean)

df = pd.concat([train['graduated'], ism_above], axis=1)
sns.set_style('whitegrid')
sns.countplot(x='graduated',hue='interview_score',data=df,palette='RdBu_r')



#More visualisations
g = sns.FacetGrid(train, col='graduated')
g.map(plt.hist, 'score', bins=3)



sns.set_style('whitegrid')
sns.countplot(x='graduated',hue='social_activity',data=train,palette='rainbow')



#feature engineering: es_above_avg for student with essey score above avarage
es_mean = (train['essay_score'].mean())
print(es_mean)
train['es_above_avg'] = train['essay_score'].map(lambda x: 0 if x < es_mean else 1)

train.head()



#feature engineering: is_above_avg for student with interview score above avarage
is_mean = (train['interview_score'].mean())
print(is_mean)
train['is_above_avg'] = train['interview_score'].map(lambda x: 0 if x < is_mean else 1)
train.head(10)



# Split into train and test dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('graduated',axis=1),
                                                    train['graduated'], test_size=0.20,
                                                    random_state=101)



#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression(solver='lbfgs', max_iter=200)
logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)
X_test.head()



predictions[0:100]



from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))



print(classification_report(y_test,predictions))
