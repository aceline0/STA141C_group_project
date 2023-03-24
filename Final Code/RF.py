#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')


# In[82]:


df = pd.read_csv('schooldropout.csv', header=0, sep=';')


# In[83]:


mapping = {'Dropout':0, 'Enrolled':1, 'Graduate':1}
df['Target'] = df['Target'].replace(mapping)


# In[84]:


df


# In[85]:


df.info()


# In[86]:


df.describe()


# In[87]:


# Plotting a correlation matrix 
cm = plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(),annot=False, cmap='RdBu',vmax=.8, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show


# In[88]:


# Calculate the correlation matrix
corr_matrix = df.corr()

# Extract the correlation coefficients for the 'Target' column
target_corr = corr_matrix['Target']

# Print the correlation coefficients
print(target_corr)


# In[89]:


columns_to_keep = ['Application mode', 'Application order', 'Daytime/evening attendance', 'Previous qualification (grade)', 'Admission grade', 'Displaced', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Target'] 
new_df = df[columns_to_keep]
new_df.head()


# In[90]:


new_df['Target'].value_counts()


# In[92]:


import matplotlib.pyplot as plt

# Get the value counts of the 'Target' column
target_counts = new_df['Target'].value_counts()

# Create a pie chart
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%')

# Add a title
plt.title('How many dropouts & graduates are there in Target column')

# Add a legend with the target value mappings in the bottom left corner
plt.legend({'Graduate (1)', 'Dropout (0)'}, loc='upper right')

# Show the chart
plt.show()


# In[93]:


X = new_df.iloc[:,0:14]
Y = new_df.iloc[:,-1]
X


# In[94]:


Y


# In[95]:


from sklearn.model_selection import train_test_split

# Splitting data into 80% training and 20% testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# Checking data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ***Random Forest***

# In[100]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Create a pipeline with feature scaling and random forest classifier
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=10, random_state=0))

# Fit the pipeline to the training data
pipe.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = pipe.predict(X_test)

# Evaluate the accuracy of the model without cross-validation
print("Without CV: ", accuracy_score(Y_test, Y_pred))

# Evaluate the accuracy of the model with cross-validation
scores = cross_val_score(pipe, X_train, Y_train, cv=10)
print("With CV: ", scores.mean())

print("Precision Score: ", precision_score(Y_test, Y_pred,average='micro'))
print("Recall Score: ", recall_score(Y_test, Y_pred,average='micro'))
print("F1 Score: ", f1_score(Y_test, Y_pred,average='micro'))


# In[105]:


param_grid = {
    'bootstrap': [False,True],
    'max_depth': [5, 10, 15, 20],
    'max_features': [4, 5, 6, None],
    'min_samples_split': [2, 10, 12],
    'n_estimators': [100, 200, 300]
}

rfc = RandomForestClassifier()

clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

clf.fit(X_train, Y_train)
best_rfc = clf.best_estimator_

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: ", accuracy)
print("Best Hyperparameters: ", clf.best_params_)

# Evaluate cross-validation score for the best model
cv_score = cross_val_score(best_rfc, X_train, Y_train, cv=10)
print("Cross-validation score: ", cv_score.mean())


# In[110]:


print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))


# 
