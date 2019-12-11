#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import os


# In[37]:


import sklearn as sklearn
from datetime import *


# In[38]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.feature_selection import RFE


# In[39]:


#from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
#from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report


# In[40]:


from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
#from modeldb.sklearn_native.ModelDbSyncer import *
#from modeldb.sklearn_native import SyncableMetrics


# In[41]:


Scoring  = pd.read_csv("ScoringDataProfiled.csv")
Training = pd.read_csv("CustomerChurnDataProfiled.csv")


# In[43]:


Training.columns


# #### Dropping the Columns based on the data from Data Profiling of Data Explorer

# In[44]:


Training.drop([ 'FIRST_NAME','INACTIVE_CUST_FLAG','LAST_NAME','NUM_DEPOSIT_TRX', 'NUM_OF_AVG_RESOLUTION','NUM_OF_GOOD_RESOLUTION','NUM_OF_POOR_RESOLUTION','create_date','product_type'],axis=1,inplace=True)


# In[45]:


Scoring.columns


# In[46]:


Scoring.drop([ 'FIRST_NAME','INACTIVE_CUST_FLAG','LAST_NAME','NUM_DEPOSIT_TRX', 'NUM_OF_AVG_RESOLUTION','NUM_OF_GOOD_RESOLUTION','NUM_OF_POOR_RESOLUTION','create_date','product_type'],axis=1,inplace=True)


# In[47]:


from sklearn.model_selection import train_test_split


# In[51]:


print (Training.describe())


# In[52]:


print (Training.dtypes)


# In[53]:


Training['interest_rate_low'] = np.where((Training['Interest_rate'] <= 16.0),1,0)
Training['interest_rate_high'] = np.where((Training['Interest_rate'] > 16.0),1,0)
Training['credit_limit_low'] = np.where((Training['Credit_card_limit'] <= 75000.0),1,0)
Training['credit_limit_high'] = np.where((Training['Credit_card_limit'] > 75000.0),1,0)
Training['Nuclear_Family'] = np.where((Training['AVG_FAMILY_SIZE'] <= 4),1,0)
Training['Joint_Family'] = np.where((Training['AVG_FAMILY_SIZE'] > 4),1,0)


# In[54]:


Training.head(5)


# In[55]:


Training['age_20_30'] = np.where((Training['AGE'] >= 20) & (Training['AGE'] <= 30),1,0)
Training['age_30_40'] = np.where((Training['AGE'] > 30) & (Training['AGE'] <= 40),1,0)
Training['age_40_50'] = np.where((Training['AGE'] > 40) & (Training['AGE'] <= 50),1,0)
Training['age_50_60'] = np.where((Training['AGE'] > 50) & (Training['AGE'] <= 60),1,0)
Training['age_60_70'] = np.where((Training['AGE'] > 60) & (Training['AGE'] <= 70),1,0)


# In[56]:


X = Training.filter(['customer_id','GENDER','INCOME_LEVEL','MARITAL_STATUS','sentiment_polarity'],axis=1)
def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x],prefix = x,dummy_na = False)
        df = df.drop(x,1)
        df = pd.concat([df,dummies],axis=1)
    return df

catg_list = ['GENDER','INCOME_LEVEL','MARITAL_STATUS','sentiment_polarity']
#for col_name in X.columns:
 #   if X[col_name].dtypes == 'object':
        #catg_list.append(col_name)

print(catg_list)
X = dummy_df(X,catg_list)
print (X)
X.columns


# In[57]:


Training = pd.merge(Training,X,on='customer_id',how='left')
Training.columns


# In[58]:


df_main = Training.drop(['GENDER','AGE', 'AVG_FAMILY_SIZE', 'Credit_card_limit', 'DEPOSIT_AMT', 'INCOME_LEVEL', 'Interest_rate',
        'MARITAL_STATUS','OCCUPATION','sentiment_polarity','_id','WDWL_AMT','product_id','sentiment_score'],axis =1)


# In[59]:


Y_x = df_main['CHURN_Y_N']
X_1 = df_main.drop(['customer_id','CHURN_Y_N'],axis=1)
from sklearn.feature_selection import chi2,f_classif
labels = Y_x.values
features = X_1.values
chi2_, pval = chi2(features, labels)
print(chi2_)
print(pval)
#Selecting the variables whose Pval is lesser than 0.3
feat_list = list(X_1.columns)

len(feat_list)
len(chi2_)

serv_var_imp_x = pd.DataFrame(
    {'Variable': feat_list,
     'Scores': chi2_.tolist(),
     'Pval': pval.tolist()                 
    })


# In[60]:


sel_features = serv_var_imp_x[serv_var_imp_x.Pval <= 0.5]
features = sel_features['Variable'].tolist()
features.append('customer_id')
features.append('CHURN_Y_N')
print (features)


# In[61]:


df = df_main[features]
print (df.columns)
print (df.shape)


# In[63]:


df1 = df.copy()


# In[64]:


df1 = df1.drop(['customer_id'],axis=1)


# In[65]:


df1.columns


# In[66]:


all_inputs = df1[['ACCOUNT_BALANCE_LOW', 'NUM_OF_COMPLAINTS', 'NUM_OF_ENQUIRIES',
       'NUM_OF_PROD_SUPRT', 'interest_rate_low', 'interest_rate_high',
       'credit_limit_low', 'credit_limit_high', 'Nuclear_Family',
       'Joint_Family', 'age_20_30', 'age_30_40', 'age_40_50', 'age_50_60',
       'age_60_70', 'GENDER_Female', 'GENDER_Male', 'INCOME_LEVEL_High',
       'INCOME_LEVEL_Low', 'INCOME_LEVEL_Medium', 'MARITAL_STATUS_Single',
       'sentiment_polarity_negative', 'sentiment_polarity_neutral']].values
all_classes = df1['CHURN_Y_N'].values


# In[67]:


# Creating a new project
'''

name = "Customer Churn Use Case"
author = "Test_user_Churn"
description = "Classification Technique"
syncer_obj = Syncer(
    NewOrExistingProject(name, author, description),
    NewOrExistingExperiment("customerChurnUseCase test user churn", "churnDesc test user churn"),
    NewExperimentRun("churn use case test user churn"))
'''


# In[68]:


#syncer_obj.add_tag(all_inputs, "data to input into model")


# In[69]:


# Make sure that you don't mix up the order of the entries
# all_inputs[5] inputs should correspond to the class in all_classes[5]
'''
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = cross_validation.train_test_split_sync(
    all_inputs, all_classes, train_size=0.80, random_state=1)
'''
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(
    all_inputs, all_classes, train_size=0.80, random_state=1)


# In[70]:


classifier = RandomForestClassifier()


feature_columns1 = df1.columns


# In[73]:


#classifier.fit_sync(training_inputs, training_classes)
classifier.fit(training_inputs, training_classes)


# In[75]:


# Validate the classifier on the testing set using classification accuracy
# decision_tree_classifier.score(testing_inputs, testing_classes)

# NOTE: score is equivalent to sklearn.metrics.accuracy_score.
'''
SyncableMetrics.compute_metrics(
    classifier, accuracy_score, testing_classes,
    classifier.predict(testing_inputs), training_inputs, "", "")

# cross_val_score returns a list of the scores, which we can visualize
# to get a reasonable estimate of our classifier's performance
cv_scores = cross_validation.cross_val_score_sync(
    classifier, all_inputs, all_classes, cv=10)
    '''


# In[76]:


#syncer_obj.sync()


# In[ ]:





# In[77]:


pipeline = Pipeline([("classifier", classifier)])


# In[78]:


#pickle pipeline
pipeline_pickle_file = "customer_churn_pipeline_test.pkl"
joblib.dump(pipeline, pipeline_pickle_file, compress = 9)


# In[79]:


pipeline.fit(df1[['ACCOUNT_BALANCE_LOW', 'NUM_OF_COMPLAINTS', 'NUM_OF_ENQUIRIES',
       'NUM_OF_PROD_SUPRT', 'interest_rate_low', 'interest_rate_high',
       'credit_limit_low', 'credit_limit_high', 'Nuclear_Family',
       'Joint_Family', 'age_20_30', 'age_30_40', 'age_40_50', 'age_50_60',
       'age_60_70', 'GENDER_Female', 'GENDER_Male', 'INCOME_LEVEL_High',
       'INCOME_LEVEL_Low', 'INCOME_LEVEL_Medium', 'MARITAL_STATUS_Single',
       'sentiment_polarity_negative', 'sentiment_polarity_neutral']],df1['CHURN_Y_N'] )


# In[80]:


df1.head()

