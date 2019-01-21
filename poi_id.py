#!/usr/bin/python

import sys, os
import pickle
import pandas as pd
import numpy as np

my_path = os.path.dirname(__file__)

sys.path.append(os.path.join(my_path,"../tools/"))
sys.path.append(os.path.join(my_path,"../final_project/"))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 
'salary', 
'deferral_payments', 
'total_payments',
'bonus', 
'deferred_income', 
'total_stock_value', 
'expenses', 
'exercised_stock_options', 
'other', 
'long_term_incentive', 
'restricted_stock', 
'to_messages', 
'from_poi_to_this_person', 
'from_messages', 
'from_this_person_to_poi', 
'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open(os.path.join(my_path,"final_project_dataset.pkl"), "r") as data_file:
    data_dict = pickle.load(data_file)

enron_data = pd.DataFrame.from_dict(data_dict, orient='index')
enron_data = enron_data.replace('NaN', np.nan)

### Task 2: Remove outliers

enron_data.drop('TOTAL', axis=0, inplace=True)

### Task 3: Create new feature(s)

enron_data['from_poi_ratio'] = enron_data.from_poi_to_this_person / enron_data.to_messages
enron_data['to_poi_ratio'] = enron_data.from_this_person_to_poi / enron_data.from_messages

enron_data.fillna(value=0, inplace=True)

### Store to my_dataset for easy export below.

data_dict = enron_data.to_dict(orient='index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = Pipeline([('pre', MinMaxScaler()),
                ('sel', SelectKBest(f_classif, k=5)),
                ('clf', GaussianNB())])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)


