#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

###   Task 1: Select which features you'll use
### features_list is a list of strings, each of which is a feature name
### The first feature must be "poi"

# this list is used by featureFormat(), targetFeatureSplit() in
# this script for testing feature importances and parameter tuning

# the final list for features to be used by final classifier & Pickle dump will 
# be defined later based on classifier performance and feature importance

features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'expenses',
                 'director_fees',
                 'other',
                 'loan_advances',
                 'deferred_income',
                 'deferral_payments',
                 'total_payments',
                 'restricted_stock_deferred',
                 'exercised_stock_options',
                 'restricted_stock',
                 'total_stock_value',
                 'from_messages',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'from_poi_to_messages_ratio',
                 'to_poi_from_messages_ratio',
                 'shared_receipt_to_messages_ratio']

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "rb") as data_file:
  data_dict = pickle.load(data_file)

### Task 2: Remove outliers 
# Remove rows that have issues (no values, aggregate row, non-human entity)

#removing eugene lockhart row

data_dict.pop('LOCKHART EUGENE E', 0)

# removing aggregate row TOTAL

data_dict.pop('TOTAL', 0)

# removing non-person row THE TRAVEL AGENCY IN THE PARK

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Basic data cleaning
# First remove email addresses as this feature does not provide additional value to the analysis

for k in data_dict.keys():
  if 'email_address' in data_dict[k].keys():
    data_dict[k].pop('email_address')

# Check data for issues related to financial data provided

print("Problems with 'total_payments' : ", end='')
payment_financial_features = ['salary',
                              'bonus',
                              'long_term_incentive',
                              'expenses',
                              'director_fees',
                              'other',
                              'loan_advances',
                              'deferred_income',
                              'deferral_payments']
problem_entries = {}
# Iterate over each row, check sum of above features against total_payments,
#   rows with mismatch added to problem_entries

for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]
from pprint import pprint as pp
if len(problem_entries):
  print("found.")
  print("Records with problems related to 'total_payments' found: ")
  pp(problem_entries)
else:
  print("None.")
print('')

# The values for Belfer appear to be shifted and will be manually corrected here
# with values from reference enron61702insiderpay.pdf

belfer_corrected = {'bonus': 'NaN',
                    'deferral_payments': 0,                   
                    'deferred_income': -102500,               
                    'director_fees': 102500,                  
                    'exercised_stock_options': 0,             
                    'expenses': 3285,                         
                    'from_messages': 'NaN',
                    'from_poi_to_this_person': 'NaN',
                    'from_this_person_to_poi': 'NaN',
                    'loan_advances': 'NaN',
                    'long_term_incentive': 'NaN',
                    'other': 'NaN',
                    'poi': False,
                    'restricted_stock': 44093,               
                    'restricted_stock_deferred': -44093,     
                    'salary': 'NaN',
                    'shared_receipt_with_poi': 'NaN',
                    'to_messages': 'NaN',
                    'total_payments': 3285,                   
                    'total_stock_value': 0}                   

# Similar issue identified and corrected here for Bhatnagar

bhatnagar_corrected = {'bonus': 'NaN',
                       'deferral_payments': 'NaN',
                       'deferred_income': 'NaN',
                       'director_fees': 0,                    
                       'exercised_stock_options': 15456290,   
                       'expenses': 137864,                    
                       'from_messages': 29,
                       'from_poi_to_this_person': 0,
                       'from_this_person_to_poi': 1,
                       'loan_advances': 'NaN',
                       'long_term_incentive': 'NaN',
                       'other': 0,                            
                       'poi': False,
                       'restricted_stock': 2604490,           
                       'restricted_stock_deferred': -2604490, 
                       'salary': 'NaN',
                       'shared_receipt_with_poi': 463,
                       'to_messages': 523,
                       'total_payments': 137864,              
                       'total_stock_value': 15456290}         

# put corrected rows into dataset

data_dict['BELFER ROBERT'] = belfer_corrected
data_dict['BHATNAGAR SANJAY'] = bhatnagar_corrected

# Verify fixed entries for Belfer and Bhatnagar

print("Second check for problems with 'total_payments' : ", end='')
problem_entries = {}
for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]
if len(problem_entries):
  print("found.")
  print("Records with problems related to 'total_payments' found:")
  pp(problem_entries)
else:
  print("None.")

### Task 3: Create new feature(s) 

# code to create new features

for k in data_dict.keys():
  from_messages = True if \
    (data_dict[k]['from_messages'] != 'NaN') else False
  to_messages = True if \
    (data_dict[k]['to_messages'] != 'NaN') else False
  to_poi = True if \
    (data_dict[k]['from_this_person_to_poi'] != 'NaN') else  False
  from_poi = True if \
    (data_dict[k]['from_poi_to_this_person'] != 'NaN') else False
  shared_receipt = True if \
    (data_dict[k]['shared_receipt_with_poi'] != 'NaN') else False

  # ratio of emails sent to PoIs to emails sent generally:
  # to_poi_from_messages_ratio = from_this_person_to_poi / from_messages
  if to_poi and from_messages:
    data_dict[k]['to_poi_from_messages_ratio'] = \
       data_dict[k]['from_this_person_to_poi'] / data_dict[k]['from_messages']
  else:
    data_dict[k]['to_poi_from_messages_ratio'] = 'NaN'

  # ratio of emails received from PoIs to emails received generally:
  # from_poi_to_messages_ratio = from_poi_to_this_person / to_messages
  if from_poi and to_messages:
    data_dict[k]['from_poi_to_messages_ratio'] = \
          data_dict[k]['from_poi_to_this_person'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['from_poi_to_messages_ratio'] = 'NaN'
  
  # ratio of emails having shared recipt with PoIs to emails received generally:
  # shared_receipt_to_messages_ratio = shared_receipt_with_poi / to_messages
  if shared_receipt and to_messages:
    data_dict[k]['shared_receipt_to_messages_ratio'] = \
       data_dict[k]['shared_receipt_with_poi'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['shared_receipt_to_messages_ratio'] = 'NaN'

### Task 4: Try a variety of classifiers 

from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.model_selection   import StratifiedShuffleSplit

# Function definition for classifier testing, validation, evaluation
def classifier_test(clf, dataset, feature_list, folds = 1000):
  data = featureFormat(dataset, feature_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  cv = StratifiedShuffleSplit(n_splits=folds, random_state = 42)
  true_neg  = 0
  false_neg = 0
  true_pos  = 0
  false_pos = 0
  for train_idx, test_idx in cv.split(features, labels):
    features_train = []
    labels_train   = []
    features_test  = []
    labels_test    = []
    for ii in train_idx:
      features_train.append(features[ii])
      labels_train.append(labels[ii])
    for jj in test_idx:
      features_test.append(features[jj])
      labels_test.append(labels[jj])

    # fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
      if prediction == 0 and truth == 0:
        true_neg += 1
      elif prediction == 0 and truth == 1:
        false_neg += 1
      elif prediction == 1 and truth == 0:
        false_pos += 1
      elif prediction == 1 and truth == 1:
        true_pos += 1
      else:
        print("Warning: Found a predicted label not == 0 or 1.")
        print("All predictions should take value 0 or 1.")
        print("Evaluating performance for processed predictions:")
        break
  try:
    total_pred = true_neg + false_neg + false_pos + true_pos
    accuracy = 1.0 * (true_pos + true_neg) / total_pred
    precision = 1.0 * true_pos / (true_pos + false_pos)
    recall = 1.0 * true_pos / (true_pos + false_neg)
    f1 = 2.0 * true_pos / (2 * true_pos + false_pos + false_neg)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print("Testing", clf, "locally...")
    print("  Predictions: %d" % total_pred)
    print("  Accuracy: %.5f\n  Precision: %.5f  Recall: %.5f" % \
          (accuracy, precision, recall))
    print("  F1: %.5f  F2: %.5f" % (f1, f2), "\n")
  except:
    print("Performance calculations failed.")
    print("Precision or recall may be undefined (no true positives).")
    print("Or else you've forgotten 'poi' in param feature_list")


# Iteration over a list of classifiers

classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               GaussianNB()]

print("Trying several classifiers with default settings for comparison...\n")
for classifier in classifiers:
  classifier_test(classifier, data_dict, features_list)

### Checking feature importances

# Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# 4-C3 - by mutual_info_classif ('mutual information')

from sklearn.feature_selection import mutual_info_classif, SelectKBest
print("\nFeature importance by mutual_info_classif:")
print(" (\"mutual information\" with regard to target, 'poi')")

# sorting feature names by magnitude of mutual information with 'poi'
mutual_info = sorted(zip(list(mutual_info_classif(features, labels)),
                         features_list[1:]), reverse = True)
for i in range(len(mutual_info)):
  print(" ", i+1, "- '%s'" % mutual_info[i][1],
        "\n        %.5f"   % mutual_info[i][0])

print('')


### Task 5: Tune your classifier to achieve better than .3 precision and recall

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline        import Pipeline

# Using GridSearchCV with SelectKBest, DecisionTreeClassifier
# Using mutual information as feature selection metric

selector = SelectKBest(mutual_info_classif)

# Using information gain as splitting criterion

classifier = DecisionTreeClassifier(criterion = 'entropy')

tune_pipe = Pipeline(steps=[('skb', selector),
                            ('clf', classifier)])

# Optimizing number of features and minimum number of samples for splitting

grid_params = {'skb__k' : (3, 4, 5, 6, 7, 8, 9),
                'clf__min_samples_split' : (3, 4, 5, 6, 7, 8, 9)}

print("Trying GridSearchCV with")
pp(tune_pipe)
print("over parameters:")
pp(grid_params)

# Optimizing for maximized F1 in order to maximize precison and recall

grid = GridSearchCV(tune_pipe, grid_params, scoring = 'f1', cv = 10,
                      n_jobs = -1)
grid.fit(features, labels)

print("\nResulting 'best' parameters for maximizing 'f1':")
pp(grid.best_params_)

# sorting features by paired information gain scores

grid_ftrs = sorted(zip(list(grid.best_estimator_.named_steps['skb'].scores_),
                             features_list[1:]), reverse = True)

# creating featuer list to pass to k-fold testing function

best_features = ['poi']
print("\nFeatures used:")
for i in range(grid.best_params_['skb__k']):
  best_features.append(grid_ftrs[i][1])

  # displaying features for inspection of GridSearchCV's varying results
    
  print(" ", i+1, "- '%s'" % grid_ftrs[i][1],
        "\n        %.5f"   % grid_ftrs[i][0])
print('')

# Test of tuned parameters with 1000-fold cross validation
classifier_test(grid.best_estimator_.named_steps['clf'],data_dict,
                best_features)

### Final Algorithm
# Manual tuning and testing informed by previous results
manual_features = ['poi',
                   'expenses',
                   'bonus',
                   'other',
                   'to_poi_from_messages_ratio',
                   'shared_receipt_with_poi']
clf = DecisionTreeClassifier(criterion = 'entropy',
                             min_samples_split = 5)

print("Trying DecisionTreeClassifier with parameter settings and feature")
print("  selection based on 'best' of varying results from optimization...")
print("  (features *reliably* top-ranked by 'mutual information' with 'poi')")
print("Features used:")
pp(manual_features[1:])
classifier_test(clf, data_dict, manual_features)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

import tester
print("Testing final classifier via tester.py...")
tester.dump_classifier_and_data(clf, data_dict, manual_features)
tester.main()
