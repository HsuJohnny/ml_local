import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import feature_process_helper
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv')
del y_train['id']
X_train, X_test = feature_process_helper.dates(X_train, X_test)
X_train, X_test = feature_process_helper.dates2(X_train, X_test)
X_train, X_test = feature_process_helper.construction(X_train, X_test)
X_train, X_test = feature_process_helper.bools(X_train, X_test)
X_train, X_test = feature_process_helper.locs(X_train, X_test)
X_train['population'] = np.log(X_train['population'])
X_test['population'] = np.log(X_test['population'])
X_train, X_test = feature_process_helper.removal2(X_train, X_test)
X_train, X_test = feature_process_helper.small_n2(X_train, X_test)
X_train, X_test = feature_process_helper.lda(X_train, X_test, y_train, cols = ['gps_height', 'latitude', 'longitude'])
X_train, X_test = feature_process_helper.dummies(X_train, X_test)


# different classifier
rf = RandomForestClassifier(criterion='gini',
                            max_features='auto',
                            min_samples_split=6,
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)

ab = AdaBoostClassifier(learning_rate=1)

# find the best number of estimator
param_grid = {"n_estimators" : [500, 750, 1000]}

# judge the performance of different estimators
gs = GridSearchCV(estimator=rf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train.values.ravel())
print(gs.best_params_)

"""
rf = RandomForestClassifier(criterion='gini',
                            min_samples_split=6,
                            n_estimators=1000,
                            max_features='auto',
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)

ab = AdaBoostClassifier(n_estimators=1000,
	                    learning_rate=1)
 
#eclf = VotingClassifier(estimators=[('rf', rf), ('ab', ab)], voting='soft')

rf.fit(X_train, y_train.values.ravel())
print "%.4f" % rf.oob_score_

# pridiction
predictions = rf.predict(X_test)

y_test = pd.read_csv('y_test.csv')
pred = pd.DataFrame(predictions, columns = [y_test.columns[1]])
del y_test['status_group']
y_test = pd.concat((y_test, pred), axis = 1)
y_test.to_csv(os.path.join('submission_files', 'y_test.csv'), sep=",", index = False)
"""