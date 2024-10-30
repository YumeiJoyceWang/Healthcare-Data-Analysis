import utils
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
from sklearn.metrics import *



# setup the randoms tate
RANDOM_STATE = 545510477


'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	test_events = pd.read_csv('../data/test/events.csv')
	feature_map = pd.read_csv('../data/test/event_feature_map.csv')

	# aggregate events
	event_indx = pd.merge(test_events, feature_map, on='event_id')
	event_indx.dropna(subset=['value'])
	sum_df = event_indx[event_indx.event_id.str.startswith(('DRUG', 'DIAG'))]
	count_df = event_indx[event_indx.event_id.str.startswith('LAB')]
	sum_df = sum_df.groupby(['patient_id', 'event_id', 'idx']).value.sum().reset_index()
	count_df = count_df.groupby(['patient_id', 'event_id', 'idx']).value.count().reset_index()

	aggregated_events = pd.concat([sum_df, count_df], ignore_index=True)
	aggregated_events = aggregated_events.rename(columns={'idx': 'feature_id', 'value': 'feature_value'})
	
	# normalize the values column using min-max normalization(the min value will be 0 in all scenarios)
	pivoted = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
	scaled = pivoted / pivoted.max()
	scaled = scaled.reset_index()
	aggregated_events = pd.melt(scaled, id_vars='patient_id', value_name='feature_value').dropna()

	# create features
	patient_features = aggregated_events.groupby('patient_id').apply(
		lambda x: list(x.sort_values('feature_id').apply(lambda y: (y.feature_id, y.feature_value), axis=1)))
	patient_features = patient_features.to_dict()

	# save file as "test_features.txt"
	deliverable3 = open('../deliverables/test_features.txt', 'wb')
	deliverable4 = open('../data/test/features_svmlight.test', 'wb')

	sorted_features = dict()
	for key in sorted(patient_features):
		sorted_features[key] = sorted(patient_features[key])
	for patient, features in sorted_features.items():
		deliverable3.write(bytes(("{} {} \r\n".format(int(patient), utils.bag_to_svmlight(features))), 'UTF-8'));  # Use 'UTF-8'
		deliverable4.write(bytes(("{} {} \r\n".format(int(patient), utils.bag_to_svmlight(features))), 'UTF-8'));  # Use 'UTF-8'

	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../data/test/features_svmlight.test")

	return X_train, Y_train, X_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	parameters = {
		"loss": ["deviance"],
		"learning_rate": [0.1, 0.3, 0.5],
		"min_samples_leaf": [3, 5, 8],
		"max_depth": [3, 5],
		"n_estimators": [100]
	}
	clf = GridSearchCV(GB(), parameters, cv=5, n_jobs=-1)
	clf.fit(X_train, Y_train)
	# print(clf.score(X_train, Y_train))
	# print(clf.best_params_)
	Y_pred = clf.predict(X_test)

	return Y_pred


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.
	#X_cv, Y_cv = utils.get_data_from_svmlight("../data/features_svmlight.validate")
	# clf = GB(n_estimators=100, learning_rate=0.1, max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE)
	# clf.fit(X_train, Y_train)
	# Y_pred_cv = clf.predict(X_cv)
	# auc = roc_auc_score(Y_cv, Y_pred_cv)
	# print(auc)




if __name__ == "__main__":
    main()

	