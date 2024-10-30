from datetime import timedelta
from collections import OrderedDict
from operator import getitem

from sklearn import preprocessing

import utils
import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    dead = mortality.copy()
    dead['timestamp'] = pd.to_datetime(dead['timestamp'])
    dead['timestamp'] = dead['timestamp'] - timedelta(days=30)

    alive = events[~events['patient_id'].isin(dead.patient_id)]
    alive = alive.groupby(['patient_id']).timestamp.max().reset_index()

    indx_date = pd.concat([dead, alive], ignore_index=True)
    indx_date.rename(columns={'timestamp':'indx_date'}, inplace=True)
    indx_date.to_csv(deliverables_path+'etl_index_dates.csv',columns = ['patient_id','indx_date'],index = False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    events['timestamp'] = pd.to_datetime(events['timestamp'])
    indx_date['indx_date'] = pd.to_datetime(indx_date['indx_date'])
    events_merge = pd.merge(events, indx_date, on='patient_id')
    filtered_events = events_merge[(events_merge.timestamp <= events_merge.indx_date) & (
                events_merge.timestamp >= (events_merge.indx_date - timedelta(days=2000)))]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    event_indx = pd.merge(filtered_events_df, feature_map_df, on='event_id')
    event_indx.dropna(subset=['value'])
    sum_df = event_indx[event_indx.event_id.str.startswith(('DRUG', 'DIAG'))]
    count_df = event_indx[event_indx.event_id.str.startswith('LAB')]
    sum_df = sum_df.groupby(['patient_id','event_id', 'idx']).value.sum().reset_index()
    count_df = count_df.groupby(['patient_id','event_id', 'idx']).value.count().reset_index()

    aggregated_events = pd.concat([sum_df, count_df], ignore_index=True)
    aggregated_events = aggregated_events.rename(columns={'idx': 'feature_id', 'value': 'feature_value'})
    # normalize the values column using min-max normalization(the min value will be 0 in all scenarios)
    pivoted = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
    scaled = pivoted / pivoted.max()
    scaled = scaled.reset_index()
    aggregated_events = pd.melt(scaled, id_vars='patient_id', value_name='feature_value').dropna()
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',
                             columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = aggregated_events.groupby('patient_id').apply(lambda x: list(x.sort_values('feature_id').apply(lambda y: (y.feature_id, y.feature_value), axis=1)))
    patient_features = patient_features.to_dict()
    mortality_df = mortality[['patient_id', 'label']]
    mortality = dict(zip(mortality_df.patient_id, mortality_df.label))

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    # patient_features = utils.sort_dict(patient_features)
    sorted_features = dict()
    for key in sorted(patient_features):
        sorted_features[key] = sorted(patient_features[key])

    for patient, features in sorted_features.items():
        deliverable1.write(bytes(("{} {} \r\n".format(mortality.get(patient, 0),
                                          utils.bag_to_svmlight(features))),'UTF-8')); #Use 'UTF-8'
        deliverable2.write(bytes(("{} {} {} \r\n".format(int(patient), mortality.get(patient, 0),
                                              utils.bag_to_svmlight(features))),'UTF-8'));



def main():
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality, feature_map = read_csv(train_path)
    deliverables_path = '../deliverables/'
    indx_date = calculate_index_date(events, mortality, deliverables_path)


    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()