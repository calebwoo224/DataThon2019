import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

data_directory = 'Duke_Datathon_10-29-19/valassis_dataset/'

def process_data(data_directory):
    interest_topics = pd.read_csv(data_directory + 'interest_topics.csv', index_col='topic_id')
    training_set = pd.read_csv(data_directory + 'training.csv', index_col='userID')
    validation = pd.read_csv(data_directory + 'validation.csv', index_col='userID')

    training_set['ltiFeatures'] = training_set['ltiFeatures'].apply(literal_eval)
    training_set['stiFeatures'] = training_set['stiFeatures'].apply(literal_eval)

    validation['ltiFeatures'] = validation['ltiFeatures'].apply(literal_eval)
    validation['stiFeatures'] = validation['stiFeatures'].apply(literal_eval)

    return interest_topics, training_set, validation

interest_topics, training_set, validation = process_data(data_directory)

class TopicsTree:
    class Node:
        def __init__(self, topic_name=None, topicID=None):
            self.name = topic_name
            self.id = topicID
            self.children = {}
    def __init__(self, topics):
        self.head = Node()
        self.head.children = {}
    def makeChildren(topics):
        for i in topics.index:
            strArray = topics.loc(i, 'topic_name').split('/')
            parent = self.head
            for str in strArray:
                parent.children.

# this function makes a wide-form dataframe where each column is a different topic for a userID
def make_topic_table(df, feature_name):
    features = df[feature_name]
    newdf = pd.DataFrame(index=df.index)
    def get_dict_key(dict, key):
        val = dict.get(key)
        if val:
            return val
        else:
            return 0
    for topicID in interest_topics.index:
        newdf[str(topicID)] = [get_dict_key(dict, str(topicID)) for dict in features]

    return newdf

from sklearn.ensemble import RandomForestClassifier

X_train = training_set['ltiFeatures']
y_train = training_set['inAudience']

X_test = validation['ltiFeatures']
y_test = validation['inAudience']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

make_topic_table(training_set, 'ltiFeatures').to_csv(data_directory + 'training_lti_topics.csv')
make_topic_table(training_set, 'stiFeatures').to_csv(data_directory + 'training_sti_topics.csv')
make_topic_table(validation, 'ltiFeatures').to_csv(data_directory + 'validation_lti_topics.csv')
make_topic_table(validation, 'stiFeatures').to_csv(data_directory + 'validation_sti_topics.csv')
