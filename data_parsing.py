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
            self.interest_map = {}
            self.children = {}
    def __init__(self, topics):
        self.head = self.Node()
        self.head.children = {}
        self.makeChildren(topics)
    def makeChildren(self,topics):
        for i in topics.index:
            strArray = topics.loc[i, 'topic_name'].split('/')
            current = self.head
            for str in strArray:
                if not current.children.get(str):
                    current.children[str] = self.Node(topic_name=str)
                current = current.children[str]
            current.id = i
    def get_node(self,topic):
        strArray = topic.split('/')
        current = self.head
        for str in strArray:
            current = current.children.get(str)
            if not current:
                raise KeyError()
        return current
    def get_id(self,topic):
        strArray = topic.split('/')
        current = self.head
        for str in strArray:
            current = current.children.get(str)
            if not current:
                raise KeyError()
        return current.id
    def get_interest(self,topic):
        strArray = topic.split('/')
        current = self.head
        for str in strArray:
            current = current.children.get(str)
            if not current:
                raise KeyError()
        return current.interest_map
    def get_interested_users(self,topic):
        return np.fromiter(self.get_interest(topic).values(), dtype=int)
    def make_map(self, data, feature_name):
        for userID in data.index:
            def add_users_interest(k,v):
                try:
                    self.get_node(interest_topics.loc[int(k), 'topic_name']).interest_map[userID] = float(v)
                except KeyError:
                    self.head.children[k] = self.Node(topicID=int(k))
            [add_users_interest(k,v) for k,v in data.loc[userID, feature_name].items()]


tree = TopicsTree(interest_topics)

tree.make_map(training_set, 'ltiFeatures')

tree.get_interest('/Sports/Team Sports/American Football')

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
