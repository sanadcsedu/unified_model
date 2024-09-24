import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import environment5 as environment5
from tqdm import tqdm
import pdb 
from collections import defaultdict
import glob 
from pathlib import Path


class OfflineSVM:
    def __init__(self, max_iter=1000):
        """
        Initializes the Online SVM model using SGDClassifier.
        """
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-3)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        #one shot prediction
        # y_pred = self.model.predict(X_test)
        # all_accuracies = accuracy_score(y_test, y_pred)

        # return all_accuracies, y_pred
        all_accuracies = []
        insight = defaultdict(list)
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            accuracy = accuracy_score([y_test[i]], y_pred)
            insight[y_test[i]].append(accuracy)
            all_accuracies.append(accuracy)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.sum(values))

        return np.mean(all_accuracies), granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def run_experiment(dataset, user_list, algorithm):
    """
    Run the experiment using Leave-One-Out Cross-Validation (LOOCV).
    """
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  
            
    split_accs = [[] for _ in range(4)]
    split_cnt = [[] for _ in range(4)]      
    test_accu = []
    # Leave-One-Out Cross-Validation
    for i, test_user_log in enumerate((user_list)):
        train_users = user_list[:i] + user_list[i+1:]  # All users except the ith one

        # Aggregate training data
        X_train = []
        y_train = []
        for feedback_file in train_users:
            user_name = get_user_name(feedback_file)
            if dataset == 'faa':
                excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
            else:
                excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
            raw_file = [string for string in excel_files if user_name in string][0]

            env = environment5.environment5()
            if dataset == 'faa':
                env.process_data('faa', raw_file, feedback_file, algorithm) 
            else:
                env.process_data('brightkite', raw_file, feedback_file, algorithm) 

            X_train.extend(env.mem_states)
            y_train.extend(env.mem_action)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # print(len(X_train))
        # Initialize and train the OnlineSVM model
        model = OfflineSVM()
        model.train(X_train, y_train)

        # Test on the left-out user
        user_name = get_user_name(test_user_log)
        if dataset == 'faa':
            excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        else:
            excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
        raw_file = [string for string in excel_files if user_name in string][0]

        env = environment5.environment5()
        if dataset == 'faa':
            env.process_data('faa', raw_file, feedback_file, algorithm) 
        else:
            env.process_data('brightkite', raw_file, feedback_file, algorithm) 

        # Convert string representations of lists to actual lists for test data
        X_test = np.array(env.mem_states)
        y_test = np.array(env.mem_action)
        # print(len(X_test))
        result = []
        # for _ in range(10):
        # Evaluate the model on the test data for this user
        accuracy, gp = model.evaluate(X_test, y_test) # online_learning
        # accuracy, y_pred = model.evaluate2(X_test, y_test) # offline_learning
        for key, val in gp.items():
            split_accs[key].append(val[1])
            split_cnt[key].append(val[0])

        test_accu.append(accuracy)
        
    return test_accu, split_accs, split_cnt

if __name__ == "__main__":
    datasets = ['brightkite', 'faa']
    for d in datasets:
        print("Dataset ", d)   
        split_accs = [[] for _ in range(4)]
        split_cnt = [[] for _ in range(4)]      
        env = environment5.environment5()
        
        if d == 'faa':
            user_list = env.user_list_faa
        else:
            user_list = env.user_list_brightkite

        test_accu, split_accs, split_cnt = run_experiment(d, user_list, 'SVM')
        print("Online SVM Testing {:.2f}".format(np.mean(test_accu)))

        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 