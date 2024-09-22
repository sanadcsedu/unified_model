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
from pathlib import Path
import json 
import glob 

class OnlineSVM:
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
        # do online prediction predict , partial_fit, predict, partial_fit
        all_accuracies = []
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            accuracy = accuracy_score([y_test[i]], y_pred)
            self.model.partial_fit([X_test[i]], [y_test[i]])
            all_accuracies.append(accuracy)

        return np.mean(all_accuracies), y_pred

    def evaluate2(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        #one shot prediction
        y_pred = self.model.predict(X_test)
        all_accuracies = accuracy_score(y_test, y_pred)

        return all_accuracies, y_pred

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def run_experiment(user_list):
    """
    Run the experiment using Leave-One-Out Cross-Validation (LOOCV).
    """
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  
            
    y_true_all = []
    y_pred_all = []
    train_accu = []
    test_accu = []
    # Leave-One-Out Cross-Validation
    for i, test_user_log in enumerate(tqdm(user_list)):
        train_users = user_list[:i] + user_list[i+1:]  # All users except the ith one

        # Aggregate training data
        X_train = []
        y_train = []
        for feedback_file in train_users:
            user_name = get_user_name(feedback_file)
            # print(user_name)
            excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
            # excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
            # print(excel_files)
            raw_file = [string for string in excel_files if user_name in string][0]

            env = environment5.environment5()
            # env.process_data('brightkite', raw_file, feedback_file, 'Actor-Critic') 
            env.process_data('faa', raw_file, feedback_file, 'Actor-Critic') 

            # env.process_data(dataset, user_log[0])
            # pdb.set_trace()
            X_train.extend(env.mem_states)
            y_train.extend(env.mem_action)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # print(len(X_train))
        # Initialize and train the OnlineSVM model
        model = OnlineSVM()
        model.train(X_train, y_train)

        # Test on the left-out user
        user_name = get_user_name(test_user_log)
        excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        # excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
        raw_file = [string for string in excel_files if user_name in string][0]

        env = environment5.environment5()
        # env.process_data('brightkite', raw_file, test_user_log, 'Actor-Critic') 
        env.process_data('faa', raw_file, feedback_file, 'Actor-Critic') 

        # Convert string representations of lists to actual lists for test data
        X_test = np.array(env.mem_states)
        y_test = np.array(env.mem_action)
        # print(len(X_test))
        result = []
        for _ in range(10):
        # Evaluate the model on the test data for this user
            # accuracy, y_pred = model.evaluate(X_test, y_test) # online_learning
            accuracy, y_pred = model.evaluate2(X_test, y_test) # offline_learning
            

            test_accu.append(accuracy)
        # y_true_all.extend(y_test)
        # y_pred_all.extend(y_pred)

    print("Online SVM Testing {:.2f}".format(np.mean(test_accu)))

if __name__ == "__main__":
    env = environment5.environment5()
    # user_list = env.user_list_brightkite
    user_list = env.user_list_faa
    # print(user_list)
    run_experiment(user_list)
        