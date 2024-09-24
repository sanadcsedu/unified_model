import pandas as pd
import ast
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
import pdb
import random
from itertools import combinations
import random
import environment5
from pathlib import Path
import glob
import os
from read_data import read_data
import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

if __name__ == "__main__":
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  
    env = environment5.environment5()
        
    action_space = {'pan': 0, 'zoom': 1, 'brush': 2, 'range select': 3}
    test_accuracy = []
    for idx, dataset in enumerate(['brightkite', 'faa']):
        split_accs = [[] for _ in range(4)]
        split_cnt = [[] for _ in range(4)]      
        print("Dataset ", dataset)
        if dataset == 'faa':
            user_list = env.user_list_faa
        else:
            user_list = env.user_list_brightkite

        # Leave-One-Out Cross-Validation
        for i, test_user_log in enumerate((user_list)):
            train_users = user_list[:i] + user_list[i+1:]  # All users except the ith one

            for feedback_file in train_users:
                user_name = get_user_name(feedback_file)
                # print(user_name)
                if dataset == 'faa':
                    excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
                else:
                    excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
                # print(excel_files)
                raw_file = [string for string in excel_files if user_name in string][0]

                obj = read_data()
                file = obj.merge(raw_file, feedback_file)
                data2 = []
                for line in file:
                    z = ast.literal_eval(str(line))
                    data2.append(z)

                sequences = [(row[2] + "-" + str(row[3]), row[1]) for row in data2]
                
                trainer = nltk.HiddenMarkovModelTagger.train([sequences])

            # Test on the left-out user
            user_name = get_user_name(test_user_log)
            if dataset == 'faa':
                    excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
            else:
                excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
            raw_file = [string for string in excel_files if user_name in string][0]
            
            obj = read_data()
            file = obj.merge(raw_file, feedback_file)
            data2 = []
            for line in file:
                z = ast.literal_eval(str(line))
                data2.append(z)

            sequences = [(row[2] + "-" + str(row[3]), row[1]) for row in data2] # Prepared Test Sequence

            predicted_tags = []
            true_tags = []
            states=[]
            insight = [[] for _ in range(4)]

            for ii in range(len(sequences)):
                try:
                    state, true_tag = sequences[ii]
                    prediction = trainer.tag([state])
                    predicted_tag = prediction[0][1]
                    trainer = nltk.HiddenMarkovModelTagger.train([sequences[:ii]])
                except ValueError:
                    continue

                predicted_tags.append(predicted_tag)
                true_tags.append(true_tag)
                states.append(state)
                
                if predicted_tag == true_tag:
                    insight[action_space[true_tag]].append(1)
                else:
                    insight[action_space[true_tag]].append(0)

            # pdb.set_trace()
            assert len(states) == len(true_tags) == len(predicted_tags)
            
            #Calculate accuracy between predicted_tags and true_tags
            accuracy = np.mean(np.array(true_tags) == np.array(predicted_tags))
            # print(user_name, accuracy)
            test_accuracy.append(accuracy)

            for ii in range(4):
                split_accs[ii].append(np.sum(insight[ii]))
                split_cnt[ii].append(len(insight[ii]))

            del trainer

        print("HMM Testing {:.2f}".format(np.mean(test_accuracy)))
        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 
