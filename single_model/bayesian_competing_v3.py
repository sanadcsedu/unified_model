import environment5
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import random
import multiprocessing
import glob 
import os 
from pathlib import Path
import pdb

eps=1e-35
class Bayesian:
    def __init__(self):
        self.freq = defaultdict(lambda: defaultdict(float))
        # self.reward = defaultdict(lambda: defaultdict(float))

    def train(self, env):
        length = len(env.mem_action)
        
        for i in range(length):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1
        
    def test(self, env):
        # Checking accuracy on the remaining data:
        current_model = self.freq
        accuracy = 0
        denom = 0
        insight = defaultdict(list)
        length = len(env.mem_action)
        for i in range(length):
            denom += 1
            try: #Picking an Action in Bayesian Way
                # actions = ['pan', 'zoom', 'brush', 'range select']
                actions = {'pan': 0, 'zoom': 1, 'brush': 2, 'range select': 3}
                freqs = []
                for a, v in actions.items():
                    if v in self.freq[env.mem_states[i]]:
                        freqs.append(self.freq[env.mem_states[i]][v])
                    else:
                        freqs.append(1)
                _max = random.choices(list(actions.values()), weights=freqs, k=1)[0]
            except IndexError:
                _max= random.choice([0, 1, 2, 3])
            
            if _max == env.mem_action[i]:
                accuracy += 1
                # self.reward[env.mem_states[i]][_max] += env.mem_reward[i]
                # self.freq[env.mem_states[i]][env.mem_action[i]] += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)

            self.freq[env.mem_states[i]][env.mem_action[i]] += 1
        
        accuracy /= denom
        self.freq.clear()
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.sum(values))

        return accuracy, granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def training(dataset, train_files):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory         
    
    model = Bayesian()

    for feedback_file in train_files:
        user_name = get_user_name(feedback_file)
        if dataset == 'faa':
            excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        else:
            excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
        raw_file = [string for string in excel_files if user_name in string][0]

        env = environment5.environment5()
        if dataset == 'faa':
            env.process_data('faa', raw_file, feedback_file, 'Greedy') 
        else:
            env.process_data('brightkite', raw_file, feedback_file, 'Greedy') 
            
        model.train(env)
                   
    return model

def testing(dataset, test_files, trained_greedy_model, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    final_accu = []
    for feedback_file in test_files:
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
            
        accu, granular_prediction = trained_greedy_model.test(env)
        
        final_accu.append(accu)
    
    return np.mean(final_accu), granular_prediction

if __name__ == '__main__':
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

        accuracies = []
        X_train = []
        X_test = []

        # Leave-One-Out Cross-Validation
        for i, test_user_log in enumerate((user_list)):
            train_files = user_list[:i] + user_list[i+1:]  # All users except the ith one
            # train_files, test_files = train_test_split(user_list, test_size=0.3, random_state=42)
            trained_greedy_model = training(d, train_files)
            # print(trained_greedy_model.reward)
            # test user
            test_files = [test_user_log]
            # print(test_files)
            testing_accu, gp = testing(d, test_files, trained_greedy_model, 'Greedy')
            # print(gp)
            for key, val in gp.items():
                # print(key, val)
                split_accs[key].append(val[1])
                split_cnt[key].append(val[0])

            # print("Testing Accuracy ", accu)
            X_test.append(testing_accu)
            # accuracies.append(accu)

        test_accu = np.mean(X_test)
        print("Bayesian Testing {:.2f}".format(test_accu))

        total = 0
        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 
            total += np.sum(split_cnt[i])
        
        for i in range(4):
            accu = round(np.sum(split_cnt[i]) / total, 2)
            print("Action {} Probability of Occurance {}".format(i, accu))
