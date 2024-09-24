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
from tqdm import tqdm
import pdb 

eps=1e-35
class Momentum:
    def __init__(self):
        self.last_action = defaultdict()

    def test(self, env):
        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        insight = defaultdict(list)
        length = len(env.mem_action)
        r = 0
        for i in range(length):
            denom += 1
            try: #Finding the last action in the current state
                # candidate = self.last_action[env.mem_states[i]]
                # Generate a random number between 0 and 1
                probability = random.random()
                if probability < 1:
                    # Select the last action for the current state
                    candidate = self.last_action[env.mem_states[i]]
                else:
                    # Select a random action from the list of possible actions
                    candidate = random.choice([0, 1, 2, 3])

            except KeyError: #Randomly picking an action if the current state is new 
                candidate = random.choice([0, 1, 2, 3])
                r += 1

            if candidate == env.mem_action[i]:
                accuracy += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)
            
            self.last_action[env.mem_states[i]] = env.mem_action[i]
        
        # pdb.set_trace()
        # print(denom, r)
        accuracy /= denom
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.sum(values))

        return accuracy, granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user


def testing(dataset, test_files, algorithm):
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
        
        model = Momentum()
        accu, granular_prediction = model.test(env)
        
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
            test_files = [test_user_log]
            
            testing_accu, gp = testing(d, test_files, 'Greedy')
            # print(gp)
            for key, val in gp.items():
                # print(key, val)
                split_accs[key].append(val[1])
                split_cnt[key].append(val[0])

            # print("Testing Accuracy ", accu)
            X_test.append(testing_accu)
            # accuracies.append(accu)

        test_accu = np.mean(X_test)
        print("Momentum Testing {:.2f}".format(test_accu))

        total = 0
        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 
            total += np.sum(split_cnt[i])