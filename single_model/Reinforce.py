import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment5
# import plotting
from collections import Counter, defaultdict
import json
from pathlib import Path
import glob
from tqdm import tqdm 
import os 
import multiprocessing

eps=1e-35
class Policy(nn.Module):
    def __init__(self,learning_rate,gamma,tau, dataset):
        super(Policy, self).__init__()
        self.data = []
        if dataset == 'faa':
            self.fc1 = nn.Linear(8, 128)
        else:
            self.fc1 = nn.Linear(9, 128)
    
        self.fc2 = nn.Linear(128, 4)

        self.gamma=gamma
        self.temperature = tau
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x / self.temperature
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class Reinforce():
    def __init__(self,env,learning_rate,gamma,tau):
        self.env = env
        self.learning_rate, self.gamma, self.temperature = learning_rate, gamma, tau
        # self.pi = Policy(self.learning_rate, self.gamma,self.temperature)

    def train(self, model):
        
        all_predictions=[]
        for _ in range(15):
            s = self.env.reset()
            s=np.array(s)
            done = False
            actions =[]
            predictions=[]
            while not done:
                prob = model(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info, _ = self.env.step(s,a,False)
                predictions.append(info)


                model.put_data((r, prob[a]))

                s = s_prime

            model.train_net()
    
            all_predictions.append(np.mean(predictions))

        return model, (np.mean(predictions)) #return last train_accuracy


    def test(self,model):
        test_accuracies = []
        
        for _ in range(1):
            s = self.env.reset()
            done = False
            predictions = []
            actions = []
            insight = defaultdict(list)

            while not done:
                prob = model(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, prediction, ground_action  = self.env.step(s, a, True)
                predictions.append(prediction)
                insight[ground_action].append(prediction)

                model.put_data((r, prob[a]))

                s = s_prime
                model.train_net()

            test_accuracies.append(np.mean(predictions))
        
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.sum(values))

        return np.mean(test_accuracies), granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def training(dataset, train_files, epoch, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    # Load hyperparameters from JSON file
    hyperparam_file='sampled_hyper_params.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
    temperatures = hyperparams['temperatures']

    best_lr = best_gamma = best_tau = max_accu = -1
    for lr in learning_rates:
        for ga in gammas:
            for tau in temperatures:
                accu = []
                model = Policy(lr, ga, tau, dataset)

                for feedback_file in train_files:
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
                    
                    agent = Reinforce(env, lr, ga, epoch)           
                            
                    model, accu_user = agent.train(model)
                    accu.append(accu_user)
                    
                #accuracy of the model learned over training data
                accu_model = np.mean(accu)
                if accu_model > max_accu:
                    max_accu = accu_model
                    best_lr = lr
                    best_gamma = ga
                    best_tau = tau
                    best_ac_model = model

    # print("Training Accuracy", max_accu)
    return best_ac_model, best_lr, best_gamma, best_tau, max_accu

def testing(dataset, test_files, trained_ac_model, best_lr, best_gamma, best_tau, algorithm):
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
        
        agent = Reinforce(env, best_lr, best_gamma, best_tau)           
        accu, granular_prediction = agent.test(trained_ac_model)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    
    return np.mean(final_accu), granular_prediction


if __name__ == '__main__':
    # datasets = ['brightkite', 'faa']
    datasets = ['faa']
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
            trained_ac_model, best_lr, best_gamma, best_tau, training_accuracy = training(d, train_files, 5, 'Reinforce')
            X_train.append(training_accuracy)
            # test user
            test_files = [test_user_log]
            testing_accu, gp = testing(d, test_files, trained_ac_model, best_lr, best_gamma, best_tau, 'Reinforce')
            # print("Testing Accuracy ", accu)
            for key, val in gp.items():
                # print(key, val)
                split_accs[key].append(val[1])
                split_cnt[key].append(val[0])

            X_test.append(testing_accu)
            # accuracies.append(accu)

        # train_accu = np.mean(X_train)
        test_accu = np.mean(X_test)
        # print("Reinforce Training {:.2f}".format(train_accu))
        print("Reinforce Testing {:.2f}".format(test_accu))

        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 