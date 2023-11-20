import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment5
# import plotting
from collections import Counter,defaultdict
import json
from pathlib import Path
import glob
from tqdm import tqdm 
import os 

eps=1e-35
class Policy(nn.Module):
    def __init__(self,learning_rate,gamma,tau):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)
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
        self.pi = Policy(self.learning_rate, self.gamma,self.temperature)

    def train(self):
        
        all_predictions=[]
        for _ in range(50):
            s = self.env.reset(all = False, test = False)
            s=np.array(s)
            done = False
            actions =[]
            predictions=[]
            while not done:
                prob = self.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info = self.env.step(s,a,False)
                predictions.append(info)


                self.pi.put_data((r, prob[a]))

                s = s_prime

            self.pi.train_net()
            all_predictions.append(np.mean(predictions))
        # print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
        return self.pi, (np.mean(predictions)) #return last train_accuracy


    def test(self,policy):
        test_accuracies = []
        
        for _ in range(1):
            s = self.env.reset(all=False, test=True)
            done = False
            predictions = []
            actions = []

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info  = self.env.step(s, a, True)
                predictions.append(info)
                
                policy.put_data((r, prob[a]))

                s = s_prime
                self.pi.train_net()

            test_accuracies.append(np.mean(predictions))
        return np.mean(test_accuracies)


def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user


def run_experiment(user_list,algo,hyperparam_file):
    # Load hyperparameters from JSON file
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward'])

    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
    temperatures = hyperparams['temperatures']

    # aggregate_plotter = plotting.plotter(None)
    y_accu_all = []

    for feedback_file in user_list:
        # Extract user-specific threshold values
        threshold_h = hyperparams['threshold']
        # plotter = plotting.plotter(threshold_h)

        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]
        
        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0
            best_agent=None
            best_policy=None
            best_temp=0

            env = environment5.environment5()
            env.process_data('faa', raw_file, feedback_file, thres)            
            for learning_rate in learning_rates:
                for gamma in gammas:
                    for temp in temperatures:
                        agent = Reinforce(env,learning_rate,gamma,temp)
                        policy,accuracies = agent.train()

                        if accuracies > max_accu:
                            max_accu=accuracies
                            best_learning_rate=learning_rate
                            best_gamma=gamma
                            best_agent = agent
                            best_policy = policy
                            best_temp=temp

            print("#TRAINING: User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature:{}".format(user_name, thres,max_accu,best_learning_rate,best_gamma,best_temp))
            test_accuracy = best_agent.test(best_policy)
            
            print("#TESTING User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature: {}".format(user_name, thres,
                                                                                                     max_accu,
                                                                                                     best_learning_rate,
                                                                                                     best_gamma,best_temp))
     
if __name__ == '__main__':
    env = environment5.environment5()
    user_list_faa = env.user_list_faa
    run_experiment(user_list_faa, 'Reinforce', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name

