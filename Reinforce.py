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
import multiprocessing

eps=1e-35
class Policy(nn.Module):
    def __init__(self,learning_rate,gamma,tau):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 4)
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
        for _ in range(5):
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

class run_reinforce:
    def __init__(self):
        pass

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

    def run_experiment(self, user_list,algo,hyperparam_file, result_queue):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        temperatures = hyperparams['temperatures']

        # aggregate_plotter = plotting.plotter(None)
        final_accu = np.zeros(9, dtype=float)
        for feedback_file in user_list:
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']
            # plotter = plotting.plotter(threshold_h)

            user_name = self.get_user_name(feedback_file)
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            raw_file = [string for string in excel_files if user_name in string][0]
            
            accu = []
            env = environment5.environment5()
            for thres in threshold_h:
                max_accu = -1
                best_learning_rate = 0
                best_gamma = 0
                best_agent=None
                best_policy=None
                best_temp=0
                env.process_data('faa', raw_file, feedback_file, thres, 'Reinforce')            
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

                test_accs = []
                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_policy
                    temp_accuracy = test_agent.test(test_model)
                    test_accs.append(temp_accuracy)
                test_accuracy = np.mean(test_accs)
                # test_accuracy = best_agent.test(best_policy)
                accu.append(test_accuracy)
            print(user_name, accu)
            final_accu = np.add(final_accu, accu)
        final_accu /= len(user_list)
        # print("Reinforce: ")
        # print(np.round(final_accu, decimals=2))
        result_queue.put(final_accu)

if __name__ == '__main__':
    env = environment5.environment5()
    user_list = env.user_list_faa
    # run_experiment(user_list, 'Actor_Critic', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name
    obj2 = run_reinforce()

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'Reinforce', 'sampled_hyper_params.json', result_queue,))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'Reinforce', 'sampled_hyper_params.json', result_queue,))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'Reinforce', 'sampled_hyper_params.json', result_queue,))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'Reinforce', 'sampled_hyper_params.json', result_queue,))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype = float)
    p1.join()
    # temp = result_queue.get()
    final_result = np.add(final_result, result_queue.get())
    p2.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    p3.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    p4.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    final_result /= 4
    print("Reinforce ", ", ".join(f"{x:.2f}" for x in final_result))