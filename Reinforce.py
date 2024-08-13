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

        self.fc1 = nn.Linear(9, 64)
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
        for _ in range(10):
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
                s_prime, r, done, info, _ = self.env.step(s,a,False)
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
            insight = defaultdict(list)

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, prediction, ground_action  = self.env.step(s, a, True)
                predictions.append(prediction)
                insight[ground_action].append(prediction)

                policy.put_data((r, prob[a]))

                s = s_prime
                self.pi.train_net()

            test_accuracies.append(np.mean(predictions))
        
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(test_accuracies), granular_prediction

class run_reinforce:
    def __init__(self):
        pass

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

    def run_experiment(self, user_list,algo,hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        temperatures = hyperparams['temperatures']

        output_list = []
        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((4, 9), dtype = float)
        final_split_accu = np.zeros((4, 9), dtype = float)

        for feedback_file in user_list:
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']
            # plotter = plotting.plotter(threshold_h)

            user_name = self.get_user_name(feedback_file)
            # excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')            

            raw_file = [string for string in excel_files if user_name in string][0]
            
            accu = []
            accu_split = [[] for _ in range(4)]
            cnt_split = [[] for _ in range(4)]
            
            output_list.append(f"user {user_name}")
            env = environment5.environment5()
            for thres in threshold_h:
                max_accu = -1
                best_learning_rate = 0
                best_gamma = 0
                best_agent=None
                best_policy=None
                best_temp=0
                # env.process_data('faa', raw_file, feedback_file, thres, 'Reinforce')            
                env.process_data('brightkite', raw_file, feedback_file, thres, 'Reinforce') 

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
                split_accs = [[] for _ in range(4)]

                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_policy
                    temp_accuracy, gp = test_agent.test(test_model)
                    test_accs.append(temp_accuracy)

                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                    
                test_accuracy = np.mean(test_accs)
                accu.append(test_accuracy)
                env.reset(True, False)

                output_list.append(f"Threshold: {thres}")
                for ii in range(4):
                    if len(split_accs[ii]) > 0:
                        # print("action: {}, count: {}, accuracy:{}".format(ii, gp[ii][0], np.mean(split_accs[ii])))
                        output_list.append(f"action: {ii}, count: {gp[ii][0]}, accuracy:{np.mean(split_accs[ii])}")
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        # print("{} Not Present".format(ii))
                        output_list.append(f"{ii} Not Present")
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)

            # print(user_name, accu)
            print(user_name, ", ".join(f"{x:.2f}" for x in accu))
            output_list.append(f"{user_name}, {', '.join(f'{x:.2f}' for x in accu)}")

            final_accu = np.add(final_accu, accu)
            for ii in range(4):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(4):            
            final_split_accu[ii] /= len(user_list)
            final_cnt[ii] /= len(user_list)
        
        result_queue.put(final_accu)
        info.put(output_list)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)

if __name__ == '__main__':
    final_output = []
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite

    # run_experiment(user_list, 'Actor_Critic', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name
    obj2 = run_reinforce()

    result_queue = multiprocessing.Queue()
    info = multiprocessing.Queue()
    info_split = multiprocessing.Queue()
    info_split_cnt = multiprocessing.Queue() 

    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'Reinforce', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'Reinforce', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'Reinforce', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'Reinforce', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    
    split_final = np.zeros((4, 9), dtype = float)
    split_final_cnt = np.zeros((4, 9), dtype = float)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype = float)
    p1.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
    # print(split_final_cnt)
    p2.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p3.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p4.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    final_result /= 4
    split_final /= 4
    split_final_cnt /= 4

    print("Reinforce ", ", ".join(f"{x:.2f}" for x in final_result))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

#FAA
# u4 0.36, 0.45, 0.68, 0.69, 0.71, 0.73, 0.76, 0.68, 0.69
# u6 0.71, 0.66, 0.83, 0.84, 0.87, 0.84, 0.84, 0.75, 0.66
# u7 0.56, 0.64, 0.56, 0.56, 0.20, 0.52, 0.67, 0.53, 0.22
# u3 0.65, 0.61, 0.56, 0.48, 0.37, 0.30, 0.06, 0.04, 0.00
# u1 0.40, 0.37, 0.43, 0.50, 0.58, 0.48, 0.46, 0.42, 0.90
# u2 0.56, 0.51, 0.44, 0.35, 0.21, 0.02, 0.17, 0.13, 0.06
# u5 0.84, 0.82, 0.82, 0.92, 0.91, 0.89, 0.85, 0.91, 0.86
# u8 0.86, 0.85, 0.82, 0.80, 0.73, 0.71, 0.64, 0.46, 0.59
# Reinforce  0.62, 0.61, 0.64, 0.64, 0.57, 0.56, 0.56, 0.49, 0.50

# Action  0 0.04, 0.10, 0.14, 0.17, 0.23, 0.16, 0.20, 0.24, 0.33
# Action  1 0.01, 0.00, 0.02, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00
# Action  2 0.69, 0.64, 0.61, 0.57, 0.60, 0.56, 0.47, 0.46, 0.25
# Action  3 0.44, 0.47, 0.59, 0.69, 0.63, 0.71, 0.77, 0.63, 0.28

# Action  0 57.62, 56.38, 51.75, 45.12, 40.00, 34.12, 25.75, 17.50, 8.25
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.12, 106.25, 92.00, 78.12, 62.00, 43.88, 33.00, 25.50, 12.25
# Action  3 120.00, 108.25, 92.75, 78.88, 65.62, 55.12, 41.12, 23.25, 11.88

#Brightkite

# u14 0.48, 0.58, 0.55, 0.56, 0.61, 0.76, 0.77, 0.79, 0.99
# u11 0.78, 0.81, 0.80, 0.79, 0.76, 0.79, 0.78, 0.68, 0.36
# u9 0.69, 0.81, 0.76, 0.88, 0.90, 0.97, 0.98, 0.99, 0.98
# u15 0.66, 0.63, 0.62, 0.72, 0.68, 0.73, 0.70, 0.70, 0.81
# u10 0.89, 0.88, 0.86, 0.88, 0.88, 0.89, 0.86, 0.81, 0.63
# u12 0.94, 0.95, 0.95, 0.96, 0.96, 0.95, 0.96, 0.94, 0.89
# u13 0.81, 0.82, 0.82, 0.81, 0.77, 0.87, 0.91, 0.90, 0.91
# u16 0.58, 0.61, 0.57, 0.64, 0.62, 0.66, 0.57, 0.52, 0.88

# Reinforce  0.73, 0.76, 0.74, 0.78, 0.77, 0.83, 0.82, 0.79, 0.81

# Action  0 0.90, 0.89, 0.87, 0.87, 0.82, 0.99, 0.99, 0.99, 0.74
# Action  1 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  2 0.79, 0.87, 0.85, 0.86, 0.85, 0.80, 0.83, 0.77, 0.56
# Action  3 0.13, 0.19, 0.19, 0.22, 0.25, 0.24, 0.17, 0.15, 0.04

ß# Action  0 211.38, 184.00, 162.38, 136.75, 105.12, 80.25, 61.00, 45.12, 27.75
# Action  1 9.12, 7.25, 6.00, 4.62, 3.88, 2.50, 2.12, 1.88, 1.12
# Action  2 147.75, 139.12, 122.75, 110.62, 101.50, 88.50, 66.25, 37.38, 16.00
# Action  3 115.62, 99.38, 84.75, 69.88, 57.38, 42.75, 30.75, 21.75, 7.25
