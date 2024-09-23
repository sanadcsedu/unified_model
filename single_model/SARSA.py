import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import environment5 as environment5
import random
import multiprocessing
import time
from pathlib import Path
import glob
from tqdm import tqdm 
import os 
from sklearn.model_selection import train_test_split
import json 
import pdb


class SARSA:
    def __init__(self):
        pass

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):

        for _ in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            state = env.reset()
            action = policy(state)

            training_accuracy = []

            # One step in the environment
            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, _ = env.step(state, action, False)
                training_accuracy.append(prediction)

                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                state = next_state
                action = best_next_action
                if done:
                    break

        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):
            state = env.reset()
            stats = []
            insight = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, True)

                stats.append(prediction)
                insight[ground_action].append(prediction)

                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                state = next_state
                if done:
                    break

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(stats), granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def training(dataset, train_files, epoch, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    #loading the hyper-parameters 
    hyperparam_file='sampled_hyper_params.json'
    with open(hyperparam_file) as f:
            hyperparams = json.load(f)
    # Extract hyperparameters from JSON file
    discount_h =hyperparams['gammas']
    alpha_h = hyperparams['learning_rates']
    epsilon_h = hyperparams['epsilon']
    
    best_discount = best_alpha = best_eps = max_accu = -1
    for eps in epsilon_h:
        for alp in alpha_h:
            for dis in discount_h:
                accu = []
                model = SARSA()
                Q = defaultdict(lambda: np.zeros(len(env.action_space)))
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

                    # updates the Q value after each user trajectory
                    # print(user[0])
                    Q, accu_user = model.sarsa(Q, env, epoch, dis, alp, eps)
                    # print(user[0], eps, alp, dis, accu_user)
                    accu.append(accu_user)
                   
                #accuracy of the model learned over training data
                accu_model = np.mean(accu)
                if accu_model > max_accu:
                    max_accu = accu_model
                    best_eps = eps
                    best_alpha = alp
                    best_discount = dis
                    best_q=Q
    # print("Training Accuracy", max_accu)
    return best_q, best_alpha, best_eps, best_discount, max_accu

def testing(dataset, test_files, trained_Q, alpha, eps, discount, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    Q = trained_Q
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

        model = SARSA()
        accu, _ = model.test(env, Q, discount, alpha, eps)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    # print("Q-Learning, {}, {:.2f}".format(k, np.mean(final_accu)))
    return np.mean(final_accu)

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

        accuracies = []
        X_train = []
        X_test = []

        # Leave-One-Out Cross-Validation
        for i, test_user_log in enumerate(tqdm(user_list)):
            train_files = user_list[:i] + user_list[i+1:]  # All users except the ith one
            # train_files, test_files = train_test_split(user_list, test_size=0.3, random_state=42)
            trained_Q, best_alpha, best_eps, best_discount, training_accuracy = training(d, train_files, 5, 'SARSA')
            X_train.append(training_accuracy)
            # test user
            test_files = [test_user_log]
            testing_accu = testing(d, test_files, trained_Q, best_alpha, best_eps, best_discount, 'SARSA')
            # print("Testing Accuracy ", accu)
            X_test.append(testing_accu)
            # accuracies.append(accu)

        # train_accu = np.mean(X_train)
        test_accu = np.mean(X_test)
        # print("SARSA Training {:.2f}".format(train_accu))
        print("SARSA Testing {:.2f}".format(test_accu))

        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 