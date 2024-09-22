import numpy as np
from collections import defaultdict
import itertools
import environment5 as environment5
import multiprocessing
from multiprocessing import Pool
import time
import random
from pathlib import Path
import glob
from tqdm import tqdm 
import os 
from sklearn.model_selection import train_test_split
import json 
import pdb

class Qlearning:
    def __init__(self):
        pass


    def epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """
        def policy_fnc(state):
            coin = random.random()
            if coin < epsilon:
                best_action = random.randint(0, nA-1)
                # print("random")
            else:
                best_action = np.argmax(Q[state])
                # print("best")
            return best_action

        return policy_fnc

    def q_learning(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: setting the environment as local fnc by importing env earlier
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        
        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            # Reset the environment and pick the first state
            state = env.reset()
            training_accuracy=[]
            for t in itertools.count():
                # Take a step
                action = policy(state)

                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                # print(Q[state], action, state)
                training_accuracy.append(prediction)
                # if action == ground_action:
                #     reward = 1
                # else:
                #     reward = -1
                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                # print("State {}, action {}".format(state, action))
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                # print(Q[state], best_next_action, next_state, Q[next_state])
                # pdb.set_trace()

                #updating based on ground action
                # best_next_action = np.argmax(Q[next_state])
                # td_target = 1 + discount_factor * Q[next_state][best_next_action]
                # td_delta = td_target - Q[state][ground_action]
                # Q[state][ground_action] += alpha * (td_delta)

                state = next_state
                if done:
                    break


        return Q, np.mean(training_accuracy)


    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):

            state = env.reset()
            stats = []
            
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
            insight = defaultdict(list)
            for t in itertools.count():
            
                action = policy(state)
                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                next_state, reward, done, prediction, ground_action = env.step(state, action, True)
                # print(state, Q[state], ground_action, action, prediction)
                # if action == ground_action:
                #     reward = 10
                # else:
                #     reward = -10
                # pdb.set_trace()
                stats.append(prediction)
                insight[ground_action].append(prediction)

                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                #updating based on ground action
                # best_next_action = np.argmax(Q[next_state])
                # td_target = 10 + discount_factor * Q[next_state][best_next_action]
                # td_delta = td_target - Q[state][ground_action]
                # Q[state][ground_action] += alpha * (td_delta)
                # print("Updated Q {}".format(Q[state]))                

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

def training(train_files, epoch):
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
                model = Qlearning()
                Q = defaultdict(lambda: np.zeros(len(env.action_space)))
                for feedback_file in train_files:
                    user_name = get_user_name(feedback_file)
                    # print(user_name)
                    # excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
                    excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
                    # print(excel_files)
                    raw_file = [string for string in excel_files if user_name in string][0]
                    env = environment5.environment5()
                    env.process_data('brightkite', raw_file, feedback_file, 'Qlearn') 
                    # env.process_data('faa', raw_file, feedback_file, 'Qlearn') 
                    #updates the Q value after each user trajectory
                    # print(user[0])
                    Q, accu_user = model.q_learning(Q, env, epoch, dis, alp, eps)
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

def testing(test_files, trained_Q, alpha, eps, discount, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    Q = trained_Q
    final_accu = []
    for feedback_file in test_files:
        user_name = get_user_name(feedback_file)
        # print(user_name)
        # excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
        # print(excel_files)
        raw_file = [string for string in excel_files if user_name in string][0]
        env = environment5.environment5()
        env.process_data('brightkite', raw_file, feedback_file, 'Qlearn') 
        # env.process_data('faa', raw_file, feedback_file, 'Qlearn') 

        model = Qlearning()
        accu, _ = model.test(env, Q, discount, alpha, eps)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    # print("Q-Learning, {}, {:.2f}".format(k, np.mean(final_accu)))
    return np.mean(final_accu)

if __name__ == "__main__":
    final_output = []
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    
    accuracies = []
    X_train = []
    X_test = []
    # for _ in range(num_iterations):
    # Leave-One-Out Cross-Validation
    for i, test_user_log in enumerate(tqdm(user_list)):
        train_files = user_list[:i] + user_list[i+1:]  # All users except the ith one
        # train_files, test_files = train_test_split(user_list, test_size=0.3, random_state=42)
        trained_Q, best_alpha, best_eps, best_discount, training_accuracy = training(train_files, 5)
        X_train.append(training_accuracy)
        # test user
        test_files = [test_user_log]
        testing_accu = testing(test_files, trained_Q, best_alpha, best_eps, best_discount, 'Qlearn')
        # print("Testing Accuracy ", accu)
        X_test.append(testing_accu)
        # accuracies.append(accu)

    train_accu = np.mean(X_train)
    test_accu = np.mean(X_test)
    print("Q-learning Training {:.2f}".format(train_accu))
    print("Q-learning Testing {:.2f}".format(test_accu))