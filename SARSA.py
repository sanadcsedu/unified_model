import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
# import matplotlib.pyplot as plt
import sys
# import plotting
import environment5 as environment5
import random
import multiprocessing
import time
from pathlib import Path
import glob


class TD_SARSA:
    def __init__(self):
        pass

    # @jit(target ="cuda")
    def epsilon_greedy_policy(self, Q, epsilon, nA):
        # def policy_fnc(state):
        #     coin = random.random()
        #     if coin < epsilon:
        #         best_action = random.randint(0, 3)
        #     else:
        #         best_action = np.argmax(Q[state])
        #     return best_action

        # return policy_fnc
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(
        self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5
    ):
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).

        # Define the valid actions for each state

        Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        
        # The policy we're following
        
        for _ in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
        
            state = env.reset(all = False, test = False)
            # action_probs = policy(state)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action = policy(state)

            training_accuracy=[]

            # One step in the environment
            for t in itertools.count():
                
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction  = env.step(state, action, True)
            
                training_accuracy.append(prediction)
            
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta + prediction)

                state = next_state
                if done:
                    break
               
        # print(np.mean(training_accuracy))
        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):
            state = env.reset(all=False, test=True)
            stats = []
            model_actions = []
            
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
            
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                model_actions.append(action)
                next_state, reward, done, prediction  = env.step(state, action, True)
            
                stats.append(prediction)
            
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta + prediction)

                state = next_state
                if done:
                    break

        return np.mean(stats)

if __name__ == "__main__":
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    obj2 = misc.misc(len(user_list))

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[:2], 'SARSA',4, result_queue,))
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[2:4], 'SARSA',4, result_queue,))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[4:6], 'SARSA',4, result_queue,))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[6:], 'SARSA',4, result_queue,))
    
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
    print("SARSA ", ", ".join(f"{x:.2f}" for x in final_result))
    
# faa: 0.77, 0.76, 0.74, 0.74, 0.72, 0.71, 0.73, 0.73, 0.70
    0.63, 0.59, 0.63, 0.63, 0.62, 0.61, 0.65, 0.65, 0.75
# SARSA 0.74, 0.71, 0.72, 0.71, 0.69, 0.71, 0.68, 0.72, 0.69
# SARSA 0.63, 0.64, 0.68, 0.62, 0.67, 0.70, 0.72, 0.69, 0.69
    0.63, 0.59, 0.63, 0.63, 0.62, 0.61, 0.65, 0.65, 0.75