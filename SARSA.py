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
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):

        Q = defaultdict(lambda: np.zeros(len(env.action_space)))

        # The policy we're following

        for _ in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            state = env.reset(all=False, test=False)

            action = policy(state)

            training_accuracy = []

            # One step in the environment
            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, _ = env.step(state, action, False)
                training_accuracy.append(prediction)

                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
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
            state = env.reset(all=False, test=True)
            stats = []
            insight = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, True)

                stats.append(prediction)
                insight[ground_action].append(prediction)
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
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


if __name__ == "__main__":
    final_output = []
    env = environment5.environment5()
    user_list = env.user_list_faa
    # user_list = env.user_list_brightkite
    obj2 = misc.misc(len(user_list))

    result_queue = multiprocessing.Queue()
    info = multiprocessing.Queue()
    info_split = multiprocessing.Queue()
    info_split_cnt = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=obj2.hyper_param,
                                 args=(env, user_list[:2], 'SARSA', 10, result_queue, info, info_split, info_split_cnt))
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(
    env, user_list[2:4], 'SARSA', 10, result_queue, info, info_split, info_split_cnt))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(
    env, user_list[4:6], 'SARSA', 10, result_queue, info, info_split, info_split_cnt))
    p4 = multiprocessing.Process(target=obj2.hyper_param,
                                 args=(env, user_list[6:], 'SARSA', 10, result_queue, info, info_split, info_split_cnt))

    split_final = np.zeros((4, 9), dtype=float)
    split_final_cnt = np.zeros((4, 9), dtype=float)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype=float)
    p1.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

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

    print("SARSA ", ", ".join(f"{x:.2f}" for x in final_result))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# FAA 
# u3 0.61, 0.63, 0.71, 0.37, 0.44, 0.61, 0.53, 0.74, 0.89
# u4 0.59, 0.55, 0.55, 0.52, 0.50, 0.70, 0.73, 0.65, 0.62
# u6 0.72, 0.57, 0.70, 0.71, 0.77, 0.72, 0.66, 0.33, 0.63
# u1 0.59, 0.54, 0.54, 0.57, 0.61, 0.63, 0.62, 0.74, 0.82
# u7 0.70, 0.67, 0.78, 0.68, 0.78, 0.66, 0.79, 0.55, 0.29
# u2 0.57, 0.52, 0.47, 0.50, 0.46, 0.61, 0.28, 0.16, 0.90
# u8 0.72, 0.69, 0.77, 0.72, 0.72, 0.65, 0.64, 0.46, 0.36
# u5 0.61, 0.78, 0.79, 0.82, 0.76, 0.82, 0.78, 0.88, 0.86

# SARSA  0.64, 0.62, 0.66, 0.61, 0.63, 0.68, 0.63, 0.56, 0.67

# Action  0 0.64, 0.60, 0.49, 0.59, 0.60, 0.41, 0.50, 0.62, 0.50
# Action  1 0.03, 0.03, 0.07, 0.03, 0.04, 0.05, 0.05, 0.01, 0.17
# Action  2 0.75, 0.69, 0.66, 0.59, 0.59, 0.58, 0.52, 0.48, 0.34
# Action  3 0.41, 0.45, 0.50, 0.47, 0.44, 0.65, 0.49, 0.21, 0.27

# Action  0 57.62, 56.38, 51.75, 45.12, 40.00, 34.12, 25.75, 17.50, 8.25
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.12, 106.25, 92.00, 78.12, 62.00, 43.88, 33.00, 25.50, 12.25
# Action  3 120.00, 108.25, 92.75, 78.88, 65.62, 55.12, 41.12, 23.25, 11.88

#Brightkite
# u14 0.41, 0.64, 0.66, 0.57, 0.67, 0.67, 0.66, 0.70, 0.93
# u15 0.51, 0.47, 0.56, 0.50, 0.59, 0.58, 0.57, 0.61, 0.76
# u16 0.59, 0.53, 0.55, 0.60, 0.70, 0.67, 0.68, 0.67, 0.82
# u10 0.65, 0.65, 0.75, 0.70, 0.46, 0.22, 0.82, 0.78, 0.60
# u13 0.82, 0.86, 0.82, 0.76, 0.73, 0.83, 0.90, 0.88, 0.89
# u9 0.68, 0.78, 0.78, 0.88, 0.88, 0.83, 0.89, 0.95, 0.89
# u11 0.74, 0.76, 0.78, 0.76, 0.82, 0.79, 0.90, 0.84, 0.94
# u12 0.79, 0.75, 0.86, 0.88, 0.67, 0.63, 0.87, 0.82, 0.66

# SARSA  0.65, 0.68, 0.72, 0.71, 0.69, 0.65, 0.79, 0.78, 0.81

# Action  0 0.91, 0.86, 0.90, 0.83, 0.90, 0.92, 0.86, 0.87, 0.71
# Action  1 0.02, 0.01, 0.02, 0.02, 0.01, 0.00, 0.01, 0.01, 0.02
# Action  2 0.51, 0.57, 0.74, 0.67, 0.63, 0.63, 0.76, 0.72, 0.47
# Action  3 0.36, 0.41, 0.32, 0.40, 0.28, 0.21, 0.42, 0.38, 0.16
# Action  0 211.38, 184.00, 162.38, 136.75, 105.12, 80.25, 61.00, 45.12, 27.75
# Action  1 9.12, 7.25, 6.00, 4.62, 3.88, 2.50, 2.12, 1.88, 1.12
# Action  2 147.75, 139.12, 122.75, 110.62, 101.50, 88.50, 66.25, 37.38, 16.00
# Action  3 115.62, 99.38, 84.75, 69.88, 57.38, 42.75, 30.75, 21.75, 7.25