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

eps=1e-35
class Greedy:
    def __init__(self):
        """Initializes the Greedy model."""
        # self.freq = defaultdict(lambda: defaultdict(float))
        self.reward = defaultdict(lambda: defaultdict(float))

    def GreedyDriver(self, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(threshold):
            # self.freq[env.mem_states[i]][env.mem_action[i]] += 1
            self.reward[env.mem_states[i]][env.mem_action[i]] += env.mem_reward[i]+eps

        # Normalizing
        for states in self.reward:
            sum = 0
            for actions in self.reward[states]:
                sum += self.reward[states][actions]
            for actions in self.reward[states]:
                self.reward[states][actions] = self.reward[states][actions] / sum
        
        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        insight = defaultdict(list)

        for i in range(threshold, length):
            denom += 1
            try: #Finding the most rewarding action in the current state
                _max = max(self.reward[env.mem_states[i]], key=self.reward[env.mem_states[i]].get)
            except ValueError: #Randomly picking an action if it was used previously in current state 
                _max= random.choice([0, 1, 2, 3])

            if _max == env.mem_action[i]:
                accuracy += 1
                self.reward[env.mem_states[i]][_max] += env.mem_reward[i]
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)

        accuracy /= denom
        self.reward.clear()
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return accuracy, granular_prediction

class run_Greedy:
    def __inti__(self):
        pass

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

    def run_experiment(self, user_list, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        threshold = hyperparams['threshold']

        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((4, 9), dtype = float)
        final_split_accu = np.zeros((4, 9), dtype = float)
        
        for feedback_file in user_list:
            user_name = self.get_user_name(feedback_file)
            # excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')
            raw_file = [string for string in excel_files if user_name in string][0]

            accu = []
            accu_split = [[] for _ in range(4)]
            cnt_split = [[] for _ in range(4)]
            
            for thres in threshold:
                avg_accu = []
                split_accs = [[] for _ in range(4)]

                for _ in range(5):
                    # env.process_data('faa', raw_file, feedback_file, thres, 'Greedy')
                    env.process_data('brightkite', raw_file, feedback_file, thres, 'Greedy') 
                    obj = Greedy()
                    temp_accuracy, gp = obj.GreedyDriver(env, thres)
                    avg_accu.append(temp_accuracy)
                    env.reset(True, False)

                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                    
                accu.append(np.mean(avg_accu))
                for ii in range(4):
                    if len(split_accs[ii]) > 0:
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)

            print(user_name, ", ".join(f"{x:.2f}" for x in accu))
            final_accu = np.add(final_accu, accu)
            for ii in range(4):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(4):            
            final_split_accu[ii] /= len(user_list)
            final_cnt[ii] /= len(user_list)
        
        # print(final_accu)
        
        result_queue.put(final_accu)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)


if __name__ == "__main__":
    final_output = []
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    obj2 = run_Greedy()

    result_queue = multiprocessing.Queue()
    info = multiprocessing.Queue()
    info_split = multiprocessing.Queue()
    info_split_cnt = multiprocessing.Queue() 

    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    
    split_final = np.zeros((4, 9), dtype = float)
    split_final_cnt = np.zeros((4, 9), dtype = float)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype = float)
    p1.join()
    # final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
    # print(split_final_cnt)
    p2.join()
    # final_output.extend(info.get())
    # print(final_result)
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p3.join()
    # print(final_result)
    # final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p4.join()
    # final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    final_result /= 4
    split_final /= 4
    split_final_cnt /= 4

    print("Greedy ", ", ".join(f"{x:.2f}" for x in final_result))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# FAA 
# u1 0.54, 0.53, 0.55, 0.57, 0.58, 0.52, 0.49, 0.59, 0.93
# u6 0.81, 0.79, 0.79, 0.79, 0.80, 0.79, 0.81, 0.67, 0.71
# u7 0.73, 0.73, 0.72, 0.71, 0.66, 0.57, 0.65, 0.56, 0.35
# u4 0.60, 0.57, 0.54, 0.51, 0.67, 0.70, 0.71, 0.64, 0.64
# u8 0.56, 0.54, 0.64, 0.60, 0.76, 0.71, 0.66, 0.52, 0.40
# u5 0.89, 0.88, 0.90, 0.89, 0.88, 0.85, 0.84, 0.91, 0.86
# u3 0.64, 0.65, 0.60, 0.54, 0.45, 0.41, 0.25, 0.03, 0.00
# u2 0.63, 0.60, 0.56, 0.50, 0.38, 0.29, 0.17, 0.13, 0.03

# Greedy  0.68, 0.66, 0.66, 0.64, 0.65, 0.60, 0.57, 0.51, 0.49

# Action  0 0.40, 0.39, 0.43, 0.41, 0.34, 0.44, 0.37, 0.50, 0.45
# Action  1 0.02, 0.06, 0.08, 0.08, 0.08, 0.06, 0.01, 0.00, 0.00
# Action  2 0.81, 0.82, 0.61, 0.60, 0.59, 0.52, 0.41, 0.38, 0.25
# Action  3 0.61, 0.61, 0.62, 0.60, 0.73, 0.67, 0.74, 0.56, 0.27

# Action  0 57.75, 56.50, 51.88, 45.25, 40.12, 34.25, 25.88, 17.62, 8.38
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.62, 106.75, 92.50, 78.62, 62.50, 44.38, 33.50, 26.00, 12.75
# Action  3 121.38, 109.62, 94.12, 80.25, 67.00, 56.50, 42.50, 24.62, 13.25

# Brightkite
# u14 0.40, 0.36, 0.31, 0.57, 0.62, 0.77, 0.77, 0.79, 1.00
# u13 0.79, 0.80, 0.81, 0.79, 0.74, 0.81, 0.87, 0.84, 0.88
# u9 0.67, 0.73, 0.76, 0.83, 0.92, 0.92, 0.90, 0.91, 0.84
# u11 0.81, 0.81, 0.81, 0.81, 0.78, 0.80, 0.81, 0.71, 0.44
# u15 0.65, 0.65, 0.68, 0.69, 0.67, 0.71, 0.65, 0.65, 0.79
# u10 0.76, 0.80, 0.85, 0.77, 0.80, 0.81, 0.86, 0.84, 0.64
# u12 0.84, 0.83, 0.83, 0.81, 0.79, 0.72, 0.93, 0.90, 0.91
# u16 0.62, 0.58, 0.58, 0.64, 0.69, 0.66, 0.58, 0.53, 0.89

# Greedy  0.69, 0.70, 0.70, 0.74, 0.75, 0.78, 0.80, 0.77, 0.80

# Action  0 0.90, 0.86, 0.86, 0.86, 0.86, 0.98, 0.98, 0.96, 0.74
# Action  1 0.00, 0.01, 0.00, 0.01, 0.01, 0.00, 0.01, 0.00, 0.00
# Action  2 0.67, 0.67, 0.72, 0.81, 0.82, 0.81, 0.81, 0.80, 0.52
# Action  3 0.25, 0.28, 0.35, 0.30, 0.30, 0.26, 0.23, 0.18, 0.09

# Action  0 212.00, 184.62, 163.00, 137.38, 105.75, 80.88, 61.62, 45.75, 28.38
# Action  1 9.25, 7.38, 6.12, 4.75, 4.00, 2.62, 2.25, 2.00, 1.25
# Action  2 148.38, 139.75, 123.38, 111.25, 102.12, 89.12, 66.88, 38.00, 16.62
# Action  3 116.25, 100.00, 85.38, 70.50, 58.00, 43.38, 31.38, 22.38, 7.88