import environment5
import numpy as np
from collections import defaultdict
import random
import misc
from collections import Counter
import pandas as pd
import multiprocessing
import json 
import glob 
import os 
from pathlib import Path

class WSLS:
    """
    WSLS (Win-Stay Lose-Switch) class implements the Win-Stay Lose-Switch algorithm for user modeling.
    """

    def __init__(self):
        
        self.bestaction = defaultdict(int)  
        self.reward = defaultdict(lambda: defaultdict(float)) 
        
    def take_random_action(self, cur_action):
        possible_actions = [action for action in [0, 1, 2] if action != cur_action]
        random_action = random.choice(possible_actions)
        return random_action

    def wslsDriver(self, env, thres):

        length = len(env.mem_action)
        threshold = int(length * thres)

        accuracy = 0
        denom = 0

        accuracy=[]
        
        for i in range(threshold, length):
            try:
                cur_action = self.bestaction[env.mem_states[i]]
                # print("DE ", cur_action, env.mem_states[i])
            except ValueError:
                cur_action = random.choice([0,1,2])
                self.reward[env.mem_states[i]][cur_action] = 0

            # print(env.mem_states[i], cur_action)
            if env.mem_reward[i] > self.reward[env.mem_states[i]][cur_action]:
                action = cur_action
                self.reward[env.mem_states[i]][action] = env.mem_reward[i]
                self.bestaction[env.mem_states[i]] = action
            else:
                # chnage from other actions in loose
                self.bestaction[env.mem_states[i]] = self.take_random_action(cur_action)
                # self.reward[env.mem_states[i]][action] = 0
    
            # performance book-keeping
            if self.bestaction[env.mem_states[i]] == env.mem_action[i]:
                accuracy.append(1)
            else:
                accuracy.append(0)
            denom += 1

        self.bestaction.clear()
        self.reward.clear()
        return np.mean(accuracy)

class run_wsls:
    def __inti__(self):
        pass

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user
    
    def run_experiment(self, user_list, hyperparam_file, result_queue):
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        threshold = hyperparams['threshold']

        # Create result DataFrame with columns for relevant statistics
        final_accu = np.zeros(9, dtype=float)
        for feedback_file in user_list:
            user_name = self.get_user_name(feedback_file)
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            raw_file = [string for string in excel_files if user_name in string][0]
            accu = []
            for thres in threshold:
                avg_accu = []
                for _ in range(5):
                    env.process_data('faa', raw_file, feedback_file, thres, 'WSLS')
                    # print(env.mem_states)
                    obj = WSLS()
                    avg_accu.append(obj.wslsDriver(env, thres))
                    env.reset(True, False)
                accu.append(np.mean(avg_accu))
            final_accu = np.add(final_accu, accu)
        
        final_accu /= len(user_list)
        result_queue.put(final_accu)

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    # run_experiment(user_list, 'Actor_Critic', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name
    obj2 = run_wsls()

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'sampled_hyper_params.json', result_queue,))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'sampled_hyper_params.json', result_queue,))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'sampled_hyper_params.json', result_queue,))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'sampled_hyper_params.json', result_queue,))
    
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
    print("WSLS ", ", ".join(f"{x:.2f}" for x in final_result))