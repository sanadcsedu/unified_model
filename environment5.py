# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
from read_data import read_data
import numpy as np

class environment5:
    def __init__(self):
        path = os.getcwd()
        self.user_list_faa = glob.glob(path + '/FeedbackLog/*faa.xlsx')
        self.action_space = {'pan': 0, 'zoom': 1, 'brush': 2, 'range select': 3}
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0

    def reset(self, all=False, test=False):
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s


    def get_state(self, task, visualization, high_level_state, algo):
        if algo == 'Qlearn':
            state = visualization + '+' + high_level_state
        else: 
            state = np.zeros(9, dtype = np.int)
            if task == 'faa':
                vizs = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
            else:
                vizs = ['bar-5', 'hist-2', 'hist-3', 'hist-4', 'geo-0-1']
            for idx, v in enumerate(vizs):
                if v == visualization:
                    state[idx] = 1
                    break
            high_level_states = ['observation', 'generalization', 'question', 'hypothesis']
            idx = 0
            for idx, s in enumerate(high_level_states):
                if s == high_level_state:
                    state[5+idx] = 1
                    break
            return state

    def process_data(self, task, raw_file, feedback_file, thres, algo):
        # pdb.set_trace()
        obj = read_data()
        data = obj.merge(raw_file, feedback_file)
        # for index, row in df.iterrows():
        for d in data:
            state = self.get_state(task, d[2], d[3], algo)
            self.mem_states.append(state)
            self.mem_reward.append(d[4])
            # print(d[1])
            self.mem_action.append(self.action_space[d[1]])

        itrs = len(data)        
        self.threshold = int(itrs * thres)


    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test = False):
        if test:
            if len(self.mem_states) > self.steps + 3:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0
        else:
        # print(self.steps)
            if self.threshold > self.steps + 1:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0

    # predicted_action = action argument refers to action number
    def step(self, cur_state, pred_action, test = False):
        _, cur_reward, cur_action = self.cur_inter(self.steps)
        
        _, temp_step = self.peek_next_step()
        next_state, next_reward, next_action = self.cur_inter(temp_step)
        # print(cur_action, pred_action)
        if cur_action == pred_action: #check if the current action matches with the predicted action 
            prediction = 1
        else:
            prediction = 0
            # cur_reward = 0

        self.take_step_action(test)

        return next_state, cur_reward, self.done, prediction

