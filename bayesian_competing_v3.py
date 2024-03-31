import pandas as pd
import ast
import numpy as np 
import itertools
from sklearn.preprocessing import LabelEncoder
import pdb
import random 
from itertools import combinations
import random
import environment5 
from pathlib import Path
import glob
import os
from read_data import read_data

class ExtendedCategoricalModel:
    def __init__(self, data, action_var_name, context_var_names, alpha):
        self.data = data
        self.action_var_name = action_var_name
        self.context_var_names = context_var_names
        self.alpha = alpha

        self.actions = ['pan', 'zoom', 'brush', 'range select']
        self.context_categories = {var_name: list(data[var_name].unique()) for var_name in context_var_names}
        # pdb.set_trace()
        self.counts = self.init_counts()
        self.probabilities = self.update_probabilities()

    def init_counts(self):
        # Initialize counts for each combination of context and action to alpha for smoothing
        counts = {}
        for context_combination in itertools.product(*[self.context_categories[var_name] for var_name in self.context_var_names]):
            counts[context_combination] = {}
            for action in self.actions:
                counts[context_combination][action] = self.alpha
        return counts

    def update_probabilities(self):
        # Update probabilities based on counts
        probabilities = {}
        for context_combination, actions in self.counts.items():
            probabilities[context_combination] = {}
            for action, count in actions.items():
                total_count = sum(self.counts[context_combination][a] for a in self.actions)
                probabilities[context_combination][action] = count / total_count
        
        return probabilities

    def update(self, observation):
        # Update counts and probabilities based on a new observation
        action = observation[self.action_var_name]
        context_combination = tuple(observation[var_name] for var_name in self.context_var_names)
        self.counts[context_combination][action] += 1
        self.probabilities = self.update_probabilities()

    def predict_next_action(self, current_context):
        context_combination = tuple(current_context[var_name] for var_name in self.context_var_names)
        # Check if the context combination exists in our probabilities
        if context_combination in self.probabilities:
            action_probabilities = self.probabilities[context_combination]
        else:
            # If the context_combination is not present, initialize probabilities to uniform distribution
            return random.choice(['pan', 'zoom', 'brush', 'range select'])    
        if random.random() < action_probabilities[max(action_probabilities, key=action_probabilities.get)]:
            return max(action_probabilities, key=action_probabilities.get)
        else:
            return random.choice(['pan', 'zoom', 'brush', 'range select'])

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

if __name__ == "__main__":
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    final_results = np.zeros(9, dtype = float)

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]

        accu = []
        obj = read_data()
        file = obj.merge(raw_file, feedback_file)
        data2 = []
        for line in file:
            z = ast.literal_eval(str(line))
            data2.append(z)

        data = pd.DataFrame(data2, columns=['id', 'time', 'action', 'visualization', 'state', 'reward'])

        model = ExtendedCategoricalModel(data, 'action', ['visualization', 'state'], alpha=1)
        # Update the model with observations (you'd loop through your observations to update)
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            accu = []
            split = int(len(data) * t)
            for idx in range(split):
                model.update(data.iloc[idx])
            for idx in range(split, len(data)):
                predicted_action = model.predict_next_action(data.iloc[idx])
                if predicted_action == data['action'][idx]:
                    accu.append(1)
                else:
                    accu.append(0)
            results.append(np.mean(accu))
        final_results = np.add(final_results, results)

    final_results /= len(user_list)
    print('FAA', ", ".join(f"{x:.2f}" for x in final_results))