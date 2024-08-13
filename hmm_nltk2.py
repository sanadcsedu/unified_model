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
import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    # user_list = env.user_list_brightkite
    final_results = np.zeros(9, dtype = float)
    final_cnt = np.zeros((4, 9), dtype = float)
    final_split_accu = np.zeros((4, 9), dtype = float)
        
    action_space = {'pan': 0, 'zoom': 1, 'brush': 2, 'range select': 3}

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        # excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')

        raw_file = [string for string in excel_files if user_name in string][0]

        accu = []
        obj = read_data()
        file = obj.merge(raw_file, feedback_file)
        data2 = []
        for line in file:
            z = ast.literal_eval(str(line))
            data2.append(z)

        sequences = [(row[2] + "-" + str(row[3]), row[1]) for row in data2]
        # pdb.set_trace()

        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        accu_split = [[] for _ in range(4)]
        cnt_split = [[] for _ in range(4)]
            
        for t in threshold:
            split = int(len(sequences) * t)
            predicted_tags = []
            true_tags = []
            states=[]
            split_accs = [[] for _ in range(4)]

            for i in range(split , len(sequences)):
                try:
                     trainer = nltk.HiddenMarkovModelTagger.train([sequences[:i]])
                     state, true_tag = sequences[i]
                     prediction = trainer.tag([state])
                     predicted_tag = prediction[0][1]
                    
                except ValueError:
                    continue

                predicted_tags.append(predicted_tag)
                true_tags.append(true_tag)
                states.append(state)
                
                if predicted_tag == true_tag:
                    split_accs[action_space[true_tag]].append(1)
                else:
                    split_accs[action_space[true_tag]].append(0)

            assert len(states) == len(true_tags) == len(predicted_tags)
            # print('States:', states)
            # print ('True Tags:', true_tags)
            # print ('Predicted Tags:', predicted_tags)
            
            #Calculate accuracy between predicted_tags and true_tags
            accuracy = np.mean(np.array(true_tags) == np.array(predicted_tags))
            for ii in range(4):
                if len(split_accs[ii]) > 0:
                    accu_split[ii].append(np.mean(split_accs[ii]))
                    cnt_split[ii].append(len(split_accs[ii]))
                else:
                    accu_split[ii].append(0)
                    cnt_split[ii].append(0)

            results.append(accuracy)

        print(user_name, ", ".join(f"{x:.2f}" for x in results))
        final_results = np.add(final_results, results)
        for ii in range(4):            
            final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
            final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

    final_results /= len(user_list)
    for ii in range(4):            
        final_split_accu[ii] /= len(user_list)
        final_cnt[ii] /= len(user_list)
    
    print('HMM', ", ".join(f"{x:.2f}" for x in final_results))
    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in final_split_accu[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in final_cnt[ii]))

# FAA
# u4 0.59, 0.55, 0.51, 0.48, 0.46, 0.51, 0.51, 0.56, 0.54
# u3 0.32, 0.36, 0.41, 0.48, 0.58, 0.70, 0.92, 0.94, 1.00
# u6 0.85, 0.83, 0.84, 0.82, 0.83, 0.79, 0.76, 0.64, 0.43
# u2 0.67, 0.62, 0.57, 0.50, 0.40, 0.25, 0.19, 0.16, 0.16
# u7 0.85, 0.83, 0.82, 0.79, 0.74, 0.70, 0.72, 0.64, 0.34
# u5 0.89, 0.87, 0.87, 0.88, 0.85, 0.81, 0.78, 0.91, 0.88
# u1 0.59, 0.58, 0.59, 0.57, 0.58, 0.52, 0.49, 0.71, 0.93
# u8 0.19, 0.21, 0.20, 0.19, 0.22, 0.28, 0.35, 0.48, 0.33

# HMM 0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.59, 0.63, 0.58

# Action  0 0.52, 0.52, 0.53, 0.52, 0.53, 0.47, 0.47, 0.56, 0.55
# Action  1 0.27, 0.28, 0.29, 0.31, 0.34, 0.34, 0.39, 0.47, 0.38
# Action  2 0.98, 0.98, 0.86, 0.86, 0.86, 0.84, 0.73, 0.72, 0.38
# Action  3 0.36, 0.35, 0.30, 0.25, 0.22, 0.20, 0.19, 0.12, 0.07

# Action  0 57.75, 56.50, 51.88, 45.25, 40.12, 34.25, 25.88, 17.62, 8.38
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.62, 106.75, 92.50, 78.62, 62.50, 44.38, 33.50, 26.00, 12.75
# Action  3 121.38, 109.62, 94.12, 80.25, 67.00, 56.50, 42.50, 24.62, 13.25

# Brightkite
# u13 0.88, 0.88, 0.89, 0.89, 0.88, 0.85, 0.89, 0.87, 0.88
# u16 0.56, 0.53, 0.49, 0.54, 0.60, 0.58, 0.52, 0.55, 0.74
# u9 0.72, 0.78, 0.81, 0.85, 0.93, 0.93, 0.93, 0.92, 0.84
# u10 0.34, 0.32, 0.29, 0.23, 0.26, 0.29, 0.24, 0.17, 0.31
# u11 0.82, 0.82, 0.80, 0.80, 0.76, 0.80, 0.79, 0.69, 0.40
# u12 0.86, 0.85, 0.85, 0.84, 0.81, 0.76, 0.77, 0.84, 0.73
# u14 0.64, 0.66, 0.74, 0.73, 0.76, 0.77, 0.77, 0.79, 1.00
# u15 0.48, 0.51, 0.53, 0.54, 0.58, 0.64, 0.57, 0.44, 0.36

# HMM 0.66, 0.67, 0.67, 0.68, 0.70, 0.70, 0.68, 0.66, 0.66

# Action  0 0.82, 0.82, 0.82, 0.79, 0.81, 0.83, 0.81, 0.82, 0.56
# Action  1 0.21, 0.22, 0.22, 0.26, 0.12, 0.14, 0.14, 0.12, 0.12
# Action  2 0.90, 0.90, 0.88, 0.88, 0.89, 0.88, 0.90, 0.83, 0.54
# Action  3 0.25, 0.24, 0.25, 0.27, 0.28, 0.25, 0.22, 0.19, 0.17

# Action  0 212.00, 184.62, 163.00, 137.38, 105.75, 80.88, 61.62, 45.75, 28.38
# Action  1 9.25, 7.38, 6.12, 4.75, 4.00, 2.62, 2.25, 2.00, 1.25
# Action  2 148.38, 139.75, 123.38, 111.25, 102.12, 89.12, 66.88, 38.00, 16.62
# Action  3 116.25, 100.00, 85.38, 70.50, 58.00, 43.38, 31.38, 22.38, 7.88
