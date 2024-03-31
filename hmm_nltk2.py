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
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    final_results = np.zeros(9, dtype = float)

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        # excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')

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
        for t in threshold:
            split = int(len(sequences) * t)
            predicted_tags = []
            true_tags = []
            states=[]
            for i in range(split , len(sequences)):
                try:
                     trainer = nltk.HiddenMarkovModelTagger.train([sequences[:i]])
                     state, true_tag = sequences[i]
                     prediction = trainer.tag([state])
                     predicted_tag = prediction[0][1]
                    
                except ValueError:
                    # print('Value Error')
                    continue

                predicted_tags.append(predicted_tag)
                true_tags.append(true_tag)
                states.append(state)

            assert len(states) == len(true_tags) == len(predicted_tags)
            # print('States:', states)
            # print ('True Tags:', true_tags)
            # print ('Predicted Tags:', predicted_tags)
            # pdb.set_trace()
            #Calculate accuracy between predicted_tags and true_tags
            accuracy = np.mean(np.array(true_tags) == np.array(predicted_tags))
            # print(accuracy)
            results.append(accuracy)
        final_results = np.add(final_results, results)

    final_results /= len(user_list)
    print('Brightkite', ", ".join(f"{x:.2f}" for x in final_results))