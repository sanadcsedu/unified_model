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
    user_list = env.user_list_faa
    final_results = np.zeros(9, dtype = float)

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]

        accu = []
        obj = read_data()
        file = obj.merge(raw_file, feedback_file)
        data2 = []
        for line in file:
            z = ast.literal_eval(str(line))
            data2.append(z)

        sequences = [(row[3] + "-" + str(row[4]), row[2]) for row in data2]

        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            split = int(len(sequences) * t)
            runs = []
            for k in range(10):
                trainer = nltk.HiddenMarkovModelTagger.train([sequences[:split]])
                accuracy = trainer.accuracy([sequences[split:]])
                runs.append(accuracy)
            results.append(np.mean(runs))
        final_results = np.add(final_results, results)

    final_results /= len(user_list)
    print('FAA', ", ".join(f"{x:.2f}" for x in final_results))

    # Brightkite 0.41, 0.39, 0.37, 0.35, 0.31, 0.28, 0.35, 0.21, 0.30
    # FAA 0.25, 0.31, 0.22, 0.29, 0.25, 0.22, 0.32, 0.25, 0.40
