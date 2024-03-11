#contains all the miscellaneous functions for running 
import pandas as pd
import SARSA
import numpy as np
# import matplotlib.pyplot as plt 
import json
import Qlearning
from collections import Counter
from pathlib import Path
import glob
from tqdm import tqdm 
import os 
import multiprocessing

class misc:
    def __init__(self, users,hyperparam_file='sampled_hyper_params.json'):
        """
        Initializes the misc class.
        Parameters:
    - users: List of users
    - hyperparam_file: File path to the hyperparameters JSON file
    """
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        # Extract hyperparameters from JSON file
        self.discount_h =hyperparams['gammas']
        self.alpha_h = hyperparams['learning_rates']
        self.epsilon_h = hyperparams['epsilon']
        self.threshold_h = hyperparams['threshold']
        # self.main_states= self.get_states()

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

    def hyper_param(self, env, users_hyper, algorithm, epoch, result_queue):
        """
            Performs hyperparameter optimization.

            Parameters:
            - env: Environment object
            - users_hyper: List of user data
            - algorithm: Algorithm name ('QLearn' or 'SARSA')
            - epoch: Number of epochs

            Returns:
            None
            """
        # result_dataframe = pd.DataFrame(
        #     columns=['Algorithm','User','Epsilon', 'Threshold', 'LearningRate', 'Discount','Accuracy','StateAccuracy','Reward'])
        best_discount = best_alpha = best_eps = -1
        pp = 1
        final_accu = np.zeros(9, dtype=float)
        for feedback_file in users_hyper:

            accu = []
            for thres in self.threshold_h:
                max_accu_thres = -1
                
                user_name = self.get_user_name(feedback_file)
                excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
                raw_file = [string for string in excel_files if user_name in string][0]

                env.process_data('faa', raw_file, feedback_file, thres, algorithm) 
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for epiepi in range(pp):
                                if algorithm == 'Qlearn':
                                    obj = Qlearning.Qlearning()
                                    Q, train_accuracy = obj.q_learning(env, epoch, dis, alp, eps)
                                else:
                                    obj = SARSA.TD_SARSA()
                                    Q, train_accuracy = obj.sarsa(env, epoch, dis, alp, eps)
                                if max_accu_thres < train_accuracy:
                                    max_accu_thres = train_accuracy
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                    best_q=Q
                                    best_obj=obj
                                max_accu_thres = max(max_accu_thres, train_accuracy)
                test_accs = []
                test_env = env
                for _ in range(5):
                    test_model = best_obj
                    test_q, test_discount, test_alpha, test_eps = best_q, best_discount, best_alpha, best_eps
                    temp_accuracy = test_model.test(env, test_q, test_discount, test_alpha, test_eps)
                    test_accs.append(temp_accuracy)
                
                test_accuracy = np.mean(test_accs)
                # test_accuracy = best_obj.test(env, best_q, best_discount, best_alpha, best_eps)
                accu.append(test_accuracy)
                env.reset(True, False)
            # print(user[0], accu)
            # print(user[0], ", ".join(f"{x:.2f}" for x in accu))
            final_accu = np.add(final_accu, accu)
        final_accu /= len(users_hyper)
        # print(algorithm)
        # print(np.round(final_accu, decimals=2))
        result_queue.put(final_accu)