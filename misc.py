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
        self.main_states= self.get_states()
        self.prog = users * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h) * len(self.threshold_h)

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

    def get_states(self, task = 'faa'):
        if task == 'faa':
            vizs = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
            high_level_states = ['observation', 'generalization', 'question', 'hypothesis']
            states = []
            for v in vizs:
                for s in high_level_states:
                    str = v + '+' + s
                    states.append(str)
            return states             

    def hyper_param(self, env, users_hyper, algorithm, epoch):
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
        y_accu_all=[]
        for feedback_file in users_hyper:

            y_accu = []

            for thres in self.threshold_h:
                max_accu_thres = -1
                
                user_name = self.get_user_name(feedback_file)
                excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
                raw_file = [string for string in excel_files if user_name in string][0]

                env.process_data('faa', raw_file, feedback_file, thres, 'Qlearn') 
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for epiepi in range(pp):
                                if algorithm == 'QLearn':
                                    obj = Qlearning.Qlearning()
                                    Q, train_accuracy = obj.q_learning(env, epoch, dis, alp, eps)
                                    print(train_accuracy)
                                else:
                                    obj = SARSA.TD_SARSA()
                                    Q, train_accuracy = obj.sarsa(env, epoch, dis, alp, eps)
                                    print(train_accuracy)
                                if max_accu_thres < train_accuracy:
                                    max_accu_thres = train_accuracy
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                    best_q=Q
                                    best_obj=obj
                                max_accu_thres = max(max_accu_thres, train_accuracy)
                # print("Top Training Accuracy: {}, Threshold: {}".format(max_accu_thres, thres))
                test_accuracy = best_obj.test(env, best_q, best_discount, best_alpha, best_eps)
                # accuracy_per_state=self.format_split_accuracy(split_accuracy)

                print(
                    "Algorithm:{} , User:{}, Threshold: {}, Test Accuracy:{},  Epsilon:{}, Alpha:{}, Discount:{}, Split_Accuracy:{}".format(
                        algorithm,
                        user_name, thres, test_accuracy, best_eps, best_alpha,
                        best_discount))
                # print("Action choice: {}".format(Counter(stats)))

                ###move to new threshold:
                env.reset(True, False)

        #     plt.plot(self.threshold_h, y_accu, label=self.get_user_name(user), marker='*')
        # mean_y_accu = np.mean([element for sublist in y_accu_all for element in sublist])
        # plt.axhline(mean_y_accu, color='red', linestyle='--',label="Average: "+ "{:.2%}".format(mean_y_accu) )
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.yticks(np.arange(0.0, 1.0, 0.1))
        # plt.xlabel('Threshold')
        # plt.ylabel('Accuracy')
        # title = algorithm  + "all_3_actions"
        # # pdb.set_trace()
        # plt.title(title)
        # location = 'figures/' + title
        # plt.savefig(location, bbox_inches='tight')
        # result_dataframe.to_csv("Experiments_Folder\\" + title + ".csv", index=False)
        # plt.close()



