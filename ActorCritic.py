import environment5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import plotting
from collections import Counter
import pandas as pd
import json
import os
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
import glob
from tqdm import tqdm 
import multiprocessing
import pdb 
import itertools


#Class definition for the Actor-Critic model
class ActorCritic(nn.Module):
    def __init__(self,learning_rate,gamma):
        super(ActorCritic, self).__init__()
        # Class attributes
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Neural network architecture
        self.fc1 = nn.Linear(9, 128)
        self.fc_pi = nn.Linear(128, 4)#actor
        self.fc_v = nn.Linear(128, 1)#critic

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #The critic network (called self.fc_v in the code) estimates the state value and is trained using the TD error to minimize the difference between the predicted and actual return.
    def pi(self, x, softmax_dim=0):
        """
        Compute the action probabilities using the policy network.

        Args:
            x (torch.Tensor): State tensor.
            softmax_dim (int): Dimension along which to apply the softmax function (default=0).

        Returns:
            prob (torch.Tensor): Tensor with the action probabilities.
        """

        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    #The actor network (called self.fc_pi ) outputs the action probabilities and is trained using the policy gradient method to maximize the expected return.
    def v(self, x):
        """
        Compute the state value using the value network.

        Args:
            x (torch.Tensor): State tensor.

        Returns:
            v (torch.Tensor): Tensor with the estimated state value.
        """

        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        """Add a transition tuple to the data buffer.
        Args:transition (tuple): Tuple with the transition data (s, a, r, s_prime, done)."""

        self.data.append(transition)

    def make_batch(self):
        """
        Generate a batch of training data from the data buffer.

        Returns:
            s_batch, a_batch, r_batch, s_prime_batch, done_batch (torch.Tensor): Tensors with the
                states, actions, rewards, next states, and done flags for the transitions in the batch.
        """

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

            s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst)), \
                                                               torch.tensor(np.array(r_lst), dtype=torch.float), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                                               torch.tensor(np.array(done_lst), dtype=torch.float)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        """
           Train the Actor-Critic model using a batch of training data.
           """
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)

        #The first term is the policy loss, which is computed as the negative log probability of the action taken multiplied by the advantage
        # (i.e., the difference between the estimated value and the target value).
        # The second term is the value loss, which is computed as the mean squared error between the estimated value and the target value
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

class Agent():
    def __init__(self, env,learning_rate,gamma,num_rollouts=10):
        self.env = env
        self.learning_rate, self.gamma, self.n_rollout=learning_rate,gamma,num_rollouts

    def train(self):
        model = ActorCritic(self.learning_rate, self.gamma)
        score = 0.0
        all_predictions = []
        for _ in range(5):
            done = False
            s = self.env.reset(all = False, test = False)

            predictions = []
            actions = []
            while not done:
                for t in range(self.n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    actions.append(a)
                    s_prime, r, done, info, _ = self.env.step(s, a, False)
                    predictions.append(info)
                   
                    model.put_data((s, a, r, s_prime, done))

                    s = s_prime

                    score += r

                    if done:
                        break
                #train at the end of the episode: batch will contain all the transitions from the n-steps
                model.train_net()

            score = 0.0
            all_predictions.append(np.mean(predictions))
        # print("############ Train Accuracy :{:.2f},".format(np.mean(all_predictions)))
        return model, np.mean(predictions)  # return last episodes accuracyas training accuracy


    def test(self,model):

        test_predictions = []
        for _ in range(1):
            # done = False
            s = self.env.reset(all=False, test=True)
            predictions = []
            insight = defaultdict(list)

            score=0
            # test = set()
            # print("len ", len(self.env.mem_states))
            for t in itertools.count():
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, pred, ground_action = self.env.step(s, a, True)
                # print(self.env.steps)
                predictions.append(pred)
                insight[ground_action].append(pred)
                model.put_data((s, a, r, s_prime, done))
                # test.add(a)
                s = s_prime

                score += r

                if done:
                    break
                model.train_net()
            test_predictions.append(np.mean(predictions))
        
        # Calculating the number of occurance and prediction rate for each action
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(test_predictions), granular_prediction

class run_ac:
    def __init__(self):
        pass

    def run_experiment(self, user_list, algo, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        threshold_h = hyperparams['threshold']
        output_list = []

        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((4, 9), dtype = float)
        final_split_accu = np.zeros((4, 9), dtype = float)
        
        # Loop over all users
        for feedback_file in user_list:

            user_name = self.get_user_name(feedback_file)
            # print(user_name)
            # excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')            

            raw_file = [string for string in excel_files if user_name in string][0]

            accu = []
            accu_split = [[] for _ in range(4)]
            cnt_split = [[] for _ in range(4)]
            
            output_list.append(f"user {user_name}")
            
            env = environment5.environment5()
            # Loop over all threshold values
            for thres in threshold_h:
                max_accu = -1
                best_agent = None
                best_model = None

                # env.process_data('faa', raw_file, feedback_file, thres, 'Actor-Critic')
                env.process_data('brightkite', raw_file, feedback_file, thres, 'Actor-Critic') 

                # Loop over all combinations of hyperparameters
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        agent = Agent(env, learning_rate, gamma)
                        model, accuracies = agent.train()

                        # Keep track of best combination of hyperparameters
                        if accuracies > max_accu:
                            max_accu = accuracies
                            best_agent = agent
                            best_model = model

                #running them 5 times and taking the average test accuracy to reduce fluctuations
                test_accs = []
                split_accs = [[] for _ in range(4)]
                
                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_model
                    temp_accuracy, gp = test_agent.test(test_model)
                    test_accs.append(temp_accuracy)

                    for key, val in gp.items():
                        # print(key, val)
                        split_accs[key].append(val[1])
                
                test_accuracy = np.mean(test_accs)
                accu.append(test_accuracy)
                env.reset(True, False)


                output_list.append(f"Threshold: {thres}")
                for ii in range(4):
                    if len(split_accs[ii]) > 0:
                        # print("action: {}, count: {}, accuracy:{}".format(ii, gp[ii][0], np.mean(split_accs[ii])))
                        output_list.append(f"action: {ii}, count: {gp[ii][0]}, accuracy:{np.mean(split_accs[ii])}")
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        # print("{} Not Present".format(ii))
                        output_list.append(f"{ii} Not Present")
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)

            # print(user_name, accu)
            print(user_name, ", ".join(f"{x:.2f}" for x in accu))
            output_list.append(f"{user_name}, {', '.join(f'{x:.2f}' for x in accu)}")

            final_accu = np.add(final_accu, accu)
            for ii in range(4):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(4):            
            final_split_accu[ii] /= len(user_list)
            final_cnt[ii] /= len(user_list)
        
        result_queue.put(final_accu)
        info.put(output_list)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user


if __name__ == '__main__':
    final_output = []
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite

    # run_experiment(user_list, 'Actor_Critic', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name
    obj2 = run_ac()

    result_queue = multiprocessing.Queue()
    info = multiprocessing.Queue()
    info_split = multiprocessing.Queue()
    info_split_cnt = multiprocessing.Queue() 

    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'Actor_Critic', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'Actor_Critic', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'Actor_Critic', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'Actor_Critic', 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
    
    split_final = np.zeros((4, 9), dtype = float)
    split_final_cnt = np.zeros((4, 9), dtype = float)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype = float)
    p1.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
    # print(split_final_cnt)
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

    print("Actor-Critic ", ", ".join(f"{x:.2f}" for x in final_result))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# FAA DATASET

# u6 0.78, 0.60, 0.81, 0.80, 0.88, 0.65, 0.84, 0.63, 0.68
# u4 0.60, 0.43, 0.71, 0.61, 0.71, 0.73, 0.76, 0.68, 0.68
# u7 0.54, 0.41, 0.72, 0.78, 0.40, 0.75, 0.78, 0.70, 0.68
# u3 0.65, 0.62, 0.57, 0.47, 0.63, 0.64, 0.42, 0.64, 0.00
# u1 0.36, 0.53, 0.44, 0.55, 0.62, 0.59, 0.54, 0.71, 0.82
# u2 0.67, 0.48, 0.56, 0.38, 0.45, 0.45, 0.49, 0.15, 0.07
# u5 0.90, 0.86, 0.90, 0.92, 0.90, 0.78, 0.84, 0.91, 0.85
# u8 0.83, 0.85, 0.83, 0.81, 0.75, 0.71, 0.64, 0.44, 0.59

# Actor-Critic  0.67, 0.60, 0.69, 0.66, 0.67, 0.66, 0.66, 0.61, 0.55

# Action  0 0.27, 0.23, 0.20, 0.17, 0.29, 0.27, 0.34, 0.38, 0.35
# Action  1 0.03, 0.01, 0.02, 0.00, 0.02, 0.03, 0.00, 0.05, 0.00
# Action  2 0.59, 0.49, 0.62, 0.56, 0.64, 0.47, 0.38, 0.49, 0.24
# Action  3 0.47, 0.55, 0.71, 0.73, 0.64, 0.53, 0.67, 0.47, 0.36
# Action  0 57.62, 56.38, 51.75, 45.12, 40.00, 34.12, 25.75, 17.50, 8.25
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.12, 106.25, 92.00, 78.12, 62.00, 43.88, 33.00, 25.50, 12.25
# Action  3 120.00, 108.25, 92.75, 78.88, 65.62, 55.12, 41.12, 23.25, 11.88

# Brightkite Dataset

# u14 0.49, 0.36, 0.63, 0.64, 0.61, 0.75, 0.75, 0.78, 0.99
# u11 0.15, 0.81, 0.63, 0.80, 0.74, 0.75, 0.80, 0.49, 0.51
# u9 0.61, 0.72, 0.61, 0.87, 0.96, 0.98, 0.97, 0.98, 0.96
# u15 0.44, 0.37, 0.62, 0.69, 0.74, 0.74, 0.64, 0.74, 0.77
# u13 0.66, 0.32, 0.80, 0.80, 0.88, 0.85, 0.92, 0.90, 0.91
# u10 0.58, 0.83, 0.85, 0.75, 0.68, 0.87, 0.84, 0.73, 0.55
# u12 0.94, 0.94, 0.95, 0.96, 0.94, 0.72, 0.93, 0.87, 0.88
# u16 0.55, 0.56, 0.42, 0.61, 0.67, 0.63, 0.57, 0.51, 0.88

# Actor-Critic  0.55, 0.61, 0.69, 0.76, 0.77, 0.79, 0.80, 0.75, 0.81

# Action  0 0.75, 0.64, 0.58, 0.90, 0.84, 0.90, 0.94, 0.81, 0.72
# Action  1 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  2 0.49, 0.74, 0.59, 0.78, 0.76, 0.80, 0.89, 0.64, 0.62
# Action  3 0.15, 0.19, 0.42, 0.22, 0.28, 0.18, 0.15, 0.30, 0.04
# Action  0 211.38, 184.00, 162.38, 136.75, 105.12, 80.25, 61.00, 45.12, 27.75
# Action  1 9.12, 7.25, 6.00, 4.62, 3.88, 2.50, 2.12, 1.88, 1.12
# Action  2 147.75, 139.12, 122.75, 110.62, 101.50, 88.50, 66.25, 37.38, 16.00
# Action  3 115.62, 99.38, 84.75, 69.88, 57.38, 42.75, 30.75, 21.75, 7.25
