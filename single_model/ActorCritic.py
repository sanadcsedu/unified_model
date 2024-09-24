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
    def __init__(self,learning_rate,gamma, dataset):
        super(ActorCritic, self).__init__()
        # Class attributes
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Neural network architecture
        if dataset == 'faa':
            self.fc1 = nn.Linear(8, 128)
        else:
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
            v (torch.Tensor): Tensor with the estimated state value.        """

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

    def train(self, model):
        # model = ActorCritic(self.learning_rate, self.gamma)
        score = 0.0
        all_predictions = []
        for _ in range(5):
            done = False
            s = self.env.reset()

            predictions = []
            while not done:
                for t in range(self.n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
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
        for _ in range(10):
            # done = False
            s = self.env.reset()
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
            granular_prediction[keys] = (len(values), np.sum(values))

        return np.mean(test_predictions), granular_prediction

def get_user_name(raw_fname):
    user = Path(raw_fname).stem.split('-')[0]
    return user

def training(dataset, train_files, epoch, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    # Load hyperparameters from JSON file
    hyperparam_file='sampled_hyper_params.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
        
    best_lr = best_gamma = max_accu = -1
    for lr in learning_rates:
        for ga in gammas:
            accu = []
            model = ActorCritic(lr, ga, dataset)

            for feedback_file in train_files:
                user_name = get_user_name(feedback_file)
                if dataset == 'faa':
                    excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
                else:
                    excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
                raw_file = [string for string in excel_files if user_name in string][0]

                env = environment5.environment5()
                if dataset == 'faa':
                    env.process_data('faa', raw_file, feedback_file, algorithm) 
                else:
                    env.process_data('brightkite', raw_file, feedback_file, algorithm) 
                    
                agent = Agent(env, lr, ga, epoch)           
                        
                model, accu_user = agent.train(model)
                accu.append(accu_user)
                   
            #accuracy of the model learned over training data
            accu_model = np.mean(accu)
            if accu_model > max_accu:
                max_accu = accu_model
                best_lr = lr
                best_gamma = ga
                best_ac_model = model

    # print("Training Accuracy", max_accu)
    return best_ac_model, best_lr, best_gamma, max_accu

def testing(dataset, test_files, trained_ac_model, best_lr, best_gamma, algorithm):
    child = os.getcwd()
    path = os.path.dirname(child) #get parent directory  

    final_accu = []
    for feedback_file in test_files:
        user_name = get_user_name(feedback_file)
        if dataset == 'faa':
            excel_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        else:
            excel_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')            
        raw_file = [string for string in excel_files if user_name in string][0]

        env = environment5.environment5()
        if dataset == 'faa':
            env.process_data('faa', raw_file, feedback_file, algorithm) 
        else:
            env.process_data('brightkite', raw_file, feedback_file, algorithm) 
        
        agent = Agent(env, best_lr, best_gamma)           
        accu, gp = agent.test(trained_ac_model)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    
    return np.mean(final_accu), gp

if __name__ == '__main__':
    datasets = ['brightkite', 'faa']
    for d in datasets:
        print("Dataset ", d)   
        split_accs = [[] for _ in range(4)]
        split_cnt = [[] for _ in range(4)]      
        env = environment5.environment5()
        
        if d == 'faa':
            user_list = env.user_list_faa
        else:
            user_list = env.user_list_brightkite

        accuracies = []
        X_train = []
        X_test = []

        # Leave-One-Out Cross-Validation
        for i, test_user_log in enumerate((user_list)):
            train_files = user_list[:i] + user_list[i+1:]  # All users except the ith one
            # train_files, test_files = train_test_split(user_list, test_size=0.3, random_state=42)
            trained_ac_model, best_lr, best_gamma, training_accuracy = training(d, train_files, 5, 'Actor-Critic')
            X_train.append(training_accuracy)
            # test user
            test_files = [test_user_log]
            testing_accu, gp = testing(d, test_files, trained_ac_model, best_lr, best_gamma, 'Actor-Critic')
            
            # print("Testing Accuracy ", accu)
            for key, val in gp.items():
                # print(key, val)
                split_accs[key].append(val[1])
                split_cnt[key].append(val[0])

            X_test.append(testing_accu)
            # accuracies.append(accu)

        # train_accu = np.mean(X_train)
        test_accu = np.mean(X_test)
        # print("Actor-Critic Training {:.2f}".format(train_accu))
        print("Actor-Critic Testing {:.2f}".format(test_accu))

        for i in range(4):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 