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


#Class definition for the Actor-Critic model
class ActorCritic(nn.Module):
    def __init__(self,learning_rate,gamma):
        super(ActorCritic, self).__init__()
        # Class attributes
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Neural network architecture
        self.fc1 = nn.Linear(8, 128)
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
                    s_prime, r, done, info = self.env.step(s, a, False)
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
            done = False
            s = self.env.reset(all=False, test=True)
            predictions = []
            score=0
            # test = set()
            while not done:
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, pred = self.env.step(s, a, True)
                predictions.append(pred)
                
                model.put_data((s, a, r, s_prime, done))
                # test.add(a)
                s = s_prime

                score += r

                if done:
                    break
                model.train_net()
            # print(test)
            test_predictions.append(np.mean(predictions))
            # print("############ Test Accuracy :{},".format(np.mean(predictions)))
        return np.mean(test_predictions)

class run_ac:
    def __init__(self):
        pass

    def run_experiment(self, user_list, algo, hyperparam_file, result_queue):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        threshold_h = hyperparams['threshold']

        final_accu = np.zeros(9, dtype=float)
        # Loop over all users
        for feedback_file in user_list:

            user_name = self.get_user_name(feedback_file)
            # print(user_name)
            excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
            raw_file = [string for string in excel_files if user_name in string][0]

            accu = []
            env = environment5.environment5()
            # Loop over all threshold values
            for thres in threshold_h:
                max_accu = -1
                best_agent = None
                best_model = None

                env.process_data('faa', raw_file, feedback_file, thres, 'Actor-Critic')
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
                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_model
                    temp_accuracy = test_agent.test(test_model)
                    test_accs.append(temp_accuracy)
                test_accuracy = np.mean(test_accs)
                accu.append(test_accuracy)

            print(user_name, accu)
            final_accu = np.add(final_accu, accu)
        final_accu /= len(user_list)
        result_queue.put(final_accu)

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user


if __name__ == '__main__':
    env = environment5.environment5()
    user_list = env.user_list_faa
    # run_experiment(user_list, 'Actor_Critic', 'sampled_hyper_params.json') #user_list_faa contains names of the feedback files from where we parse user_name
    obj2 = run_ac()

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:2], 'Actor_Critic', 'sampled_hyper_params.json', result_queue,))
    p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[2:4], 'Actor_Critic', 'sampled_hyper_params.json', result_queue,))
    p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:6], 'Actor_Critic', 'sampled_hyper_params.json', result_queue,))
    p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[6:], 'Actor_Critic', 'sampled_hyper_params.json', result_queue,))
    
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
    print("Actor-Critic ", ", ".join(f"{x:.2f}" for x in final_result))