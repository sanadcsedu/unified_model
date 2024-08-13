import environment5 as environment5
import misc
from pathlib import Path
import glob
import os 
from read_data2 import read_data
import pandas as pd 
from collections import defaultdict

def get_user_name(raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

#  The following Calculates the Ratio of Picking an Action (Data Focus Shifts) NOT PROBABILITY
if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    obj2 = misc.misc(user_list)
    new_data = []
    A = ['same-scatterplot-0-1', 'same-bar-2', 'same-bar-4', 'same-hist-3', 'modify-scatterplot-0-1', 'modify-bar-2', 'modify-bar-4', 'modify-hist-3']
    for feedback_file in user_list:
        print(feedback_file)
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]
        obj = read_data()
        data = obj.merge(raw_file, feedback_file)
        
        actions_count = defaultdict(int)
        actions_prob = defaultdict(int)
        split_time = int((data[-1][1] - data[0][1])/2)
        denominator = round(split_time / 60, 2)
        print(denominator)
        
        idx = 0
        for _ in iter(int, 1):
            actions_count[data[idx][6]] += 1
            if data[idx][1] > split_time:
                phase = 'First'
                # print('First')
                for keys, values in actions_count.items():
                    # print(keys, values)
                    actions_prob[keys] = round(actions_count[keys] / denominator, 2)
                break
            idx += 1
        entry = [user_name, 'FAA']
        for a in A:
            entry.append(actions_prob[a])
        entry.append(phase)
        print(entry)
        new_data.append(entry)

        actions_count.clear()
        actions_prob.clear()
        # denominator = 0
        split_time = int((data[-1][1] - data[0][1])/2)
        for i in range(idx, len(data)-1):            
            # denominator += 1
            actions_count[data[i][6]] += 1
            
        phase = 'Second'
        for keys, values in actions_count.items():
            # print(keys, values)
            actions_prob[keys] = round(actions_count[keys] / denominator, 2)
        # for keys, values in actions_prob.items():
            # print(keys, values)
        entry = [user_name, 'FAA']
        for a in A:
            entry.append(actions_prob[a])
        entry.append(phase)
        print(entry)
        new_data.append(entry)

    df = pd.DataFrame(new_data, columns = ['User', 'Dataset', 'same-scatterplot-0-1', 'same-bar-2', 'same-bar-4', 'same-hist-3', 'modify-scatterplot-0-1', 'modify-bar-2', 'modify-bar-4', 'modify-hist-3', 'Phase'])
    # print(df)
    df.to_csv('FAA_ratio.csv', index=False)
             

# # The following Calculates the Probability of Picking an Action (Data Focus Shifts)
# if __name__ == "__main__":
#     env = environment5.environment5()
#     user_list = env.user_list_faa
#     obj2 = misc.misc(user_list)
#     new_data = []
#     A = ['same-scatterplot-0-1', 'same-bar-2', 'same-bar-4', 'same-hist-3', 'modify-scatterplot-0-1', 'modify-bar-2', 'modify-bar-4', 'modify-hist-3']
#     for feedback_file in user_list:
#         print(feedback_file)
#         user_name = get_user_name(feedback_file)
#         excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
#         raw_file = [string for string in excel_files if user_name in string][0]
#         obj = read_data()
#         data = obj.merge(raw_file, feedback_file)
        
#         actions_count = defaultdict(int)
#         actions_prob = defaultdict(int)
#         denominator = 0
#         split_time = int((data[-1][1] - data[0][1])/2)
#         idx = 0
#         for _ in iter(int, 1):
#             denominator += 1
#             actions_count[data[idx][6]] += 1
#             if data[idx][1] > split_time:
#                 phase = 'First'
#                 # print('First')
#                 for keys, values in actions_count.items():
#                     # print(keys, values)
#                     actions_prob[keys] = round(actions_count[keys] / denominator, 2)
#                 break
#             idx += 1
#         entry = [user_name, 'FAA']
#         for a in A:
#             entry.append(actions_prob[a])
#         entry.append(phase)
#         print(entry)
#         new_data.append(entry)

#         actions_count.clear()
#         actions_prob.clear()
#         denominator = 0
#         split_time = int((data[-1][1] - data[0][1])/2)
#         for i in range(idx, len(data)-1):            
#             denominator += 1
#             actions_count[data[i][6]] += 1
            
#         phase = 'Second'
#         for keys, values in actions_count.items():
#             # print(keys, values)
#             actions_prob[keys] = round(actions_count[keys] / denominator, 2)
#         # for keys, values in actions_prob.items():
#             # print(keys, values)
#         entry = [user_name, 'FAA']
#         for a in A:
#             entry.append(actions_prob[a])
#         entry.append(phase)
#         print(entry)
#         new_data.append(entry)

#     df = pd.DataFrame(new_data, columns = ['User', 'Dataset', 'same-scatterplot-0-1', 'same-bar-2', 'same-bar-4', 'same-hist-3', 'modify-scatterplot-0-1', 'modify-bar-2', 'modify-bar-4', 'modify-hist-3', 'Phase'])
#     # print(df)
#     df.to_csv('FAA_ratio.csv', index=False)
             

        

