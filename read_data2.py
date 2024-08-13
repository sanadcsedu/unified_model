#in this version of read_data2, we are focusing on the shifts in users data focus.
# thus the actions are same, modify-[vis_name], modify-bar, modify-hist, modify-hist2 etc.
# our focus is not on the raw actions

import csv
import pdb
import glob
import random
import os
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict


class read_data:
    def __init__(self):
        path = os.getcwd()

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            if row['type'] == "interface" or row['type'] == 'simulation' or row['type'] == 'none':
                continue
            if row['type'] == 'recall':
                data.append([seconds, 'observation'])
            else:
                data.append([seconds, row['type']])
        return data

    def raw_to_memory(self, csv_reader):
        # data[0] = time, data[1] = action, data[2] = visualization
        next(csv_reader)
        data = []
        for lines in csv_reader:
            time = lines[1].split(":")
            mm = int(time[1])
            ss = int(time[2])
            seconds = mm * 60 + ss
            data.append([seconds, lines[2], lines[3]])
        return data

    # Used for merging the raw interaction (reformed) files with the Excel feedback files
    def merge(self, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)

        holder = []
        idx = 0
        idx2 = 0
        prev_action = None
        for idx in range(len(feedback_data)):
            # print(feedback_data[idx])
            while idx2 < len(raw_data) and feedback_data[idx][0] >= raw_data[idx2][0]:
                # includes time, which is necessary for statistical tests
                # 0: index, 1: time 2: action, 3: visualization, 4: high_level_state, 5: reward
                holder.append([idx2, raw_data[idx2][0], raw_data[idx2][1], raw_data[idx2][2], feedback_data[idx][1], 0])

                # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward    
                # holder.append([idx2, raw_data[idx2][1], raw_data[idx2][2], feedback_data[idx][1], 0])

                idx2 += 1
            if len(holder) > 1:
                # holder[idx2 - 1][5] = 1
                holder[idx2 - 1][4] = 1
            idx += 1

        # 0: index, 1: time 2: action, 3: visualization, 4: high_level_state, 5: reward 6: action(Data Focus Shift)
        for idx in range(len(holder)-1):
            cur_vis = holder[idx][3]
            next_vis = holder[idx+1][3]
            if cur_vis == next_vis:
                action = 'same-'+ next_vis
            else:
                action = 'modify-'+ next_vis
            holder[idx].append(action)
        holder[len(holder)-1].append(None)

        return holder[:len(holder)-1]


if __name__ == "__main__":
    obj = read_data()
    print(obj.raw_files)
    print(obj.excel_files)
    obj.get_files()