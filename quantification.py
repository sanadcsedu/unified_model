#This version of the Mann Whitney test checks stationarity of the actions based on recieved rewards
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
        self.excel_files = glob.glob(path + '/FeedbackLog/*brightkite.xlsx')
        self.dictionary = defaultdict(int)

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            # if row['type'] == "interface" or row['type'] == 'simulation' or row['type'] == 'none':
            #     continue
            # if row['type'] == 'recall':
            #     data.append([seconds, 'observation'])
            # else:
            data.append([seconds, row['type']])
        return data


    def merge(self, excel_fname):
        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)
        
        d = defaultdict(int)
        for idx in range(len(feedback_data)):
            # print(feedback_data[idx])
            d[feedback_data[idx][1]] += 1
            # pdb.set_trace()
        # print(d)
        for keys, values in d.items():
            self.dictionary[keys] += values

if __name__ == "__main__":
    obj = read_data()
    for f in obj.excel_files:
        obj.merge(f)
        # pdb.set_trace()
    print(obj.dictionary)
    # print(obj.raw_files)
    # print(obj.excel_files)
    # obj.get_files()