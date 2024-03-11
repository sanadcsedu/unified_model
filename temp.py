import environment5 as environment5
import misc
from pathlib import Path
import glob
import os 
from read_data import read_data
import pandas as pd 
import pdb 

def get_user_name(raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_brightkite
    obj2 = misc.misc(user_list)
    new_data = []
    actions = ["Brush", "Pan", "Zoom", "Select"]
    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/brightkite_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]
        obj = read_data()
        data = obj.merge(raw_file, feedback_file)
        for itr in data:
            print(itr)
        pdb.set_trace()