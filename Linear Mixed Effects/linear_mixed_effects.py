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
        numerator_brush = 0
        numerator_pan = 0
        numerator_zoom = 0
        numerator_select = 0

        denominator = 0
        split_time = int((data[-1][1] - data[0][1])/2)
        idx = 0
        for _ in iter(int, 1):
            # pdb.set_trace()
            denominator += 1
            if data[idx][2] == 'brush':
                numerator_brush += 1
            elif data[idx][2] == 'pan':
                numerator_pan += 1
            elif data[idx][2] == 'zoom':
                numerator_zoom += 1
            else:
                numerator_select += 1
            
            # print("{} {} {} {} {} {}".format(idx, data[idx][2], numerator_brush, numerator_pan, numerator_zoom, numerator_select))
            if data[idx][1] > split_time:
                phase = 'First'
                denominator = 1
                probability_brush = round(numerator_brush / denominator, 2)
                probability_pan = round(numerator_pan / denominator, 2) 
                probability_zoom = round(numerator_zoom / denominator, 2)
                probability_select = round(numerator_select / denominator, 2)

                # probability_brush = round((numerator_brush * 60 / split_time), 2)
                # probability_pan = round((numerator_pan * 60 / split_time), 2)
                # probability_zoom = round((numerator_zoom * 60 / split_time), 2)
                # probability_select = round((numerator_select * 60 / split_time), 2)
                
                new_data.append([user_name, 'Brightkite', probability_brush, probability_pan, probability_zoom, probability_select, phase])
                
                # freq = [numerator_brush, numerator_pan, numerator_zoom, numerator_select]
                # max_index = freq.index(max(freq))
                # new_data_entry = [user_name, 'Brightkite', actions[max_index], phase]
                # new_data.append(new_data_entry)

                break
            idx += 1
        # pdb.set_trace()
        # print("idx 1 " + str(idx))
        numerator_brush = 0
        numerator_pan = 0
        numerator_zoom = 0
        numerator_select = 0
        denominator = 0

        for i in range(idx, len(data)):            
            denominator += 1
            if data[i][2] == 'brush':
                numerator_brush += 1
            elif data[i][2] == 'pan':
                numerator_pan += 1
            elif data[i][2] == 'zoom':
                numerator_zoom += 1
            else:
                numerator_select += 1
            
            # print("{} {} {} {} {} {}".format(i, data[i][2], numerator_brush, numerator_pan, numerator_zoom, numerator_select))
            
            if i == len(data) - 1:
                phase = 'Second' 
                denominator = 1
                probability_brush = round(numerator_brush / denominator, 2)
                probability_pan = round(numerator_pan / denominator, 2) 
                probability_zoom = round(numerator_zoom / denominator, 2)
                probability_select = round(numerator_select / denominator, 2)

                # #rate of action per minute
                # probability_brush = round((numerator_brush * 60 / split_time), 2)
                # probability_pan = round((numerator_pan * 60 / split_time), 2)
                # probability_zoom = round((numerator_zoom * 60 / split_time), 2)
                # probability_select = round((numerator_select * 60 / split_time), 2)

                new_data.append([user_name, 'Brightkite', probability_brush, probability_pan, probability_zoom, probability_select, phase])

                # freq = [numerator_brush, numerator_pan, numerator_zoom, numerator_select]
                # max_index = freq.index(max(freq))
                # new_data_entry = [user_name, 'Brightkite', actions[max_index], phase]
                # new_data.append(new_data_entry)

        # pdb.set_trace()

    #Flight_Performance Data
    user_list = env.user_list_faa
    obj2 = misc.misc(user_list)

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]
        obj = read_data()
        data = obj.merge(raw_file, feedback_file)
        numerator_brush = 0
        numerator_pan = 0
        numerator_zoom = 0
        numerator_select = 0

        denominator = 0
        split_time = int((data[-1][1] - data[0][1])/2)
        idx = 0
        for _ in iter(int, 1):
            # pdb.set_trace()
            denominator += 1
            if data[idx][2] == 'brush':
                numerator_brush += 1
            elif data[idx][2] == 'pan':
                numerator_pan += 1
            elif data[idx][2] == 'zoom':
                numerator_zoom += 1
            else:
                numerator_select += 1
            
            # print("{} {} {} {} {} {}".format(idx, data[idx][2], numerator_brush, numerator_pan, numerator_zoom, numerator_select))
            
            if data[idx][1] > split_time:
                phase = 'First'
                denominator = 1
                probability_brush = round(numerator_brush / denominator, 2)
                probability_pan = round(numerator_pan / denominator, 2) 
                probability_zoom = round(numerator_zoom / denominator, 2)
                probability_select = round(numerator_select / denominator, 2)

                # probability_brush = round((numerator_brush * 60 / split_time), 2)
                # probability_pan = round((numerator_pan * 60 / split_time), 2)
                # probability_zoom = round((numerator_zoom * 60 / split_time), 2)
                # probability_select = round((numerator_select * 60 / split_time), 2)

                new_data.append([user_name, 'FAA', probability_brush, probability_pan, probability_zoom, probability_select, phase])
                # freq = [numerator_brush, numerator_pan, numerator_zoom, numerator_select]
                # max_index = freq.index(max(freq))
                # new_data_entry = [user_name, 'FAA', actions[max_index], phase]
                # new_data.append(new_data_entry)

                break
            idx += 1
        # if user_name == 'u2':
        #     pdb.set_trace()

        # print("idx 1 " + str(idx))
        numerator_brush = 0
        numerator_pan = 0
        numerator_zoom = 0
        numerator_select = 0
        denominator = 0

        for i in range(idx, len(data)):            
            denominator += 1
            if data[i][2] == 'brush':
                numerator_brush += 1
            elif data[i][2] == 'pan':
                numerator_pan += 1
            elif data[i][3] == 'zoom':
                numerator_zoom += 1
            else:
                numerator_select += 1
            
            # print("{} {} {} {} {} {}".format(i, data[i][2], numerator_brush, numerator_pan, numerator_zoom, numerator_select))
            
            if i == len(data) - 1:
                phase = 'Second' 
                denominator = 1
                probability_brush = round(numerator_brush / denominator, 2)
                probability_pan = round(numerator_pan / denominator, 2) 
                probability_zoom = round(numerator_zoom / denominator, 2)
                probability_select = round(numerator_select / denominator, 2)

                # probability_brush = round((numerator_brush * 60 / split_time), 2)
                # probability_pan = round((numerator_pan * 60 / split_time), 2)
                # probability_zoom = round((numerator_zoom * 60 / split_time), 2)
                # probability_select = round((numerator_select * 60 / split_time), 2)
    
                new_data.append([user_name, 'FAA', probability_brush, probability_pan, probability_zoom, probability_select, phase])
                # freq = [numerator_brush, numerator_pan, numerator_zoom, numerator_select]
                # max_index = freq.index(max(freq))
                # new_data_entry = [user_name, 'FAA', actions[max_index], phase]
                # new_data.append(new_data_entry)

    df = pd.DataFrame(new_data, columns = ['User', 'Dataset', 'Brush', 'Pan', 'Zoom', 'Select', 'Phase'])
    # print(df)
    df.to_csv('imMens_actions_count.csv', index=False)