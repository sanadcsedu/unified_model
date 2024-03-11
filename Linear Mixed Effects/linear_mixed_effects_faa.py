import environment5 as environment5
import misc
from pathlib import Path
import glob
import os 
from read_data import read_data
import pandas as pd 

def get_user_name(raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    obj2 = misc.misc(user_list)
    new_data = []

    for feedback_file in user_list:
        user_name = get_user_name(feedback_file)
        excel_files = glob.glob(os.getcwd() + '/RawInteractions/faa_data/*.csv')
        raw_file = [string for string in excel_files if user_name in string][0]
        obj = read_data()
        data = obj.merge(raw_file, feedback_file)
        numerator_bar_4 = 0
        numerator_bar_2 = 0
        numerator_scatterplot_0_1 = 0
        numerator_hist_3 = 0

        denominator = 0
        split_time = int((data[-1][1] - data[0][1])/2)
        idx = 0
        for _ in iter(int, 1):
            denominator += 1
            if data[idx][3] == 'bar-4':
                numerator_bar_4 += 1
            elif data[idx][3] == 'bar-2':
                numerator_bar_2 += 1
            elif data[idx][3] == 'scatterplot-0-1':
                numerator_scatterplot_0_1 += 1
            else:
                numerator_hist_3 += 1
                
            if data[idx][1] > split_time:
                phase = 'First'
                probability_bar_4 = round(numerator_bar_4 / denominator, 2)
                probability_bar_2 = round(numerator_bar_2 / denominator, 2)
                probability_hist_3 = round(numerator_hist_3 / denominator, 2)
                probability_scatterplot_0_1 = round(numerator_scatterplot_0_1 / denominator, 2)

                new_data.append([user_name, 'FAA', probability_bar_2, probability_bar_4,probability_scatterplot_0_1, probability_hist_3, phase])
                break
            idx += 1

        # print("idx 1 " + str(idx))
        numerator_bar_4 = 0
        numerator_bar_2 = 0
        numerator_scatterplot_0_1 = 0
        numerator_hist_3 = 0
        denominator = 0

        for i in range(idx, len(data)):            
            denominator += 1
            if data[i][3] == 'bar-4':
                numerator_bar_4 += 1
            elif data[i][3] == 'bar-2':
                numerator_bar_2 += 1
            elif data[i][3] == 'scatterplot-0-1':
                numerator_scatterplot_0_1 += 1
            else:
                numerator_hist_3 += 1
            
            if i == len(data) - 1:
                phase = 'Second' 
                probability_bar_4 = round(numerator_bar_4 / denominator, 2)
                probability_bar_2 = round(numerator_bar_2 / denominator, 2)
                probability_hist_3 = round(numerator_hist_3 / denominator, 2)
                probability_scatterplot_0_1 = round(numerator_scatterplot_0_1 / denominator, 2)

                new_data.append([user_name, 'FAA', probability_bar_2, probability_bar_4,probability_scatterplot_0_1, probability_hist_3, phase])

    df = pd.DataFrame(new_data, columns = ['User', 'Dataset', 'bar2', 'bar4', 'scatterplot01', 'hist3', 'Phase'])
    print(df)
    df.to_csv('FAA.csv', index=False)
             

        

