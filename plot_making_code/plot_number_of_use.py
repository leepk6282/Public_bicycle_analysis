import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date


def plot_number_of_use(config):
    """
    설명
    Args:
        config: configuration
    """

    data_plot_path = config['path']['plot']
    os.makedirs(data_plot_path, exist_ok=True)
    array = preprocessing(config)
    make_plot(config, array, temperature(), data_plot_path)
    
def preprocessing(config):
    """
    데이터베이스로부터 필요한 데이터 추출 및 전처리
    Args:
        config: configuration
    """
    start_year = config['start_date']['year']
    usage_np = np.zeros([date.today().year-start_year+1, 12])
    path = config['path']['usage']
    for file in tqdm(os.listdir(path)):
        usage_np[int(file[:4])-start_year, int(file[4:6])-1] = len(pd.read_csv(f'{path}{file}'))
    return usage_np

def make_plot(config, array, temperature, save_path):
    '''
    numpy array를 가지고 plot 생성
    Args:
        config:
        array:
        temperature:
        save_path:
    '''
    start_year = config['start_date']['year']
    fig, ax = plt.subplots(figsize = (24,12))
    ax.set_title('Number of use')
    ax.set_xlabel('Date')
    ax.set_ylabel('Numbers')
    for years in range(len(array)):
        if years == 0 :
            _ = array[years][array[years] != 0]
            ax.plot(range(12-len(_),12) ,_, 'o-.', 
                    label = f'{start_year+years}', color = config['colors'][years])
        elif years == len(array)-1:
            _ = array[years][array[years] != 0]
            ax.plot(range(years*12, years*12+len(_)) ,_, 'o-.', 
                    label = f'{start_year+years}', color = config['colors'][years])
        else:
            ax.plot(range(years*12, (years+1)*12) ,array[years], 'o-.',
                    label = f'{start_year+years}', color = config['colors'][years])
    ax.set_xticks(range(0, len(array.flatten())+1, 12),
                  [f'{int(str(start_year)[2:])+x//12}/{(x%12)+1}' for x in range(0, len(array.flatten())+1, 12)])
    ax.set_xlim([-1, len(array.flatten())+1])
    ax.legend(loc='upper left')
    plt.savefig(f'{save_path}long_plot.png')

    fig, ax = plt.subplots(figsize = (16, 8))
    ax2 = ax.twinx()
    ax.set_title('Number of use')
    ax.set_xlabel('Date')
    ax.set_ylabel('Numbers')
    label_tem, color_tem = ['mean', 'lowest', 'highest'], ['green', 'blue', 'red']
    for idx, tem_array in enumerate(temperature):
        ax2.plot(range(12), tem_array, label = label_tem[idx], color = color_tem[idx])
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Degree Celsius')
    for years in range(len(array)):
        if years == 0 :
            _ = array[years][array[years] != 0]
            ax.plot(range(12-len(_),12) ,_, 'o-.',
                    label = f'{start_year+years}', color = config['colors'][years])
        elif years == len(array)-1:
            _ = array[years][array[years] != 0]
            ax.plot(range(len(_)) ,_, 'o-.', 
                    label = f'{start_year+years}', color = config['colors'][years])
        else:
            ax.plot(range(12) ,array[years], 'o-.', label = f'{start_year+years}', color = config['colors'][years])
    ax.set_xticks(range(12), [date(2000,month,1).strftime("%b") for month in range(1, 13)])
    ax.legend(loc='upper left')
    plt.savefig(f'{save_path}short_plot.png')

def temperature():
    """
    평균 기온, 최저 기온, 최고 기온을 월별로 표시
    출처 : https://data.kma.go.kr/stcs/grnd/grndTaList.do?pgmNo=70
    """
    tem = pd.read_csv('./data/temperature.csv', encoding='cp949')
    tem['year'] = tem.iloc[:,0].apply(lambda x: int(x[1:5]))
    tem['month'] = tem.iloc[:,0].apply(lambda x: int(x[6:8]))
    tem['day'] = tem.iloc[:,0].apply(lambda x: int(x[9:]))
    tem_np = np.zeros([3,12])
    for month in range(1, 13):
        tem_np[:,month-1] = tem[tem.loc[:,'month'] == month].mean()[1:4]
    return tem_np

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    plot_number_of_use(config)    