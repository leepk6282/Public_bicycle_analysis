import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath('.'))
from data_making_code.bikeStationList import bikeStationList
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_rent_return(config):
    """
    설명
    Args:
        config: configuration
    """
    result_df, result_df_sex, result_df_age = preprocessing(config)
    make_demand_plot(config, result_df)
    make_imbalance_plot(config, result_df)
    make_sex_bar(config, result_df_sex)
    make_age_bar_q(config, result_df_age)

def make_demand_plot(config, array:pd.DataFrame):
    """
    설명
    Args:
        config:
        array: pd.DataFrame.shape = (*,3), columns = ['rent','return','rent-return']
    """
    save_path = config['path']['plot']
    fig, ax = plt.subplots(figsize = (12,6))
    ax.set_title('Counts')
    ax.set_xlabel('% of stations')
    ax.set_ylabel('Counts')
    for idx, mode in enumerate(['Rent', 'Return']):
        array = array.sort_values(by=[array.columns[idx]], ascending=False)
        ax.plot([x/len(array)*100 for x in range(len(array))], array.iloc[:,idx], '-', label=f'{mode} Counts', color=config['colors'][idx*3+3])
    desc = array.iloc[:,idx].describe()
    ax.axhline(y=desc[1]+desc[2]*3, linestyle='--', color='red',label='mean+3*std')
    ax.legend(loc='upper right')
    ax.set_xticks(range(0,101,5))
    ax.set_xticklabels([f'{x}%' for x in range(0,101,5)])
    plt.grid(True)
    plt.savefig(f"{save_path}/demand_sort.png")

def make_imbalance_plot(config, array:pd.DataFrame):
    """
    설명
    Args:
        config:
        array: pd.DataFrame.shape = (*,3), columns = ['rent','return','rent-return']
    """
    save_path = config['path']['plot']
    fig, ax = plt.subplots(figsize = (12,6))
    ax.set_title('Imbalance of demand and supply')
    ax.set_xlabel('% of stations')
    ax.set_ylabel('Rents - Returns')
    array = array.sort_values(by=[array.columns[2]], ascending=False)
    ax.plot([x/len(array)*100 for x in range(len(array))], array.iloc[:,2], '-', label='Imbalance', color=config['colors'][6])
    desc = array.iloc[:,2].describe()
    ax.axhline(y=desc[1]+desc[2]*3, linestyle='--', color='red',label='mean+3*std')
    ax.axhline(y=desc[1]-desc[2]*3, linestyle='--', color='red',label='mean-3*std')
    ax.set_xticks(range(0,101,5))
    ax.set_xticklabels([f'{x}%' for x in range(0,101,5)])
    plt.grid(True)
    ax.legend(loc='upper right')
    plt.savefig(f"{save_path}/imbalance_sort.png")

def make_sex_bar(config, array:pd.DataFrame):
    """
    설명
    Args:
        config:
        array: pd.DataFrame.shape = (*,2), columns = ['Male', 'Female']
    """
    save_path = config['path']['plot']
    fig, ax = plt.subplots(figsize = (12,6))
    ax.set_title('Differences of Male and Female')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Counts')
    ax.bar(array.index, height=array.loc[:,'Male'], width=0.4, align='edge', label='Male', color=config['colors'][2])
    ax.bar(array.index, height=array.loc[:,'Female'], width=-0.4, align='edge', label='Female', color=config['colors'][7])
    ax.set_xticks(range(0,61,5))
    ax.set_xticklabels(range(0,61,5))
    ax.set_xlim(0, 61)
    plt.grid(True)
    ax.legend(loc='upper right')
    plt.savefig(f"{save_path}/differences_of_sex.png")

def make_age_bar_q(config, array:pd.DataFrame):
    """
    설명
    Args:
        config:
        array: pd.DataFrame.shape = (4, 75), index= [25%, 50%, 75%, counts], columns = Age
    """
    save_path = config['path']['plot']
    minutes = config['age']['minutes']
    ages = config['age']['age']
    fig, ax = plt.subplots(figsize = (12,6))
    ax_ = plt.twinx()
    ax.set_title('Differences of Age')
    ax.set_xlabel('Ages')
    ax.set_ylabel('Minutes')
    for idx, x in enumerate([0.75, 0.5, 0.25]):
        ax.bar(range(ages[0], ages[1]), array[2-idx,:], align = 'edge', label=f'{x*100}%', color=config['colors'][idx*3])
    ax.set_xticks(range(ages[0],ages[1],5))
    ax.set_xticklabels(range(ages[0],ages[1],5))
    ax.set_xlim(ages[0], ages[1])
    plt.grid(True)
    ax.legend(loc='upper right')
    for idx, age in enumerate(range(ages[0], ages[1])):
        ax_.axhline(y=array[3,idx], xmin=(idx+0.15)/len(array[3]) , xmax=(idx+0.7)/len(array[3]), color='red', linewidth=3)
    ax_.set_ylabel('Counts')
    custom_line = [Line2D([0], [0], color='red', lw=3)]
    ax_.legend(custom_line, ['counts'], loc = 'upper left')
    plt.savefig(f"{save_path}/differences_of_age.png")

def pivot_and_sum(config, indices):
    """
    pivot_table 만들고 sum 진행 결과 반환
    Args:
        config:
        indices
    """
    usage_path = config['path']['usage']
    minutes = config['age']['minutes']
    ages = config['age']['age']
    file_list = os.listdir(usage_path)
    result_df_rent, result_df_return, result_df_M, result_df_F = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    age_min_df = pd.DataFrame(np.zeros([len(range(minutes[0], minutes[1])), len(range(ages[0], ages[1]))]), index=range(minutes[0], minutes[1]), columns=range(ages[0],ages[1]), dtype = 'int')
    for file in tqdm(file_list):
        file_df = pd.read_csv(f'{usage_path}{file}')
        booleans = (file_df.loc[:,'RENT_STATION_ID'].apply(lambda x: x in indices)) & \
                   (file_df.loc[:,'RETURN_STATION_ID'].apply(lambda x: x in indices))
        file_df = file_df.loc[booleans]
        file_df.insert(len(file_df.columns), 'Age', file_df.loc[:, 'RENT_DT'].apply(lambda x: int(x[:4])) - file_df.loc[:, 'BIRTH_YEAR'], True)
        file_df.loc[:,'Age'].fillna(0, inplace=True)
        file_df.loc[:,'Age'] = file_df.loc[:,'Age'].astype('int')
        file_df.loc[:, 'Age'] = file_df.loc[:, 'Age'].apply(lambda x: abs(x))
        file_df.loc[:, 'USE_MIN'] = file_df.loc[:, 'USE_MIN'].apply(lambda x: abs(x))
        file_pivot_rent = pd.pivot_table(file_df, index= 'RENT_STATION_ID', values='SEX_CD', aggfunc='count')
        file_pivot_return = pd.pivot_table(file_df, index= 'RETURN_STATION_ID', values='SEX_CD', aggfunc='count')
        file_pivot_M = pd.pivot_table(file_df, index= 'USE_MIN', values='USE_DST', columns='SEX_CD', aggfunc='count').loc[:,'M']
        file_pivot_F = pd.pivot_table(file_df, index= 'USE_MIN', values='USE_DST', columns='SEX_CD', aggfunc='count').loc[:,'F']
        file_pivot_age = pd.pivot_table(file_df, index= 'USE_MIN', values='USE_DST', columns='Age', aggfunc='count')
        file_pivot_age = file_pivot_age.loc[(file_pivot_age.index >= age_min_df.index[0])&(file_pivot_age.index <= age_min_df.index[-1]),
                                            (file_pivot_age.columns <= age_min_df.columns[-1]) & (file_pivot_age.columns >= age_min_df.columns[0])]
        file_pivot_age.fillna(0, inplace=True)
        file_pivot_age = file_pivot_age.astype('int')
        result_df_rent = pd.concat([result_df_rent, file_pivot_rent], axis=1)
        result_df_return = pd.concat([result_df_return, file_pivot_return], axis=1)
        result_df_M = pd.concat([result_df_M, file_pivot_M], axis=1)
        result_df_F = pd.concat([result_df_F, file_pivot_F], axis=1)
        age_min_df = age_min_df + file_pivot_age
        age_min_df.fillna(0, inplace=True)
        age_min_df = age_min_df.astype('int')
    result_df_rent = pd.DataFrame(result_df_rent.sum(axis=1), index= result_df_rent.index, columns=['rent'])
    result_df_return = pd.DataFrame(result_df_return.sum(axis=1), index= result_df_return.index, columns=['return'])
    result_df_M = pd.DataFrame(result_df_M.sum(axis=1), index= result_df_M.index, columns=['Male'])
    result_df_F = pd.DataFrame(result_df_F.sum(axis=1), index= result_df_F.index, columns=['Female'])
    # result_df_age = pd.DataFrame(result_df_age.sum(axis=1), index= result_df_age.index, columns=['Age'])
    result_df = pd.concat([result_df_rent, result_df_return], axis=1)
    result_df_sex = pd.concat([result_df_M, result_df_F], axis=1)
    result_df.fillna(0, inplace=True)
    result_df_sex.fillna(0, inplace=True)
    result_df.insert(2, 'rent-return', result_df.iloc[:,0] - result_df.iloc[:,1], True)
    result_df_sex.sort_index(inplace=True)
    # result_df_age.sort_index(inplace=True)
    result_df_sex = result_df_sex.iloc[1:,:]
    array = np.zeros([4, len(age_min_df.columns)])
    for idx1, age in enumerate(age_min_df.columns):
        list_ = []
        for idx2, x in enumerate(age_min_df.loc[:,age]):
            if x >0:
                list_ = list_ + [idx2+1 for x in range(x)]
            else:
                list_ = [0]
        for idx3, q in enumerate([0.25, 0.50, 0.75]):
            array[idx3, idx1] = np.quantile(a=np.array(list_), q=q)
        array[3,idx1] = len(list_)
    return result_df, result_df_sex, array

def preprocessing(config):
    preprocessed_station, _, _ = bikeStationList(config)
    result_df, result_df_sex, result_df_age = pivot_and_sum(config, indices=preprocessed_station.index)
    return result_df, result_df_sex, result_df_age

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    plot_rent_return(config)