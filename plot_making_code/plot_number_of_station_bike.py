import json
import requests
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

def plot_number_of_station_bike(config):
    """
    설명
    Args:
        config: configuration
    """
    data_path = config['path']['data']
    # if 'parking.csv' in os.listdir(data_path):
    #     parking = pd.read_csv(f'{data_path}parking.csv')
    # else:
    #     parking = preprocessing(config)
    # make_parking_plot(config, parking)
    if 'scatter.csv' in os.listdir(data_path):
        scatter = pd.read_csv(f'{data_path}scatter.csv')
    else:
        scatter = preprocessing_specific(config)
    make_scatter_plot(config, scatter)

def preprocessing_specific(config):
    usage_path = config['path']['usage']
    station_list = config['station']
    result_df = pd.DataFrame()
    for file_name in tqdm(os.listdir(usage_path)):
        df = pd.read_csv(f'{usage_path}{file_name}')
        _ = df.loc[(df.loc[:,'RENT_STATION_ID'].apply(lambda x: x in station_list))|(df.loc[:,'RETURN_STATION_ID'].apply(lambda x: x in station_list)),['RENT_STATION_ID', 'RETURN_STATION_ID', 'RENT_DT', 'RTN_DT', 'SEX_CD']]
        result_df = pd.concat([result_df, _])
    result_df = result_df.loc[~result_df.loc[:,'RTN_DT'].isna()]
    result_df.reset_index(drop=True, inplace=True)
    data_path = config['path']['data']
    result_df.to_csv(f'{data_path}scatter.csv')
    return result_df

def make_scatter_plot(config, array_:pd.DataFrame):
    colors = config['colors_vs']
    save_path = config['path']['plot']
    station_list = config['station']
    for station in station_list:
        for mode in [['RENT', 'RENT'], ['RETURN', 'RTN']]:
            _ = array_.loc[array_.loc[:,f'{mode[0]}_STATION_ID'] == station]
            _.loc[:,'RENT_DT'] = _.loc[:,'RENT_DT'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
            _.loc[:,'RTN_DT'] = _.loc[:,'RTN_DT'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
            _.insert(len(_.columns), 'x', _.loc[:,f'{mode[1]}_DT'].apply(lambda x: x.hour+x.minute/60+x.second/3600), True)
            _.insert(len(_.columns), 'y', (_.loc[:,"RTN_DT"]-_.loc[:,"RENT_DT"]).apply(lambda x: x.total_seconds()/60))
            fig = plt.figure(figsize=(12,6))
            gs = fig.add_gridspec(2, 2,  width_ratios=(8, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)
            # ax.set_title(f'{station} {mode[0]}')
            plt.suptitle(f'{station} {mode[0]}')
            ax.set_xlabel('Hours')
            ax.set_ylabel('Usage hours')
            for idx, sex in enumerate(['Unknown', 'M', 'F']):
                if sex in ['M', 'F']:
                    __ = _.loc[_.loc[:,'SEX_CD'] == sex]
                    x = np.array(__.loc[:,'x'], dtype='float64')
                    y = np.array(__.loc[:,'y'], dtype='float64')
                    ax.scatter(x=x, y=y, c=[colors[idx]]*len(__), sizes=np.array([1]*len(__)), alpha=0.03, label=f'{sex}')
                    ax_histx.hist(x, bins=np.arange(0,24,0.1), histtype='step', color=colors[idx])
                    ax_histy.hist(y, bins=np.arange(0,120,0.5), histtype='step', color=colors[idx], orientation='horizontal')
                else:
                    __ = _.loc[~_.loc[:,'SEX_CD'].isna()]
                    ax.scatter(x=np.array(__.loc[:,'x'], dtype='float64'), y=np.array(__.loc[:,'y'], dtype='float64'),
                                          c=[colors[idx]]*len(__), sizes=np.array([1]*len(__)), alpha=0.03, label=f'{sex}')
            ax.set_ylim(bottom=-1, top=121)
            ax.set_yticks(range(0,121,10))
            ax.set_yticklabels(range(0,121,10))
            ax.set_xlim(left=-0.1, right=24.1)
            ax.set_xticks(range(0,25))
            custom_legends = [Line2D([0], [0], marker='o', color=colors[0], label='Unknown', markersize=6, alpha=1, linestyle=''),
                              Line2D([0], [0], marker='o', color=colors[1], label='M', markersize=6, alpha=1, linestyle=''),
                              Line2D([0], [0], marker='o', color=colors[2], label='F', markersize=6, alpha=1, linestyle='')]
            ax.legend(handles=custom_legends, loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.savefig(f'{save_path}_{station}_{mode[0]}.png', dpi=1000)

def make_parking_plot(config, array:pd.DataFrame):
    """
    만들어진 데이터를 이용해 주중 / 주말 대여소 대기 자전거 대수 시간별 그래프 작성
    Args:
        config:
        array: array.shape() = (*,4), array.columns = ['rackTotCnt', 'parkingBikeTotCnt', 'stationId', 'stationDt']
    """
    station_list = config['station']
    save_path = config['path']['plot']
    colors = config['colors']
    array.insert(len(array.columns), 'weekend', 
                 array.loc[:,'stationDt'].apply(lambda x: datetime.date(x//1000000,(x//10000)%100,(x//100)%100).weekday() in [5, 6]), True)
    array.insert(len(array.columns), 'day', 
                 array.loc[:,'stationDt'].apply(lambda x: (x//100)%100), True)
    array.insert(len(array.columns), 'hour', 
                 array.loc[:,'stationDt'].apply(lambda x: x%100), True)    
    for station in station_list:
        _ = array.loc[array.loc[:,"stationId"] == station]
        _.reset_index(drop=True, inplace=True)
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.set_title('Parking counts')
        ax.set_xlabel('Hours')
        ax.set_ylabel('counts')
        for dd in _.drop_duplicates(subset=['day']).loc[:,'day']:
            ax.plot(_.loc[_.loc[:,'day']==dd,'hour'], _.loc[_.loc[:,'day']==dd,'parkingBikeTotCnt'], '--',
                    color=colors[2 + 5*_.loc[(_.loc[:,'day']==dd)&(_.loc[:,'hour']==0),'weekend'].values[0]])
        ax.axhline(y=_.loc[:,'rackTotCnt'].median(), linestyle='--', color='red')
        custom_line = [Line2D([0], [0], color=colors[2], linestyle='--', lw=3, label='Weekday'),
                       Line2D([0], [0], color=colors[7], linestyle='--', lw=3, label='Weeekend'),
                       Line2D([0], [0], color='red', linestyle='--', lw=3, label='capacity')]
        ax.legend(handles=custom_line)
        plt.savefig(f'{save_path}{station}.png')

def preprocessing(config):
    """
    api를 통해 지난 7일 동안의 주정차 데이터 불러오기
    """
    station_list = config['station']
    keys = config['keys']
    data_path = config['path']['data']

    hourly_usage_pd = pd.DataFrame()
    ordinal = datetime.date.toordinal(datetime.date.today())
    for D in tqdm(range(ordinal-7, ordinal)):        
        YYYY = datetime.date.fromordinal(D).year
        MM = datetime.date.fromordinal(D).month
        DD = datetime.date.fromordinal(D).day
        for hh in range(24):
            thousand = 0
            iters_hour = True
            while iters_hour:
                start_idx, end_idx = 1000*thousand+1, 1000*(thousand+1)
                response = requests.get(f'http://openapi.seoul.go.kr:8088/{keys}/json/bikeListHist/{start_idx}/{end_idx}/{YYYY}{MM}{DD}{hh:0>2}')
                try:
                    hourly_usage_json = response.json()['getStationListHist']
                except:
                    break
                total_count = int(hourly_usage_json['list_total_count'])
                _ = pd.DataFrame.from_dict(hourly_usage_json['row'])
                _ = _.loc[_.loc[:,'stationId'].apply(lambda x: x in station_list),['rackTotCnt', 'parkingBikeTotCnt', 'stationId', 'stationDt']]
                hourly_usage_pd=pd.concat([hourly_usage_pd,_])
                if end_idx-start_idx+1 > total_count :
                    iters_hour = False
                else:
                    thousand += 1
    hourly_usage_pd.drop_duplicates(subset=['stationId', 'stationDt'], inplace=True)
    hourly_usage_pd.reset_index(drop=True, inplace=True)
    hourly_usage_pd.to_csv(f'{data_path}parking.csv')
    return hourly_usage_pd

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    warnings.filterwarnings(action='ignore')
    plot_number_of_station_bike(config)