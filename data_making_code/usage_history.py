import json
import requests
import time
from datetime import date
import pandas as pd
from tqdm import tqdm
import os

def usage_history(config):
    """
    지금 현재 없는 데이터를 찾고, 해당 데이터를 api를 통해 받아 온다.
    """
    data_usage_path = config['path']['usage']
    os.makedirs(data_usage_path, exist_ok=True)
    start_year = config['start_date']['year']
    start_month = config['start_date']['month']
    exist_date = list(map(lambda x: [int(x[:4]), int(x[4:6])], os.listdir(data_usage_path)))
    for not_exist_date in make_year_month_list(start_year, start_month):
        if not_exist_date not in exist_date:
            print(not_exist_date)
            get_history(config, not_exist_date[0], not_exist_date[1])

def make_year_month_list(year:int, month:int):
    """
    시작일부터 어제까지의 연, 월을 리스트로 생성
    Args:
        year: start year
        month: start month
    """
    year_month_list = []
    for idx, yy in enumerate(range(year, date.today().year+1)):
        for mm in range(1, 13):
            if (idx==0) & (month-mm>0):
                continue
            elif (2022==yy) & (mm-date.today().month>=0):
                continue
            else:
                year_month_list.append([yy, mm])
    return year_month_list

def get_history(config, year: int, month: int):
    """
    해당 연,월에 데이터가 없다면 api로부터 받아옴
    Args:
        config: configuration
        year: year
        month: month
    """
    keys = config['keys']
    data_usage_path = config['path']['usage']

    hourly_usage_pd = pd.DataFrame()
    iters_day = True
    ordinal = date.toordinal(date(year, month, 1))
    while iters_day:
        YYYYMMDD = date.fromordinal(ordinal)
        for HH in range(24):
            thousand = 0
            iters_hour = True
            while iters_hour:
                start_idx, end_idx = 1000*thousand+1, 1000*(thousand+1)
                response = requests.get(url=f"http://openapi.seoul.go.kr:8088/{keys}/json/tbCycleRentData/{start_idx}/{end_idx}/{YYYYMMDD}/{HH}")
                try:
                    hourly_usage_json = response.json()['rentData']
                except:
                    break
                total_count = int(hourly_usage_json['list_total_count'])
                hourly_usage_pd=pd.concat([hourly_usage_pd,pd.DataFrame.from_dict(hourly_usage_json['row'])])
                if end_idx >= total_count :
                    iters = False
                thousand += 1
            print(YYYYMMDD, HH)
        if date.fromordinal(ordinal).month == date.fromordinal(ordinal+1).month:
            ordinal += 1
        else:
            iters_day = False
            break
    hourly_usage_pd.to_csv(f'{data_usage_path}{date.fromordinal(ordinal).year}{date.fromordinal(ordinal).month:0>2}.csv')

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    usage_history(config)