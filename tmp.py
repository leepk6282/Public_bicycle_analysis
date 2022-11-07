import pandas as pd
import numpy as np

a=pd.read_csv('./data/bikeStationList.csv', encoding='cp949')
print(a)
print(a[a.iloc[:,1].apply(lambda x: x.split(' ')[0])=='서울특별시'])