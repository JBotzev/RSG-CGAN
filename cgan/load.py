import sys
from numpy import shape
import csv
sys.path.append('..')
import numpy as np
import pandas as pd

#Load .csv renewables data into GANs model

def load_wind(label_type=1):
    #Example dataset created for evnet_based GANs wind scenarios generation
    # Data from NREL wind integrated datasets
    with open('datasets/Germany_wind_signal_C.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    min = np.ndarray.min(rows)
    print("Maximum value of wind", m)
    print("Minimum value of wind", min)
    for x in range(rows.shape[1]):
        train = rows[:, x].reshape(-1, 24)
        # train = train / m
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    # print("Shape TrX", shape(trX))

    label_str = label_type
    if label_type == 1:
        label_str = 'datasets/labels_wind_24.csv'
    if label_type == 2:
        label_str = 'datasets/bi_labels_Germany_C.csv'

    with open(label_str, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    label = np.array(rows, dtype=int)
    print("Label shape", shape(label))
    return trX, label


def load_solar(label_type=1):
    data = pd.read_csv('datasets/Germany_solarseason_C.csv')
    trX = data.iloc[:,1:-1].to_numpy()
    label = data['month'].to_numpy().reshape(249888,1)

    max = trX.max().max()
    trX = trX / max

    print('solar')
    print(trX)
    print(label)
    return trX, label


# def load_solar_data():
#     with open('datasets/solar label.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         rows = [row for row in reader]
#     labels = np.array(rows, dtype=int)
#     print(shape(labels))
#
#     with open('datasets/Germany_solar_C.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         rows = [row for row in reader]
#     rows = np.array(rows, dtype=float)
#     # rows=rows[:104832,:] #Change to the time points in your own dataset
#     print(shape(rows))
#     trX = np.reshape(rows.T,(-1,24)) #Corresponds to the GAN input dimensions.
#     print(shape(trX))
#     m = np.ndarray.max(rows)
#     print("maximum value of solar power", m)
#     trY=np.tile(labels,(32,1))
#     trX=trX/m
#     return trX,trY
#
# def load_wind_data_spatial():
#     with open('datasets/spatial.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         rows = [row for row in reader]
#     rows = np.array(rows, dtype=float)
#     m = np.ndarray.max(rows)
#     print("Maximum value of wind", m)
#     rows=rows/m
#     return rows

