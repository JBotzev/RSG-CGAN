import sys
from numpy import shape
import csv
sys.path.append('..')
import numpy as np

#Load .csv renewables data into GANs model

def load_wind():
    #Example dataset created for evnet_based GANs wind scenarios generation
    # Data from NREL wind integrated datasets
    with open('datasets/Germany_wind_signal_C.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)

    for x in range(rows.shape[1]):
        train = rows[:, x].reshape(-1, 24)
        # train = train / m
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape TrX", shape(trX))

    with open('datasets/labels_wind_24.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    label = np.array(rows, dtype=int)
    print("Label shape", shape(label))
    return trX, label


def load_wind_data_spatial():
    with open('datasets/spatial.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    rows=rows/m
    return rows


def load_solar_data():
    with open('datasets/solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)
    print(shape(labels))

    with open('datasets/solar.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows=rows[:104832,:] #Change to the time points in your own dataset
    print(shape(rows))
    trX = np.reshape(rows.T,(-1,576)) #Corresponds to the GAN input dimensions.
    print(shape(trX))
    m = np.ndarray.max(rows)
    print("maximum value of solar power", m)
    trY=np.tile(labels,(32,1))
    trX=trX/m
    return trX,trY


