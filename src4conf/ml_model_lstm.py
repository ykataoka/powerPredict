# use sklearn 0.18 (source activate carnd-term1)
import pandas as pd
import numpy as np
import h5py
import itertools
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold  # v0.18
from sklearn.model_selection import KFold  # v0.18
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pickle

from keras import backend as K
from keras.layers import Input, Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import Callback

from datetime import datetime 

# Overfit in Random Forest
"""
1. n_estimators : increase it to avoid overfitting
2. max_features : reduce the features (default = auto(sqrt))
3. max_depth : hyper parameter
4. min_samples_leaf : hyper parameter
5. validation number
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

# Tour of California(ToC) Dataset
"""
Index(['﻿TimeStampEpoch', 'TimeStampEpochFormatted', 'TimeStampStart',
       'TimeStampEnd', 'ProcessingTime', 'RaceSpeed', 'RaceMaxSpeed',
       'RaceDistanceToFinish', 'RaceDistanceFromStart', 'Status', 'Bib',
       'HasYellowJersey', 'Latitude', 'Longitude', 'CurrentSegmentId',
       'Altitude', 'Slope', 'HeartRate', 'HeartRateRollAvg', 'Cadence',
       'CadenceRollAvg', 'Power', 'PowerRollAvg', 'CurrentSpeed',
       'CurrentSpeedRollAvg', 'PositionInTheRace', 'GapToFirstRiderT',
       'GapToPreviousRiderT', 'AverageSpeedR', 'MaximumSpeedR',
       'DistanceToFinish', 'DistanceFromStart'],
      dtype='object') 32
"""

# Pipeline
"""
1. Read the data
2. Feature Engineering
3. Machine Learning
4. Physics prediction
5. Evaluation
"""


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, mae = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, mae: {}\n'.format(loss, mae))

# Functions
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        print("Normalized confusion matrix")
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_shape = cm.shape

        # change to 3 digits decimal for better visualization
        cm = cm.flatten()
        cm = [float("{:.3f}".format(i)) for i in list(cm)]
        cm = np.array(cm).reshape(cm_shape)
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() * 1./2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def read_rider_data():
    # dataset
    if dataset == 'small':   # 1st dataset Apr. 20th
        data = pd.read_csv('../data/ToCStage0700.csv')
    elif dataset == 'middle':  # 2nd dataset May 5th
        data = pd.read_csv('usable_data2.csv')
    return data


def read_weather_data():
    # dataset
    if dataset == 'small':   # 1st dataset Apr. 20th
        data = pd.read_csv('../data/ToCStage0700.csv')
    elif dataset == 'middle':  # 2nd dataset May 5th
        data = pd.read_csv('usable_data.csv')
    return data


def convertWatt2class10(y, y_org):
    # Yasu's Special Level Design
    # 1. 0~5[watt] as level 0   (8.5% data)
    # 2. >500[watt] as level 10  (1.2% data)
    # 3. others are evenly splitted
    powers = []
    tmp = pd.DataFrame(y_org)
    idx_level1 = tmp[tmp[0] <= 5].index
    idx_level10 = tmp[tmp[0] >= 500].index
    idx_rest = tmp.index.difference(idx_level10)
    idx_rest = idx_rest.difference(idx_level1)

    split_num = 8
    restsize = idx_rest.shape[0]
    power_rest = tmp[idx_rest].sort_values()
    rest = [list(power_rest.iloc[int(restsize*i/split_num):
                                 int(restsize*(i+1)/split_num)])
            for i in range(split_num)]
    powers = [list(tmp[idx_level1])] + rest + [list(tmp[idx_level10])]

    return np.round(powers)

# Setting Parameter
title = 'LSTM_LB10'  # 'small' or 'middle'
dataset = 'middle'  # 'small' or 'middle'
feature = 'physics_ml'  # 'full' or 'onlyGPS' or 'physics_ml'
label = 'PowerRollAvg'  # 'Power' or 'PowerRollAvg' or 'PowerRollAvgLevel' or "PowerCandence"
search = 'Fixed'  # 'GridSearch' or 'Fixed'

# Constant Parameter
if __name__ == '__main__':
    """
    1. Read data / Load data

    - ToC 2016
    ['Bib', 'currentSpeed', 'Altitude', 'Slope',  # rider's feature
     'windSpeed', 'windDirection',                # External Info.
     'Power', 'Cadence']                          # Label data
    """
    h5f = h5py.File('lstm_LB10_dataX.h5', 'r')
    dataX = h5f['dataX'][:]
    h5f.close()
    dataX = dataX[:, 1:]  # remove the label data
    print("dataX shape -> ", dataX.shape)

    h5f = h5py.File('lstm_LB10_dataY.h5', 'r')
    dataY = h5f['dataY'][:]
    h5f.close()

    h5f = h5py.File('lstm_LB10_param.h5', 'r')
    scale_mean_, scale_var_ = h5f['param']
    scale_mean = scale_mean_[0]  # 0: param for power scale
    scale_var = scale_var_[0]  # 0: param for power scale
    h5f.close()

    """
    3. Predicton
    """
    # Stratified K fold cross validation
    split_num = 5
    skf = StratifiedKFold(n_splits=split_num)
    kf = KFold(n_splits=split_num)
    train_mae = []
    test_mae = []
    i = 0
    sample_prediction = []

    # Hyper Parameter
    look_back = 10

    for index, (train_index, test_index) in enumerate(kf.split(dataX)):
#    for train_index, test_index in skf.split(dataX, dataY):
        print('debug : ', index)
        
        # read data
        X_train, X_test = dataX[train_index], dataX[test_index]
        y_train, y_test = dataY[train_index], dataY[test_index]

        # LSTM Model
        model = Sequential()
        model.add(LSTM(10, #return_sequences=True,
                       input_shape=(X_test.shape[1], look_back),
                       dropout=0.4))
#        model.add(LSTM(32, return_sequences=True, dropout=0.4))
#        model.add(LSTM(10, dropout=0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=['mae'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=256,
                            verbose=2, validation_data=(X_test, y_test))
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        # Speed Measure
        start_time = datetime.now()
        for i in range(200):
            predY_plot = model.predict(X_test[i:i+1])  # 1. pred
            time_elapsed = datetime.now() - start_time
            sample_prediction.append(time_elapsed.microseconds)
            start_time = datetime.now()

        # Evaluation
        mae = mean_absolute_error(y_train, trainPredict)
        mae = mae * np.sqrt(scale_var) + scale_mean
        train_mae.append(mae)
        print("MAE(train, fold) : ", np.average(mae))
        mae = mean_absolute_error(y_test, testPredict)
        mae = mae * np.sqrt(scale_var) + scale_mean
        test_mae.append(mae)
        print("MAE(test, fold) : ", np.average(mae))

        # Plot : Graph of prediction and groundtruth
        dataY_plot = dataY[48200:49600]
        dataY_plot = np.array(dataY_plot) * np.sqrt(scale_var) + scale_mean
        predY_plot = model.predict(dataX[48200:49600])
        predY_plot = predY_plot * np.sqrt(scale_var) + scale_mean
        plt.plot(list(range(len(dataY_plot))), dataY_plot,
                 color='b', alpha=0.9, lw=2, label='Ground Truth')
        plt.plot(list(range(len(predY_plot))), predY_plot,
                 color='r', alpha=0.5, lw=2, label='Prediction')
        plt.xlabel('Time[sec]')
        plt.ylabel('Power Roll Avg[watt]')
        plt.title('Comparison Between Ground Truth and Prediction (LSTM)')
        plt.grid(True)
        plt.legend(ncol=1)
        filename = 'testplot/' + title + '_performance_Power_Time.png'
        plt.savefig(filename, dpi=300)
        plt.clf()

        # Plot : Graph of mae
        train_mae_out = np.array(history.history['mean_absolute_error']) * np.sqrt(scale_var) + scale_mean
        test_mae_out = np.array(history.history['val_mean_absolute_error']) * np.sqrt(scale_var) + scale_mean
        plt.plot(train_mae_out)
        plt.plot(test_mae_out)
        plt.title('Mean Absolute Error over the training')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(['train', 'test'], loc='upper left')
        filename = 'testplot/' + title + '_performance_MAE_Epoch.png'
        plt.savefig(filename, dpi=300)
        plt.clf()

        if index == 0:
            break
        
#    with open('pred_wattage_' + title + '_small.pkl', mode='wb') as f:
#        pickle.dump(model, f)
#    with open('feature_scaler0524.pkl', mode='wb') as f:
#        pickle.dump(scaler, f)


#print sample time
print(sample_prediction)
print('time max:', max(sample_prediction))
print('time min:', min(sample_prediction))
print('time avg:', np.average(sample_prediction))
print('time std:', np.std(sample_prediction))

print("MAE(train, all) : ", np.average(train_mae))
print("MAE(test, all) : ", np.average(test_mae))
