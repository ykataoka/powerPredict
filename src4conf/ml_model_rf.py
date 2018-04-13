# use sklearn 0.18 (source activate carnd-term1)
import pandas as pd
import numpy as np
import itertools
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold  # v0.18
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pickle
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
    data = read_rider_data()

    """
    2. Feature Selection
    """
    col_full = ['Altitude', 'Slope', 'HeartRate', 'HeartRateRollAvg',
                'Cadence', 'CadenceRollAvg', 'CurrentSpeed']
    col_physics_ml = ['Altitude', 'Slope', 'CurrentSpeed', 'CurrentSpeedRollAvg',
                      'dhdt', 'dvdt', 'dv2dt', 'rv', 'rv2']
    col_gps = ['Altitude', 'Slope', 'CurrentSpeed']

    if feature == 'full':
        X = data.ix[:, col_full]
    elif feature == 'colGPS':
        X = data.ix[:, col_gps]
    elif feature == 'physics_ml':
        X = data.ix[:, col_physics_ml]
    X = np.array(X)
    print("Size of feature : ", X.shape)

    if label == 'Power':
        y = data.ix[:, ['Power']]
    elif label == 'PowerRollAvg':
        y = data.ix[:, ['PowerRollAvg']]
    elif label == 'PowerRollAvgLevel':
        y = data.ix[:, ['PowerRollAvgLevel']]
    elif label == 'PowerCadence':
        y = data['PowerRollAvg'] * data['Cadence']
    y = np.array(y).reshape(-1)
    print("Size of label : ", y.shape)

    """
    3. Predicton
    """
    # Stratified K fold cross validation
    split_num = 5
    skf = StratifiedKFold(n_splits=split_num)
    train_rmse = []
    test_rmse = []
    train_mae = []
    test_mae = []
    i = 0
    sample_prediction = []
    # Grid Search
    # param_dist = {"n_estimators": [20, 50, 100],
    #               "max_depth": [3, None],
    #               "max_features": [0.6, 0.9],
    #               "min_samples_split": [0.7, 0.9],
    #               "min_samples_leaf": [1, 2, 3],
    #               "bootstrap": [True, False],
    #               "criterion": ["mae"]}
    param_dist = {"n_estimators": [20],
                  "max_depth": [3, None],
                  "max_features": [0.6],
                  "min_samples_split": [0.7],
                  "min_samples_leaf": [2],
                  "bootstrap": [True],
                  "criterion": ["mae"]}

    for train_index, test_index in skf.split(X, y):
        # read data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # normalization
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)  # apply the X_train scaler

        # Training to regression model
        if search == 'Fixed':
            model = RandomForestRegressor(n_estimators=200,
                                          max_features=1.0,
                                          max_depth=25,
                                          min_samples_leaf=3)
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        elif search == 'GridSearch':
            clf = RandomForestRegressor()
            grid_search = GridSearchCV(clf, param_grid=param_dist)
            start = time()
            print("grid_search start")
            grid_search.fit(X, y)
            print("GridSearchCV took %.2f sec for %d candidate parameter settings."
                  % (time() - start, len(grid_search.cv_results_['params'])))
            print(grid_search.best_estimator_)
            y_pred_test = grid_search.predict(X_test)
            y_pred_train = grid_search.predict(X_train)

        # Evaluation
        rmse = mean_squared_error(y_train, y_pred_train) ** 0.50
        mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse.append(rmse)
        train_mae.append(mae)
        print("RMSE(train, fold) : ", np.average(rmse))
        print("MAE(train, fold) : ", np.average(mae))
        rmse = mean_squared_error(y_test, y_pred_test) ** 0.50
        mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse.append(rmse)
        test_mae.append(mae)
        print("RMSE(test, fold) : ", np.average(rmse))
        print("MAE(test, fold) : ", np.average(mae))

        """
        Plot
        """
        if label == 'PowerRollAvgLevel':
            # 1. confusion matrix (training data)
            y_pred_train = np.round(y_pred_train)
            y_pred_test = np.round(y_pred_test)

            #    cnf_matrix = confusion_matrix(y_train, y_pred_train)
            cnf_matrix = confusion_matrix(y_test, y_pred_test)
            np.set_printoptions(precision=2)
            plt.figure(figsize=(7, 7))
            class_names = [str(k+1) for k in range(10)]
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')
            filename = 'testplot/RF_' + dataset + feature + label + search + \
                       '_confusion' + str(i) + '.png'
            plt.savefig(filename, dpi=300)
            plt.clf()

#        # 1. confusion matrix (training data)
#        y10_train = convertWatt2class10(y_train)
#        y10_test = convertWatt2class10(y_test)
#        y10_pred_train = convertWatt2class10(y_pred_train)
#        y10_pred_test = convertWatt2class10(y_pred_test)
#
#        #    cnf_matrix = confusion_matrix(y10_train, y10_pred_train)
#        cnf_matrix = confusion_matrix(y10_test, y10_pred_test)
#        np.set_printoptions(precision=2)
#        plt.figure(figsize=(7, 7))
#        class_names = [str(k+1) for k in range(10)]
#        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                              title='Normalized confusion matrix')
#        filename = 'testplot/RF_' + dataset + feature + label + search + \
#                   '_confusion' + str(i) + '.png'
#        plt.savefig(filename, dpi=300)
#        plt.clf()

        # 2. line graph of prediction and groundtruth
        plt.plot(data.index, y, color='b', alpha=0.5, lw=2,
                 label='Ground Truth')
        plt.scatter(data.index, y, color='c', alpha=0.5, s=1)

        # prediction line (train+test)
        tmp_train = pd.DataFrame(y_pred_train, index=data.index[train_index])
        tmp_test = pd.DataFrame(y_pred_test, index=data.index[test_index])
        tmp = pd.concat([tmp_train, tmp_test])
        tmp = tmp.sort_index()
#        plt.plot(tmp.index, tmp[0], color='r', alpha=0.5,
#                 lw=1, label='predict')
        plt.plot(test_index, tmp_test, color='r', alpha=0.5,
                 lw=1, label='predict')

        # Training performance
        train_time = data.index[train_index]
        plt.scatter(train_time, y_pred_train, color='g', alpha=0.5, s=2,
                    label='predict (train)')

        # Test performance
        test_time = data.index[test_index]
        plt.scatter(test_time, y_pred_test, color='r', alpha=0.5, s=2,
                    label='predict (test)')

        # Speed Measure
        start_time = datetime.now()
        for i in range(200):
            predY_plot = model.predict(X_test[i:i+1])  # 1. pred
            time_elapsed = datetime.now() - start_time
            sample_prediction.append(time_elapsed.microseconds)
            start_time = datetime.now()

        # title and x and y label
        plt.xlabel('Time[sec]')
        if label == 'Power':
            plt.ylabel('Power[watt]')
            plt.title("Actual and Predicted 'Power'[watt]")
        elif label == 'PowerRollAvg':
            plt.ylabel('Power Roll Avg[watt]')
            plt.title("Actual and Predicted 'Power Roll Avg'[watt]")
        elif label == 'PowerRollAvgLevel':
            plt.ylabel('PowerRollAvgLevel [10 level]')
            plt.title("Actual and Predicted 'Power Level'")

        # xrange
        plt.xlim(len(tmp.index)*5*i/(5*split_num),
                 len(tmp.index)*(5*i+1)/(5*split_num))

        # other parameter
        plt.grid(True)
        plt.legend(ncol=1)
        filename = 'testplot/RF_' + dataset + feature + label + search \
                   + '_graph_' + str(i) + '.png'
        plt.savefig(filename, dpi=300)
        plt.clf()
        i += 1

    with open('pred_wattage_model0619_small.pkl', mode='wb') as f:
        pickle.dump(model, f)
    with open('feature_scaler0524.pkl', mode='wb') as f:
        pickle.dump(scaler, f)

#print sample time
print(sample_prediction)
print('time max:', max(sample_prediction))
print('time min:', min(sample_prediction))
print('time avg:', np.average(sample_prediction))
print('time std:', np.std(sample_prediction))

#over all performance sample time
print("RMSE(train, all) : ", np.average(train_rmse))
print("RMSE(test, all) : ", np.average(test_rmse))
print("MAE(train, all) : ", np.average(train_mae))
print("MAE(test, all) : ", np.average(test_mae))
