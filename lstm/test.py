#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.models import load_model
import pickle
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='/tmp/test.log',
                    filemode='w')

logging.debug('debug message')
logging.info('info message')
logging.warning('warning message')
logging.error('error message')
logging.critical('critical message')

def load_lstm_model(path):
    model = load_model(path)
    return model
def load_test_data(path = "./test_data/test_data_01_08"):
    with open(path, 'rb') as f:
        test_x, test_y = pickle.load(f)
        return test_x,test_y

def testModel():
    model = load_lstm_model("./lstm_model_01_07")
    test_x, test_y = load_test_data()
    pred_result = model.predict(test_x, batch_size=1)
    pred_y = []
    real_y = []
    for i,p_y in enumerate(pred_result):
        pred = np.argmax(p_y)
        pred_y.append(pred + 1)
        real = np.argmax(test_y[i])
        real_y.append(real+1)
    return real_y, pred_y
def get_accuracy_perClass(test_y,pred_y,classNum ):
    target_names = ["class" + str(i+1) for i in range(0,classNum)]
    res = np.zeros((classNum,classNum))
    for i,real in enumerate(test_y):
        temp =  res[real-1,pred_y[i]-1]
        res[real - 1, pred_y[i]-1] = temp +1
    print(classification_report(test_y,
                                pred_y,
                                target_names=target_names))

    data = pd.DataFrame(res.tolist(),
                        index=["真实"+str(i) for i in range(1,classNum+1)],
                        columns = ["预测"+str(i) for i in range(1,classNum+1)] )
    pd.set_option('display.max_columns',1000)
    print(data)
if __name__ == "__main__":
    test_y, pred_y = testModel()
    get_accuracy_perClass(test_y,pred_y,9)