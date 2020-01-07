#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from sklearn.svm import SVC
from util import DataProc,get_accuracy_perClass

class SVM(object):
    '''use the svm algorithm in the skicit-learn to predict the class of text'''

    def __init__(self, train_dataproc, predict_dataproc):
        self.__train_data = train_dataproc
        self.__pred_data = predict_dataproc
        self.__svc = SVC()


    def train(self):
        '''train data,and general a model'''

        train_y = self.__train_data.getY()                                  #所有训练文本类别
        train_x = self.__train_data.getX()                                  #所有训练文本内容向量

        self.__svc.fit(train_x, train_y)                                    #开始训练，

        #we can save the model in .pkl file
        self.model_presistence()                                            #将SVC对象持久化，相当于将模型持久化



    def predict(self):
        '''predict the data'''
        self.read_model()
        predict_y = self.__pred_data.getY()                                 #
        predict_x = self.__pred_data.getX()

        test_data = np.array(predict_x)

        res = self.__svc.predict(test_data)                                 #开始预测，结果在res中

        accu = 0

        for i in range(len(predict_y)):
            if predict_y[i] == res[i]:                                      #统计正确率
                accu += 1

        accu = accu / len(predict_y)


        print('the accuracy is %f' %accu)                                   #输出正确率
        return predict_y,res



    def model_presistence(self):
        '''save a model in .pkl file'''

        fileObject = open('SVM.pkl', 'wb')
        pickle.dump(self.__svc, fileObject)                                 #将SVC持久化
        fileObject.close()


    def read_model(self):
        '''load a model from a .pkl file'''

        fileName = 'SVM.pkl'
        fileObject = open(fileName, 'rb')
        self.__svc = pickle.load(fileObject)                                #读取SVC




if __name__ == '__main__':
    train_data = DataProc('./train_data')
    predict_data = DataProc('./test_data_tf-idf_vect')

    svm = SVM(train_data, predict_data)

    #开始训练
    # svm.train()
    #开始预测
    test_y,pred_y = svm.predict()
    # 预测结果分析
    get_accuracy_perClass(test_y, pred_y, 9)