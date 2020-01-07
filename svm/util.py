import pickle
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
def read_data():
    X,_,Y = pickle.load(open('../traindata/9_class/text_data', 'rb'))
    return X,Y
def index_map(i):
    indexmap = {
        1:1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 3,
        8: 6,
        9: 7,
        10: 8,
        11: 9
    }
    return indexmap[i]
class DataProc(object):
    '''get datas from the file'''

    def __init__(self, fileName):
        self.fileName = fileName
        self.x_data = []
        self.y_data = []
        self.readData()
    def readData(self):
        X_train, y_train = pickle.load(open(self.fileName, "rb"))
        self.x_data = np.array(X_train)                          #转换为二维数组
        self.y_data = y_train
    def getY(self):
        '''get all labels,is the first column of the array'''
        # return np.array(self.y_data).reshape(-1,1)
        return self.y_data
    def getX(self):
        '''get the data of text except the first column'''
        return self.x_data

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

def gen_text_data():
    '''
    将原来HAN 中11分类的测试数据转换为 9分类的测试数据
    :return:
    '''
    with open("../traindata/new_train/test_data_4000", 'rb') as f:
        test_x, test_y = pickle.load(f)
        new_test_y = []

        for y in test_y:
            real = np.argmax(y)
            new_test_y.append(index_map(real + 1))
    pickle.dump((test_x, new_test_y), open("./SVM_test_data", "wb"))

if __name__ == "__main__":
    gen_text_data()

