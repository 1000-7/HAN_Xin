from keras.layers import Dense, Activation, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import matplotlib.pyplot as plt
from util import read_dataset, batch_iter,read_data
import logging
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3 #指定分配30%空间
session = tf.Session(config=config)# 设置session
KTF.set_session(session)

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)



    
def buildModel( modelConfig):
#     建立RNN分类模型

    model = Sequential()
    model.add(Embedding(modelConfig["vocab_size"],
                        modelConfig["embed_size"],
                        input_length=modelConfig["input_len"],
                        ))
    model.add(SpatialDropout1D(modelConfig["dropout"]))
    model.add(LSTM(modelConfig["hidden_size"],
                               dropout=modelConfig["dropout"],
                               recurrent_dropout=modelConfig["dropout"]))
    model.add(Dense(modelConfig["num_classes"]))
    model.add(Activation("sigmoid"))
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    return model

def showTrainResult(history):
    plt.subplot(211)
    plt.plot(history.history["acc"], color="g",
             label="Train")
    plt.plot(history.history["val_acc"], color="b",
             label="Validation")
    plt.legend("best")
    
    plt.subplot(212)
    plt.plot(history.history["loss"], color="g",
             label="Train")
    plt.plot(history.history["val_loss"], color="b",
             label="Validation")
    plt.legend("best")
    
    plt.tight_layout()
    plt.show()


def main(train_data_path,
         num_epochs = 50,
         batch_size=160,
         model_path = "./model/lstm_model_01_07",
         pad_maxlen = 500):

    modelConfig = {"num_classes":9,
                   "hidden_size":32,
                   "batch_size":160,
                   "test_size":0.2,
                   "input_len" : pad_maxlen,
                   "vocab_size" : 625293,
                   "embed_size" : 200,
                   "dropout" : 0.2}
    model = buildModel(modelConfig)

    for epoch in range(num_epochs):

        print('current epoch %s' % (epoch + 1))
        train_x, train_y, dev_x, dev_y = read_dataset(train_data_path)
        history = model.fit(train_x,
                            train_y,
                            batch_size=batch_size,
                            verbose=1,
                            validation_data=(dev_x, dev_y))
    model.save(model_path)
    showTrainResult(history)




if __name__ == '__main__':
    main("train_data/")
