import os
import pickle
import random
import datetime
import tensorflow as tf
from util import read_dataset, batch_iter
from lstm_model import lstm
from tensorflow.python.framework import graph_util
from sklearn.metrics import classification_report


# mini batch每个batch迭代
def get_batch(epoches, batch_size, train_x=None, train_y=None, seq_length=None):
    if train_x is None:
        train_x, train_y = pickle.load(open("./data/pkl/train.pkl", "rb"))
    data = list(zip(train_x, train_y, seq_length))

    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(train_y), batch_size):
            yield data[batch: (batch + batch_size)]


def train_step(model_train, batch, label):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.dropout_keep_prob: 0.5
    }
    _, summary, step, loss, accuracy, = model_train.sess.run(
        fetches=[model_train.optimizer, model_train.merged_summary_train,
                 model_train.global_step, model_train.model.loss,
                 model_train.model.accuracy],
        feed_dict=feed_dict)

    # 写入tensorBoard
    model_train.summary_writer_train.add_summary(summary, step)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))
    return step


def dev_step(model_train, batch, label, return_predict=False):
    feed_dict = {
        model_train.model.sentence: batch,
        model_train.model.label: label,
        model_train.model.dropout_keep_prob: 1.0
    }

    summary, step, loss, accuracy, predict = model_train.sess.run(
        fetches=[model_train.merged_summary_test, model_train.global_step,
                 model_train.model.loss, model_train.model.accuracy,
                 model_train.model.predict],
        feed_dict=feed_dict)

    # 写入tensorBoard
    model_train.summary_writer_test.add_summary(summary, step)
    print("\t test: step {}, loss {}, accuracy {}".format(step, loss, accuracy))



class textRnnTrain(object):
    def __init__(self):
        configenv = tf.ConfigProto()
        configenv.gpu_options.per_process_gpu_memory_fraction = 0.6
        configenv.gpu_options.allow_growth = True
        self.sess = tf.Session(config = configenv)
        self.model = lstm(num_layers=1,
                                   sequence_length=500,
                                   embedding_size=100,
                                   vocab_size=625293,
                                   rnn_size=100,
                                   num_classes=9)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # 定义optimizer
        self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.batches = get_batch(1, 160)

        # tensorBoard
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('accuracy', self.model.accuracy)
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/rnn_summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/rnn_summary/test", graph=self.sess.graph)


def main(model_train,
         train_data_path = "./train_data_small/",
         num_epochs = 25,
         batch_size = 160,
         evaluate_every = 100,
         model_path = "lstm_model_01_07"):
    # 晴空计算图

    for epoch in range(num_epochs):
        print('current epoch %s' % (epoch + 1))
        train_x, train_y, dev_x, dev_y = read_dataset(train_data_path)
        print("data load finished")
        for i in range(0, len(train_y), batch_size):
            x = train_x[i:i + batch_size]
            y = train_y[i:i + batch_size]
            try:
                step = train_step(model_train, x, y)
            except ValueError:
                print("error")
                continue
            if step % evaluate_every == 0:
                try:
                    dev_step(model_train, dev_x, dev_y)
                except ValueError:
                    continue
        # 写入序列化的pb文件
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            model_train.sess,
            graph_def,
            output_node_names=["softmaxLayer/predict"]  # < -- 参数：output_node_names，输出节点名
        )
        with tf.gfile.GFile(model_path + '.pb', mode='wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)

    # dev_step(model_train, test_x, test_y, return_predict=True)
    # print(classification_report(y_true=test_y, y_pred=predict))


if __name__ == "__main__":

    Net = textRnnTrain()
    main(Net)