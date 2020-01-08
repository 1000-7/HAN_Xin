from nltk.tokenize import WordPunctTokenizer
import pickle
import random
import logging
from tqdm import tqdm
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_data(path = "./lstm_train_data"):
    X,Y = pickle.load(open('./lstm_train_data', 'rb'))
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
def text2ids( data_path ='../traindata/9_class/text_data',
              data_save_path = "./train_data/",
              train = True,
              word2id_dict_path = '../traindata/word2index_dict',
             num_classes = 9,
             pad_maxlen=500
             ):

#     将文本转化为其中词语对应的id:
    if train :
        text_list, _, label_index_list = pickle.load(open(data_path, 'rb'))
    else:
        text_list,label_index_list = pickle.load(open(data_path, 'rb'))
    vocab = pickle.load(open(word2id_dict_path, 'rb'))
    UNKNOWN = 0
    word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型
    data_x = []
    data_y = []
    for j,text in tqdm(enumerate(text_list)):
        doc = [0] * pad_maxlen
        words = word_tokenizer.tokenize(text)
        for i, word in enumerate(words):
            if i < pad_maxlen:
                doc[i] = vocab.get(word, UNKNOWN)
        data_x.append(doc)
        labels = [0] * num_classes
        if train:
            labels[label_index_list[j] - 1] = 1
        else:
            real = np.argmax(label_index_list[j])
            labels[index_map(real + 1)-1] = 1
        data_y.append(labels)
    print(len(data_y))
    if train:
        pickle.dump((data_x,data_y), open(data_save_path +'lstm_train_data', 'wb'))
    else:

        pickle.dump((data_x, data_y), open(data_save_path, 'wb'))
def split_train_data(data_path,
                     data_save_path = "./train_data/"):
    data_x, data_y = pickle.load(open(data_path, 'rb'))
    num = len(data_y)
    percent = 1
    index_list = list(range(num))
    iter = 0
    random.shuffle(index_list)
    linspace =np.linspace(0, num-1,11,dtype=np.int64)
    for i,index in enumerate(linspace[0:len(linspace)-1]):
        x = []
        y = []
        for il in index_list[index:linspace[i+1]]:
            x.append(data_x[il])
            y.append(data_y[il])
        x = x[:int(len(x) * percent)]
        y = y[:int(len(y) * percent)]
        print(np.sum(y,axis=0))
        pickle.dump((np.array(x), np.array(y)), open(data_save_path+ 'train_data_balance_' + str(iter), 'wb'))
        iter = iter+1
    logger.info("pickle dump end")
    print("train data generate finished")

    return data_x, data_y




def read_dataset(datapath):
    num = random.randint(0, 9)
    logger.info("now epoch use data: %s" % str(num))
    with open(datapath+"train_data_balance_" + str(num), 'rb') as f:
        data_x, data_y = pickle.load(f)
        length = len(data_x)
        train_x, dev_x = data_x[:int(length * 0.99)], data_x[int(length * 0.99) + 1:]
        train_y, dev_y = data_y[:int(length * 0.99)], data_y[int(length * 0.99) + 1:]
        return train_x, train_y, dev_x, dev_y


def batch(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                b[i, j, k] = word

    return b, document_sizes, sentence_sizes

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    # test data
    # text2ids(data_path="../traindata/new_train/test_data_4000",train=False,data_save_path="./test_data/test_data_01_08")
    # split_train_data(data_path='./test_data/test_data_01_08',data_save_path="./test_data/")
    # train data---------------------------
    # text2ids()
    split_train_data(data_path='./train_data/lstm_train_data')