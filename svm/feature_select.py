import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from util import read_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


max_features = 2000
word_tokenizer = WordPunctTokenizer()

X, Y = read_data()
def tf_idf():
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_test, y_test = pickle.load(open("./SVM_test_data", "rb"))
    t_vec_s = TfidfVectorizer(analyzer='word', stop_words='english',
                              max_features=max_features,
                              tokenizer=word_tokenizer.tokenize)
    t_vec_s.fit(X)
    feratures = t_vec_s.get_feature_names()
    d1 = pd.DataFrame(feratures,columns=["特征"])
    # d1.to_excel("./features.xlsx")
    print("tf idf 特征选择完毕")
    # X_train = t_vec_s.transform(X_train)
    # X_train = X_train.toarray()
    X_test = t_vec_s.transform(X_test)
    X_test = X_test.toarray()
    pickle.dump(t_vec_s, open("./tf-idf-transformer", "wb"))
    # pickle.dump((X_train,y_train),open("./train_data","wb"))
    pickle.dump((X_test,y_test),open("./test_data_tf-idf_vect","wb"))
    print("tfidf 生成完毕!")

if __name__ == '__main__':
    tf_idf()