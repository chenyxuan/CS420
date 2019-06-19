# coding=utf-8

import codecs

from keras import layers
from keras import Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn import datasets
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def scanner (file_ptr) :
    corpus = []
    labels = []
    
    while True :
        sstr = file_ptr.readline()
        if len(sstr) == 0 :
            break
        else :
            labels.append(int(sstr.split()[0]))
            corpus.append(sstr.split()[1])

    return [corpus, labels]


def compress (X, Y) :
    res = []
    p = 0.27;
    for i in range(len(X)) :
        res.append(X[i] * (1 - p) + float(Y[i]) * p)

    return res


def extract (pred) :
    pred_1 = []

    for p in pred :
        pred_1.append(float(p[1]))

    return pred_1

def visualize_presum (X, Y, fn) :
#    cost_visual = open("./data/cost_visual.txt", "w")    
    V = []
    D = []
    for i in range(len(X)) :
        V.append(abs(X[i] - Y[i]))
    V.sort()
    
    for i in range(len(X)) :
        D.append(V[i])
        if i > 0 :
            D[i] += D[i - 1] 
#    print >> cost_visual, cost_contr
    plt.plot(V, D)
    plt.savefig("./out/" + fn)
    
def compress_all( V, VL ) :
    tot = 0
    res = []
    for i in range(len(V)) :
        tot += V[i]
        for j in range(len(VL[i])) :
            val = float(VL[i][j]) * (float(V[i]))
            if i == 0 :
                res.append(val)
            else :
                res[j] += val

    for i in range(len(res)) :
        res[i] /= tot
    return res
    
def splitter (raw) :
    res = []
    for words in raw :
        splitted = []
        for word in words :
#            print word
            splitted.append(word)
#        print splitted
        res.append(splitted)
    return res

def create_model (num_filters, kernel_sizes, vocab_size, embedding_dim, fixed_len, dense_size, optimizer_chosen, loss_chosen) :
    entry = Input(shape = (fixed_len, ))
    embedding = layers.Embedding(
                    input_dim = vocab_size, 
                    input_length = fixed_len, 
                    output_dim = embedding_dim, 
                    trainable = True
                    )(entry)
                    
    pools = []
    for kernel_size in kernel_sizes : 
        conv = layers.Conv1D(num_filters, kernel_size, activation = 'relu')(embedding)
        pool = layers.GlobalMaxPooling1D()(conv)
        pools.append(pool)
        
    pools = layers.concatenate(pools)
    dense = layers.Dense(dense_size, activation = 'relu') (pools)
    final = layers.Dense(1, activation = 'sigmoid') (dense)
    
    model = Model(inputs = [entry], outputs = [final])
    model.compile(optimizer = optimizer_chosen,
                    loss = loss_chosen,
                    metrics = ['accuracy'])
    return model
    
def main() :
    train_data = codecs.open("../input/handout/handout/train_shuffle.txt", "r", encoding = 'utf-8')
    X_train, y_train = scanner(train_data)
    X_test, y_test = scanner(train_data)

    X_train = splitter(X_train)
    X_test = splitter(X_test)
    tokenizer = Tokenizer(num_words = 4396)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    
    fixed_len = 77
    X_train = pad_sequences(X_train, padding = 'post', maxlen = fixed_len)
    X_test = pad_sequences(X_test, padding = 'post', maxlen = fixed_len)

    embedding_dim = 66

#    def create_model(num_filters, kernel_sizes, vocab_size, embedding_dim, fixed_len, dense_size, optimizer_chosen):    
    param_grid = dict(vocab_size = [vocab_size],
                      embedding_dim = [embedding_dim],
                      fixed_len = [fixed_len],
                      optimizer_chosen = ['Adadelta'],
                      num_filters = [32, 64, 128, 256],
                      kernel_sizes = [[2, 3, 4, 5], [2, 3, 4], [3, 4, 5]],
                      dense_size = [20, 22, 24, 26, 28, 32, 48],
                      batch_size = [128, 192, 256, 369, 512],
                      epochs = [5, 6, 7, 8, 9, 10],
                      loss_chosen = ['mse', 'msle', 'binary_crossentropy', 'kullback_leibler_divergence']
                      )
                      
    model = KerasClassifier(build_fn = create_model,
                            verbose = False)
    grid = RandomizedSearchCV(estimator = model, param_distributions = param_grid,
                              cv = 5, verbose = False, n_iter = 80, scoring = 'roc_auc')

    grid_result = grid.fit(X_train, y_train)
    
    print("best params: ", grid_result.best_params_)
    
    '''
    test auc:  0.9563269965594331
    best params:  {'kernel_sizes': [2, 3, 4, 5], 'vocab_size': 2458, 'fixed_len': 77, 'dense_size': 18, 'optimizer_chosen': 'Adadelta', 'embedding_dim': 66, 'batch_size': 128, 'epochs': 9, 'num_filters': 64}

    test auc:  0.9516285878877243
    best params:  {'kernel_sizes': [2, 3, 4], 'vocab_size': 2458, 'fixed_len': 77, 'dense_size': 24, 'optimizer_chosen': 'Adadelta', 'embedding_dim': 66, 'batch_size': 256, 'epochs': 9, 'num_filters': 128}

    '''
    
if __name__ == "__main__":
    main()

