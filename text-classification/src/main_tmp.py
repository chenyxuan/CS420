# coding=utf-8

import codecs
import matplotlib.pyplot as plt

from math import sqrt

from keras import layers
from keras import Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import datasets
from xgboost import XGBClassifier

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

def compress_two (a, b) :
	res = []
	for i in range(len(a)) :
		res.append(sqrt(a[i] * b[i]))
	return res
	
def extract (pred) :
	pred_1 = []

	for p in pred :
		pred_1.append(float(p[1]))

	return pred_1

def visualize (X, Y, fn) :
	fpr, tpr, thresholds = roc_curve(Y, X) 
	
	plt.plot(fpr, tpr)
	plt.savefig("./out/" + fn)


def visualize_presum (X, Y, fn) :
	V = []
	D = []
	for i in range(len(X)) :
		V.append(abs(X[i] - Y[i]))
	V.sort()
	
	for i in range(len(X)) :
		D.append(V[i])
		if i > 0 :
			D[i] += D[i - 1] 
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
#			print word
			splitted.append(word)
#		print splitted
		res.append(splitted)
	return res

def create_model (num_filters, kernel_sizes, vocab_size, embedding_dim, fixed_len, dense_size, optimizer_chosen) :
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
					loss = 'binary_crossentropy',
					metrics = ['accuracy'])
	return model
	
		
def main() :
	train_data = open("./data/train.txt", "r")
	eval_data = open("./data/eval.txt", "r")

	raw = scanner(train_data)
	vectorizer = CountVectorizer(analyzer = 'char')
	transformer = TfidfTransformer()
	vectorizer.fit(raw[0])
	transformer.fit(vectorizer.transform(raw[0]))

	train_features = transformer.transform(vectorizer.transform(raw[0]))
	train_labels = raw[1]

	raw = scanner(eval_data)
	eval_features = transformer.transform(vectorizer.transform(raw[0]))
	eval_labels = raw[1]

#  ------------------------------------------------------------------------------------------------------------
	
	knn = KNeighborsClassifier(n_neighbors = 27, weights = 'distance')
	knn.fit(train_features, train_labels)
	pred_k = extract(knn.predict_proba(eval_features))

	print "K-Neigbors: ", roc_auc_score(eval_labels, pred_k)

#  ------------------------------------------------------------------------------------------------------------

	mlp = MLPClassifier(solver = 'lbfgs', activation = 'relu', alpha = 1e-06, tol = 0.0001)
	mlp.fit(train_features, train_labels)
	pred_m = extract(mlp.predict_proba(eval_features))

	print "Multi-Layer Perception: ", roc_auc_score(eval_labels, pred_m)

#  ------------------------------------------------------------------------------------------------------------	

	lr = SGDClassifier(loss = 'log', max_iter = 1000, alpha = 1e-05, penalty = 'l2', tol = 1e-06)			
	lr.fit(train_features, train_labels) 
	pred_l = extract(lr.predict_proba(eval_features))

	print "Logistic Regression: ", roc_auc_score(eval_labels, pred_l)

#  ------------------------------------------------------------------------------------------------------------
	
	xgb = XGBClassifier(
		objective = 'binary:logistic',
		subsample = 0.7,
		seed = 317,
		reg_lamba = 0,
		gamma = 0.1,
		eta = 0.008,
		max_depth = 18,
		min_child_weight = 0.02)
		
	xgb.fit(train_features, train_labels)
	pred_x = extract(xgb.predict_proba(eval_features))

	print "XGBClassifier: ", roc_auc_score(eval_labels, pred_x)

#  ------------------------------------------------------------------------------------------------------------

	X = train_features
	y = train_labels

	gb = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.2)
	gb.fit(X, y)
	pred_g = extract(gb.predict_proba(eval_features))

	print "Gradient Boosting: ", roc_auc_score(eval_labels, pred_g)

#  ------------------------------------------------------------------------------------------------------------
	
	X = train_features
	y = train_labels

	rf = RandomForestClassifier(n_estimators = 400, random_state = 317)
	rf.fit(X, y)
	pred_r = extract(rf.predict_proba(eval_features))
	
	print "Random Forest: ", roc_auc_score(eval_labels, pred_r)


#  ------------------------------------------------------------------------------------------------------------

	train_data = codecs.open("./data/train.txt", "r", encoding = 'utf-8')
	eval_data = codecs.open("./data/eval.txt", "r", encoding = 'utf-8')

	X_train, y_train = scanner(train_data)
	X_test, y_test = scanner(eval_data)
	
	X_train = splitter(X_train)
	X_test = splitter(X_test)
	
	tokenizer = Tokenizer(num_words = 4396)
	tokenizer.fit_on_texts(X_train)
	X_train = tokenizer.texts_to_sequences(X_train)
	X_test = tokenizer.texts_to_sequences(X_test)
	
	vocab_size = len(tokenizer.word_index) + 1
#	print vocab_size
	
	fixed_len = 77
	X_train = pad_sequences(X_train, padding = 'post', maxlen = fixed_len)
	X_test = pad_sequences(X_test, padding = 'post', maxlen = fixed_len)

#	best params:  {'kernel_sizes': [2, 3, 4], 'vocab_size': 2458, 'fixed_len': 77, 'dense_size': 24, 'optimizer_chosen': 'Adadelta', 'embedding_dim': 66, 'batch_size': 256, 'epochs': 9, 'num_filters': 128}
#   best params:  {'kernel_sizes': [3, 4, 5], 'vocab_size': 2458, 'fixed_len': 77, 'dense_size': 22, 'optimizer_chosen': 'Adadelta', 'embedding_dim': 66, 'batch_size': 128, 'epochs': 9, 'loss_chosen': 'binary_crossentropy', 'num_filters': 256}

	embedding_dim = 66
	dense_size = 22
	num_filters = 256
	model = create_model(num_filters, [3, 4, 5], vocab_size, embedding_dim, fixed_len, dense_size, 'Adadelta');
	model.fit(X_train, y_train, epochs = 9, batch_size = 128,
						verbose = True)
	pred = model.predict(X_test)
	print "CNN: ", roc_auc_score(y_test, pred) 
	
#  ------------------------------------------------------------------------------------------------------------
	
	weights = []
	preds = []
	
	pred_k = extract(knn.predict_proba(eval_features))
	weights.append(0.13)
	preds.append(pred_k)
	
	pred_m = extract(mlp.predict_proba(eval_features))
	weights.append(0.23)
	preds.append(pred_m)
	
	pred_l = extract(lr.predict_proba(eval_features))
	weights.append(0.43)
	preds.append(pred_l)
	
	pred_x = extract(xgb.predict_proba(eval_features))
	weights.append(10.00)
	preds.append(pred_x)

	pred_g = extract(gb.predict_proba(eval_features))
	weights.append(1.67)
	preds.append(pred_g)
	
	pred_r = extract(rf.predict_proba(eval_features))	
	weights.append(10.83)
	preds.append(pred_r)
	
	pred_c = model.predict(X_test)
	weights.append(100.93)
	preds.append(pred_c)
	
#	pred_e = compress_all(weights, preds)
	pred_e = compress_two(pred_c, pred_r)
	print "Ensemble Method: ", roc_auc_score(eval_labels, pred_e)

	'''
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('ROC')
	plt.axis(xmin = 0, xmax = 1, ymin = 0, ymax = 1)
	visualize(pred_k, eval_labels, "KNN.png")
	visualize(pred_m, eval_labels, "MLP.png")
	visualize(pred_l, eval_labels, "LR.png")
	visualize(pred_x, eval_labels, "XGB.png")
	visualize(pred_g, eval_labels, "GB.png")
	visualize(pred_r, eval_labels, "RF.png")
	visualize(pred_c, eval_labels, "CNN.png")
	visualize(pred_e, eval_labels, "ALL.png")
	'''
	"""
	K-Neigbors:  0.8713173528567701
	Multi-Layer Perception:  0.9080607309890348
	Logistic Regression:  0.9246652689171543
	XGBClassifier:  0.9473337263978462
	Gradient Boosting:  0.9344798611597959
	Random Forest:  0.9466551100992531
	Ensemble Method:  0.9496043548454615
	
	
	updated at 11:21, 04.04 April
	
	K-Neigbors:  0.8713173528567701
	Multi-Layer Perception:  0.9096265164212032
	Logistic Regression:  0.9247943850616657
	XGBClassifier:  0.9472994299219605
	Gradient Boosting:  0.9397776579642722
	Random Forest:  0.9474330853059271
	Ensemble Method:  0.9504582362230299

	"""
#  ------------------------------------------------------------------------------------------------------------
	
	input_file = open("./handout/test_handout.txt", "r")
	output_file = open("./out/output_without_gb.csv", "w")
	corpus = []
	while True :
		sstr = input_file.readline()
		if len(sstr) == 0 :
			break
		else :
			corpus.append(sstr.split()[0])
	test_data = transformer.transform(vectorizer.transform(corpus))
	
	
	splitted_file = codecs.open("./handout/test_handout.txt", "r", encoding = 'utf-8')
	splitted_raw = []
	while True :
		sstr = splitted_file.readline()
		if len(sstr) == 0 :
			break
		else :
			splitted_raw.append(sstr.split()[0])
	splitted_data = splitter(splitted_raw) 
	splitted_data = tokenizer.texts_to_sequences(splitted_data)
	splitted_data = pad_sequences(splitted_data, padding = 'post', maxlen = fixed_len)

	preds = []
	
	pred_k = extract(knn.predict_proba(test_data))
	preds.append(pred_k)
	
	pred_m = extract(mlp.predict_proba(test_data))
	preds.append(pred_m)
	
	pred_l = extract(lr.predict_proba(test_data))
	preds.append(pred_l)
	
	pred_x = extract(xgb.predict_proba(test_data))
	preds.append(pred_x)

	pred_g = extract(gb.predict_proba(test_data))
	preds.append(pred_g)
	
	pred_r = extract(rf.predict_proba(test_data))	
	preds.append(pred_r)
	
	pred_c = model.predict(splitted_data)
	preds.append(pred_c)
	
#	pred_e = compress_all(weights, preds)
	pred_e = compress_two(pred_c, pred_r)
	answer = pred_e
	
	print >> output_file, "id,pred"
	for i in range(len(answer)) :
		print >> output_file, '%d,%.10f' % (i, answer[i])
	

if __name__ == "__main__":
    main()

