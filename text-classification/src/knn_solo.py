# coding=utf-8

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
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

def visualize (X, Y, fn) :
#	cost_visual = open("./data/cost_visual.txt", "w")	
	fpr, tpr, thresholds = roc_curve(Y, X) 
	
#	print >> cost_visual, cost_contr
	plt.plot(fpr, tpr)
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

	weights = []
	preds = []
	
	kc = KNeighborsClassifier(n_neighbors = 27, weights = 'distance')
	kc.fit(train_features, train_labels)
	pred_k = extract(kc.predict_proba(eval_features))

	weights.append(0.10)
	preds.append(pred_k)
	
	print "K-Neigbors: ", roc_auc_score(eval_labels, pred_k)
	visualize(pred_k, eval_labels, "KNN.png")
	
if __name__ == "__main__":
    main()

