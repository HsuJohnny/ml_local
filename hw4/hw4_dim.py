import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
import sys

arg = sys.argv

test_name = arg[1]
res = arg[2]

def get_eigenvalues(data):
    SAMPLE = 5 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

print ("loading data...")
datas = np.load('train_data.npz')
test_datas = np.load(test_name)
X = datas['X']
y = datas['y']

pred = []
print ("calculating SVR...")
clf = SVR(C=4)
clf.fit(X,y)

test_X = []
for i in range(200):
	data = test_datas['%i' % i]
	vs = get_eigenvalues(data)
	test_X.append(vs)

test_X = np.array(test_X)
pred = clf.predict(test_X)

with open(res, 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred):
        print(f'{i},{np.log(d)}', file=f)
"""
C's value:
best :2
"""