import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
from scipy import misc

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


images = [] 
for i in range(481):
    image = misc.imread("hand/hand.seq%i.png" % (i+1) )
    image = image.reshape(480*512)
    images.append(image)

images = np.array(images)

print ("calculating PCA...")
pca = PCA(n_components=100, svd_solver='full')
new_images = pca.fit_transform(images)

datas = np.load('train_data.npz')
test_datas = np.load('data.npz')
X = datas['X']
y = datas['y']

pred = []
print ("calculating SVR...")
clf = SVR(C=4)
clf.fit(X,y)

test_X = []

print ("calculating eigenvalues...")
vs = get_eigenvalues(new_images)
test_X.append(vs)

test_X = np.array(test_X)
pred = clf.predict(test_X)

with open('hand_pred.csv', 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred):
        print(f'{i},{d}', file=f)
