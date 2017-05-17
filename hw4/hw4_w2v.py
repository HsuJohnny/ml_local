import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import word2vec
from sklearn.manifold import TSNE
import numpy as np
from adjustText import adjust_text
from nltk.tag.perceptron import PerceptronTagger

#word2vec.word2phrase('Book5TheOrderOfThePhoenix/all.txt', 'all_phrase.txt', verbose=True)
#word2vec.word2vec('all_phrase.txt', 'all.bin', size=100, verbose=True)

print ("loading...")
model = word2vec.load('all.bin')
voc = model.vocab
top800 = voc[0:500]

print ("tagging...")
pretrain = PerceptronTagger()
top800_tag = pretrain.tag(top800)
top800_plot = []
discard = ["(", ")", ".", ",", ":", ";", "!", "?", "'", "\"", "-", "_", "“", "’"]
wanted = ['JJ', 'NNP', 'NN', 'NNS']
flag = True
for item in top800_tag:
	if (len(item[0])>1):
		for x in item[0]:
			if x in discard:
				flag = False
				break	
		if (item[1] in wanted and flag):
			top800_plot.append(item[0])
		flag = True
#print (top800_plot)
top800_vec = []
for word in top800_plot:
	vec = model[word]
	top800_vec.append(vec)
top800_vec = np.array(top800_vec)
#print (top800_vec[0])

print ("doing TSNE...")
t_model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
output = t_model.fit_transform(top800_vec)

nor_output = (output - np.mean(output, axis=0))/np.std(output, axis=0)

print ("plotting...")
plt.figure(figsize=(10,10))
"""
for i, label in enumerate(top800_plot):
	x, y = nor_output[i, :]
	plt.scatter(x,y)
	
	plt.annotate(label,
				 xy=(x,y),
				 xytext=(5,2),
				 textcoords='offset points',
				 ha='right',
				 va='bottom')
"""
xs = []
ys = []
for item in output:
	xs.append(item[0])
	ys.append(item[1]) 

plt.scatter(xs,ys)
texts = []
for x, y, label in zip(xs, ys, top800_plot):
	texts.append(plt.text(x, y, label))
plt.title(str(
			 adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5)))
		     +' iterations')
plt.show()
