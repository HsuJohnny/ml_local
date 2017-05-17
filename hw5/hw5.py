import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras import backend as K

max_length = 300
batch_size = 32
word_map = []

def tag2num(tags):
	global word_map
	num = []
	for tag in tags:
		index = word_map.index(tag)
		num.append(index)
	return num

print ("loading data...")
with open('train_data.csv') as f:
	lines = f.readlines()
with open('test_data') as f:
	test_lines = f.readlines()

counter = 0
tag_on = 0
tag_start = 0
tag_end = 0
text_start = 0
text_off = 1

texts = []
tags = []

lines = np.array(lines)
for line in lines:
	if(line[0] == 'i'):
		continue
	for i in range(len(line)):
		if(line[i] == ','):
			counter += 1
			if(counter == 2):
				text_off = 0
		if(line[i] == '"' and i < 6):
			tag_start = i + 1
			tag_on = 1
		if(line[i] == '"' and i != tag_start-1 and tag_on == 1):
			tag_end = i
			tag_on = 0
		if(text_off == 0):
			text_start = i + 1
			text_off = 1
	tags.append(line[tag_start:tag_end])
	texts.append(line[text_start:])
	counter = 0
	tag_end = 0
	tag_start = 0
	text_off = 1
	tag_on = 0
texts = np.array(texts)
tags = np.array(tags)

split_tags = []
flag = 0
for tag in tags:
	split_tags.append(tag.split())
for split_tag in split_tags:
	for i in range(len(split_tag)):
		if(split_tag[i] not in word_map):
			word_map.append(split_tag[i])
			flag += 1
# transform the tags into number
index_tags = []
for split_tag in split_tags:
	tag_num = tag2num(split_tag)
	index_tags.append(tag_num)
index_tags = np.array(index_tags)

token = Tokenizer(num_words=50000)
token.fit_on_texts(texts)
seq = token.texts_to_sequences(texts)
word_index = token.word_index
print ('Found %s unique tokens.' % len(word_index))
pad_seq = pad_sequences(seq)

# set the label to the same length, 38, of boolean
cat_index_tags=[]
for index_tag in index_tags:
	cat_index_tag = np.zeros(38)
	for i in index_tag:
		cat_index_tag[i] = 1
	cat_index_tags.append(cat_index_tag)
cat_index_tags = np.array(cat_index_tags)

print ("building model...")
model = Sequential()
model.add(Embedding(50000, 128))
model.add(LSTM(64, return_sequences=True, input_shape=(306, 128)))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=38, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print ("training...")
model.fit(pad_seq, cat_index_tags, batch_size=batch_size, epochs=10)
print ("training finished.")
model.save('hw5_model.h5')
score, acc = model.evaluate(pad_seq, cat_index_tags,batch_size=batch_size)
print ('\nTrain Acc:', acc)

model.predict()
