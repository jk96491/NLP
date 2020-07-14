import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import Utills
import Task_02_Word2Vec.Trainer as Trainer

dtype = torch.FloatTensor

sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence, word_dict, word_list = Utills.SetSentencesForWord(sentences)

batch_size = 20
embedding_size = 2
maxEpoch = 10000

voc_size = len(word_list)
skip_grams = Utills.SetSkipGram(word_sequence, word_dict)

model = Trainer.train(maxEpoch, skip_grams, batch_size, voc_size, embedding_size, dtype)

for i, label in enumerate(word_list):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()












