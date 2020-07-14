import torch
from torch.autograd import Variable
from NNLM_01.NNLM import NNLM_Net
import Utills
from NNLM_01 import Trainer

max_Epoch = 10000

dtype = torch.FloatTensor

sentenceList = ["i like baseball Yankees",
                "i love pizza Combination",
                "i hate cheese Cake",
                "i play music Scorpions"]

word_dict, number_dict, vocaCount = Utills.SetSentences(sentenceList)

n_gram_size = 3
n_hidden = 2
m = 2

model = NNLM_Net(vocaCount, n_gram_size, m, n_hidden, dtype)

input, output = Utills.make_batch(sentenceList, word_dict)
input = Variable(torch.LongTensor(input))
output = Variable(torch.LongTensor(output))

if __name__ == '__main__':
    Trainer.train(model, input, output, max_Epoch)

    predict = model(input).data.max(1, keepdim=True)[1]
    print([sen.split()[:n_gram_size] for sen in sentenceList], '->', [number_dict[n.item()] for n in predict.squeeze()])

