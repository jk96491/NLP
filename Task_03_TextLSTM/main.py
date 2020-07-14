import torch
import Task_03_TextLSTM.Trainer as Trainer
import Utills

dtype = torch.FloatTensor

number_dict, vocaCount, word_dict = Utills.set_SeqData()

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

n_gram_size = 3
n_hidden = 128
maxEpoch = 1000

input_batch, target_batch = Utills.make_batchForSeqData(seq_data, word_dict, vocaCount)

model = Trainer.train(maxEpoch, input_batch, target_batch, vocaCount, n_hidden)

inputs = [sen[:3] for sen in seq_data]

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])





