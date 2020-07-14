import numpy as np

def SetSentences(sentences):
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    vocaCount = len(word_dict)

    return word_dict, number_dict, vocaCount

def SetSentencesForWord(sentences):
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}

    return word_sequence, word_dict , word_list


def make_batch(sentenceList, word_dict):
    input_batch = []
    target_batch = []

    for curSentence in sentenceList:
        word = curSentence.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


def random_batch(data, size, voc_size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])
        random_labels.append(data[i][1])

    return random_inputs, random_labels


def SetSkipGram(word_sequence, word_dict):
    skip_grams = []

    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

        for w in context:
            skip_grams.append([target, w])

    return skip_grams

