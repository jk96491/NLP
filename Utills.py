
def SetSentences(sentences):
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    vocaCount = len(word_dict)

    return word_dict, number_dict, vocaCount

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

