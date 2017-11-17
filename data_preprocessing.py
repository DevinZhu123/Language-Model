import matplotlib.pyplot as plt
import operator
import numpy as np


def get_vocab_map(path_to_file):
    word_count = {}

    with open(path_to_file, 'r') as file:
        for line in file:
            words = ['START']
            words.extend(line.lower().strip().split(' '))
            words.append('END')
            for word in words:
                if word in word_count:
                    word_count[word] = word_count[word] + 1
                else:
                    word_count[word] = 1
    # print(len(word_count))
    sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    words = [x[0] for x in sorted_word_count]
    counts = [x[1] for x in sorted_word_count]
    # print(len(words))
    x_pos = np.arange(len(words))

    # plt.bar(x_pos, counts, color='g')
    # plt.xticks(x_pos, words)
    # plt.show()

    # truncate the vocabulary size to 8000, and build the word-index map.
    word_index = {}
    for i in range(7999):
        word_index[words[i]] = i
    word_index['UNK'] = 7999

    return word_index


def parse_4gram(word_index, path_to_file):
    dataset = []
    with open(path_to_file, 'r') as file:
        for line in file:
            words = ['START']
            words.extend(line.lower().strip().split(' '))
            words.append('END')
            length = len(words)
            for i in range(length-3):
                dataset.append(tuple(word_index[words[i+j]] if words[i+j] in word_index else word_index['UNK'] for j in range(4)))
    return dataset


"""

word_index = get_vocab_map('./data/train.txt')
index_word = {}
for key, value in word_index.iteritems():
    index_word[value] = key
dataset = parse_4gram(word_index, './data/train.txt')

gram_dict = {}
for gram in dataset:
    if gram in gram_dict:
        gram_dict[gram] = gram_dict[gram] + 1
    else:
        gram_dict[gram] = 1
sorted_gram_dict = sorted(gram_dict.items(), key=operator.itemgetter(1), reverse=True)

for i in range(40):
    print [[index_word[x], sorted_gram_dict[i][1]] for x in sorted_gram_dict[i][0]]
print len(sorted_gram_dict)

"""

