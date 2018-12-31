import collections
import datetime
import re
from hazm import *


# str = '(۱۲): احمد[] پورمندى- اقدامى‌ها شجاعانه،'
# print(word_tokenize(str))
# t = '۲۰۰'
# print(list(map(int, re.findall(r'\d+', t))))
FILE_NAME = 'data/fas_newscrawl_2017_100K/fas_newscrawl_2017_100K-sentences.txt'
normalizer = Normalizer()


def my_normalizer(content):
    for i in range(len(content)):
        sentence = content[i]
        # remove number of sentence and \t
        sentence = sentence[sentence.index('\t')+1:]
        # hazm normalizer
        normalizer.normalize(sentence)
        # my tokenizer
        sentence = list(filter(lambda s: s != '', re.compile('[ /\'"،؛ء–«»():\-_.$,\[\]!؟\n\t]').split(sentence)))
        content[i] = sentence
    content = [token for sentence in content for token in sentence]
    return content


def read_input(input_file):
    with open(input_file, encoding='utf-8') as f:
        content = f.readlines()
        tokens = my_normalizer(content)
        return tokens


begin = datetime.datetime.now()
document = read_input(FILE_NAME)
end = datetime.datetime.now()

with open('tokens.txt', 'w') as f:
    for i in range(len(document)):
        print(document[i], end=", ", file=f),
        if (i+1) % 30 == 0:
            print(file=f)

# in 100K sentences we have about 107K unique words (total words are about 2 millions)
unique_words = list(set(document))

print((end - begin).total_seconds())
