from __future__ import unicode_literals

import re
import string
from hazm import *

# str = '(۱۲): احمد[] پورمندى- اقدامى‌ها شجاعانه،'
# print(word_tokenize(str))
# print(list(filter(lambda s: s != '', re.compile('[ \'"،():\-_.$,\[\]!؟\n\t]').split(str))))
# s = '2	۰۰۰ ميليارد تومان افزايش خواهد يافت.'
# print(s.index('\t'))
# d = s[s.index('\t')+1:]
# print(d)
# t = '۲۰۰'
# print(list(map(int, re.findall(r'\d+', t))))
FILE_NAME = 'data/fas_newscrawl_2017_100K/fas_newscrawl_2017_100K-sentences.txt'

normalizer = Normalizer()

def read_input(input_file):
    with open(input_file, encoding='utf-8') as f:
        content = f.readlines()
        # remove number of sentence and \t
        content = [sentence[sentence.index('\t')+1:] for sentence in content]
        # hazm normalizer
        content = [normalizer.normalize(sentence) for sentence in content]
        # my tokenizer
        content = [list(filter(lambda s: s != '', re.compile('[ \'"،():\-_.$,\[\]!؟\n\t]').split(sentence)))
                   for sentence in content]
        # make all numbers english
        # content = [[list(map(int, re.findall(r'\d+', token))) for token in sentence] for sentence in content]

        return content


documents = read_input(FILE_NAME)
print(documents)

