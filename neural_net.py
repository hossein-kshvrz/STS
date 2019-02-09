import os
from collections import Counter
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, MaxPooling1D, multiply, subtract, Lambda, concatenate, Dense, Conv1D, Reshape
from keras.utils import to_categorical
from tqdm import tqdm

path = 'data/sts_benchmark/'

TRAIN_DATA_FILE = path + 'sts-train.csv'
TEST_DATA_FILE = path + 'sts-test.csv'

train_df = pd.read_csv(TRAIN_DATA_FILE, '\t', error_bad_lines=False)
test_df = pd.read_csv(TEST_DATA_FILE, '\t', quoting=3, error_bad_lines=False)

# sent1_file_name = 'data/sts_benchmark/sent1.txt'
# sent2_file_name = 'data/sts_benchmark/sent2.txt'
# sent1_test_file_name = 'data/sts_benchmark/sent1_test.txt'
# sent2_test_file_name = 'data/sts_benchmark/sent2_test.txt'
# scores_file_name = 'data/sts_benchmark/scores.txt'
# scores_test_file_name = 'data/sts_benchmark/scores_test.txt'


# def content_provider(file_name):
#     with open(file_name) as file:
#         content = file.readlines()
#         content = [line.split('\n')[0] for line in content]
#         return content


# sentences1 = content_provider(sent1_file_name)
# sentences2 = content_provider(sent2_file_name)
# sentences1_test = content_provider(sent1_test_file_name)
# sentences2_test = content_provider(sent2_test_file_name)
# scores = content_provider(scores_file_name)
# scores_test = content_provider(scores_test_file_name)

sentences1 = train_df['A plane is taking off.'].fillna("NAN_WORD").tolist()
sentences2 = train_df['An air plane is taking off.'].fillna("NAN_WORD").tolist()
sentences1_test = test_df['A girl is styling her hair.'].fillna("NAN_WORD").tolist()
sentences2_test = test_df['A girl is brushing her hair.'].fillna("NAN_WORD").tolist()

scores = train_df['5.000'].to_numpy()
scores_test = test_df['2.500'].to_numpy()

scores = to_categorical(scores)
scores_test = to_categorical(scores_test)

tokenizer = WordPunctTokenizer()
vocab = Counter()


def text_to_wordlist(text, lower=False):
    text = tokenizer.tokenize(text)
    if lower:
        text = [t.lower() for t in text]
    vocab.update(text)
    return text


def process_sentences(list_sentences, lower=False):
    words = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(text, lower=lower)
        words.append(txt)
    return words


all_sentences = sentences1 + sentences2 + sentences1_test + sentences2_test
words = process_sentences(all_sentences, lower=True)

print("The vocabulary contains {} unique tokens".format(len(vocab)))

model = Word2Vec(words, size=300, window=5, min_count=5, workers=8, sg=0, negative=5)
word_vectors = model.wv
print("Number of word vectors: {}".format(len(word_vectors.vocab)))

model.wv.most_similar_cosmul(positive=['she', 'man'], negative=['he'])

MAX_NB_WORDS = len(word_vectors.vocab)
MAX_SEQUENCE_LENGTH = 30

word_index = {t[0]: i + 1 for i, t in enumerate(vocab.most_common(MAX_NB_WORDS))}

sequences1 = [[word_index.get(t, 0) for t in word]
              for word in words[:len(sentences1)]]

sequences2 = [[word_index.get(t, 0) for t in word]
              for word in words[len(sentences1):len(sentences1) + len(sentences2)]]

test_sequences1 = [[word_index.get(t, 0) for t in word]
                   for word in
                   words[len(sentences1) + len(sentences2):len(sentences1) + len(sentences2) + len(sentences1_test)]]

test_sequences2 = [[word_index.get(t, 0) for t in word]
                   for word in words[len(sentences1) + len(sentences2) + len(sentences1_test):]]

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH,
                      padding="post", truncating="post", value=0.0)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH,
                      padding="post", truncating="post", value=0.0)


test_data1 = pad_sequences(test_sequences1, maxlen=MAX_SEQUENCE_LENGTH,
                           padding="post", truncating="post", value=0.0)
test_data2 = pad_sequences(test_sequences2, maxlen=MAX_SEQUENCE_LENGTH,
                           padding="post", truncating="post", value=0.0)

print('Shape of data1 tensor:', data1.shape)
print('Shape of data2 tensor:', data2.shape)

print('Shape of test_data tensor:', test_data1.shape)
print('Shape of test_data tensor:', test_data2.shape)


WV_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))

wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        wv_matrix[i] = embedding_vector
    except:
        pass

wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)

sent1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences1 = wv_layer(sent1_input)

sent2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences2 = wv_layer(sent2_input)

conved_sequences1 = Conv1D(300, kernel_size=1, activation='relu', kernel_initializer='he_uniform')(embedded_sequences1)
conved_sequences2 = Conv1D(300, kernel_size=1, activation='relu', kernel_initializer='he_uniform')(embedded_sequences2)

x_pool = MaxPooling1D(pool_size=30)(conved_sequences1)
y_pool = MaxPooling1D(pool_size=30)(conved_sequences2)


x_mult_y = multiply([x_pool, y_pool])
x_minus_y = subtract([x_pool, y_pool])
abs_x_minus_y = Lambda(lambda a: (a**2)**0.5)(x_minus_y)

concatenation = concatenate([x_mult_y, abs_x_minus_y])

fcnn_input = Reshape((600,))(concatenation)

fcnn_layer_one = Dense(300, input_shape=(600,), activation='tanh')(fcnn_input)
fcnn_layer_two = Dense(6, input_shape=(300,), activation='softmax')(fcnn_layer_one)

model = Model(inputs=[sent1_input, sent2_input], outputs=[fcnn_layer_two])

print(model.summary())

filepath = 'weights/weights.last.hdf5'
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='auto')

model.fit([data1, data2],
          scores,
          validation_data=([test_data1, test_data2], scores_test),
          epochs=50,
          batch_size=300,
          callbacks=[checkpoint])


print(model.predict([test_data1[:4], test_data2[:4]]))
print(scores[:4])
