import os
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, MaxPooling1D, multiply, subtract, Lambda, concatenate, Dense, Conv1D, Flatten, \
    Reshape
from keras.utils import to_categorical

tr1_file_name = 'data/sts_benchmark/tr1'
tr2_file_name = 'data/sts_benchmark/tr2'
scores_file_name = 'data/sts_benchmark/scores'


def content_provider(file_name):
    with open(file_name) as file:
        content = file.readlines()
        content = [line.split('\n')[0] for line in content]
        return content


def word_encoder(sntcs1, sntcs2):
    all_sentences = sntcs1 + sntcs2
    wrd_dict = dict()
    identifier = 1
    for sntc in all_sentences:
        for wrd in sntc.split('.')[0].split():
            if wrd not in wrd_dict:
                wrd_dict[wrd] = identifier
                identifier += 1
    return wrd_dict


def sentence_encoder(dictionary, sntcs):
    sntcs_vector = []
    for sntc in sntcs:
        sntc_vector = []
        for wrd in sntc.split('.')[0].split():
            sntc_vector.append(dictionary[wrd])
        while len(sntc_vector) < 30:
            sntc_vector.append(0)
        sntcs_vector.append(sntc_vector)
    return sntcs_vector


def to_one_hot(score):
    list = [0] * 6
    list[int(score)] = 1
    return list


sentences1 = content_provider(tr1_file_name)
sentences2 = content_provider(tr2_file_name)
word_dictionary = word_encoder(sentences1, sentences2)
print("number of unique words: ", len(word_dictionary))
sentences1 = np.array([np.array(sntc) for sntc in sentence_encoder(word_dictionary, sentences1)])
sentences2 = np.array([np.array(sntc) for sntc in sentence_encoder(word_dictionary, sentences2)])
# print(sentences1)
# print(sentences2)
scores = np.array([np.array(to_one_hot(float(item))) for item in content_provider(scores_file_name)])
# print(scores)

# mask zero?
text1 = Input(shape=(30,), dtype='int32', name='text1')
x = Embedding(output_dim=300, input_dim=2551, input_length=30)(text1)
x_pool = MaxPooling1D(pool_size=30)(x)

text2 = Input(shape=(30,), dtype='int32', name='text2')
y = Embedding(output_dim=300, input_dim=2551, input_length=30)(text2)
y_pool = MaxPooling1D(pool_size=30)(y)


x_mult_y = multiply([x_pool, y_pool])
x_minus_y = subtract([x_pool, y_pool])
abs_x_minus_y = Lambda(lambda a: (a**2)**0.5)(x_minus_y)

concatenation = concatenate([x_mult_y, abs_x_minus_y])

fcnn_input = Reshape((600,))(concatenation)
fcnn_layer_one = Dense(300, input_shape=(600,), activation='tanh')(fcnn_input)
fcnn_layer_two = Dense(6, input_shape=(300,), activation='softmax')(fcnn_layer_one)

model = Model(inputs=[text1, text2], outputs=[fcnn_layer_two])

filepath = 'weights/weights.last.hdf5'
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

model.fit([sentences1, sentences2], [scores],
          epochs=6, batch_size=300, callbacks=callbacks_list)
