import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model
from keras.layers import Input, Dense, Embedding, Add, Lambda, Flatten, Reshape, concatenate, GlobalMaxPooling1D, Dot, \
    multiply, GRU
from keras.regularizers import l2
from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
from TransformerEncoder import TransformerEncoder
from TrainablePositionalEmbeddings import TransformerPositionalEmbedding

from loadDataset import loadDataset
from time import time
import logging

transformer_depth = 2
heads = 4
query_dim = 128
value_dim = 128
positional_ff_dim = 256

word_embeddings_dim = 768
news_embeddings_dim = 768
# max_news_num=10
# max_neigb_num=5
news_size = 2743
user_size = 82698

layers = [128, 512]
neighbour = [5, 10, 15, 20, 25, 30]
emb_dim = [16, 32, 64, 128, 256, 512]
clicked_num = [10]

prefix = './data/'
news_name = prefix + 'news_title_updt'
word_emb = prefix + 'emb/word_emb'
model_path = './train_model/model_epoch%d.hdf5'

title = []
with open(news_name, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split('|')[1]
        title.append(str(line))
f.close()
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(title)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(title)
# print(encoded_docs)
# pad documents to a max length of 4 words
max_sequence_len = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_sequence_len, padding='post')
# print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
with open(word_emb, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip('\n').split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 768))  # *
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# define model
def get_sentence_emb(max_sequence_len, embedding_matrix, word_embeddings_dim,
                     transformer_depth, heads,
                     query_dim, value_dim, positional_ff_dim):
    X = Input(shape=(max_sequence_len,), dtype='int32')
    wordEmb = Embedding(vocab_size, 768, weights=[embedding_matrix], trainable=False)  # *
    positional_embedding_layer = TransformerPositionalEmbedding(name='Positional_embedding')
    next_step_input = positional_embedding_layer(wordEmb(X))
    attention = [None for i in range(transformer_depth)]
    for i in range(transformer_depth):
        next_step_input, attention[i] = TransformerEncoder(word_embeddings_dim,
                                                           heads,
                                                           query_dim,
                                                           value_dim,
                                                           positional_ff_dim,
                                                           dropout_rate=0.1,
                                                           name='Transformer' + str(i))(next_step_input)

    print('next_step_input', next_step_input.shape)
    # sentence_representation = Lambda(lambda x: K.mean(x, axis=1))(next_step_input)  # 效果没下面的好
    sentence_representation = Lambda(lambda x: x[:, 0, :])(next_step_input)
    print('sentence_representation', sentence_representation.shape)
    news_emb_layer = Dense(emb_dim[3],
                           kernel_regularizer=l2(0.001),
                           kernel_initializer='glorot_normal',
                           activation='relu',
                           name='news_emb_layer')  # relu?

    news_emb = news_emb_layer(sentence_representation)
    print('news_emb', news_emb.shape)
    model = Model(inputs=[X], outputs=news_emb)
    return model


sentence_emb = get_sentence_emb(max_sequence_len,
                                embedding_matrix, word_embeddings_dim,
                                transformer_depth=transformer_depth,
                                heads=heads,
                                query_dim=128,
                                value_dim=128,
                                positional_ff_dim=256)
news_emb_mat = sentence_emb.predict(padded_docs)


# print(news_emb_mat)
# print(len(news_emb_mat[0]))
def slice(x, index):
    return x[:, index, :, :]


def slice_2(x, index):
    return x[:, index, :]


def news_ncn(news_emb, ncn_path, att_size):
    # latent_size = news_emb.shape[1].value
    neigb_num, size = ncn_path.shape[1], ncn_path.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ncn_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ncn_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(ncn_path)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([news_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(ncn_path)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([news_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='ncn_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([ncn_path, atten])
    return output


def news_nun(news_emb, nun_path, att_size):
    # latent_size = news_emb.shape[1].value
    neigb_num, size = nun_path.shape[1], nun_path.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='nun_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='nun_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(nun_path)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([news_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(nun_path)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([news_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='nun_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([nun_path, atten])
    return output


def news_ntn(news_emb, ntn_path, att_size):
    # latent_size = news_emb.shape[1].value
    neigb_num, size = ntn_path.shape[1], ntn_path.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ntn_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ntn_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(ntn_path)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([news_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(ntn_path)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([news_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='ntn_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([ntn_path, atten])
    return output


def news_nan(news_emb, nan_path, att_size):
    # latent_size = news_emb.shape[1].value
    neigb_num, size = nan_path.shape[1], nan_path.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='nan_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='nan_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(nan_path)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([news_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(nan_path)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([news_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='nan_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([nan_path, atten])
    return output


def news_attention(user_emb, news_t, att_size):
    # latent_size = user_emb.shape[1].value
    neigb_num, size = news_t.shape[1], news_t.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='news_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='news_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(news_t)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([user_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(news_t)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='news_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([news_t, atten])
    return output


def user_original_emb(user_original, dim):
    out = GRU(dim)(user_original)  # 可能有问题
    return out


def user_ucu_emb(user_emb, user_ucu, neigb_num, att_size, click_num, dim):
    # gru embedding
    path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': 0})(user_ucu)  # (None,click_num,dim)
    output = GRU(dim)(path_input)  # 可能有问题 #(None,dim)

    for i in range(1, neigb_num):
        path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': i})(user_ucu)
        tmp_output = GRU(dim)(path_input)
        output = concatenate([output, tmp_output])
    ucu_neigb_input = Reshape((neigb_num, dim))(output)  # (None,neigb_num,dim)

    # attention embedding
    size = ucu_neigb_input.shape[2]
    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ucu_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='ucu_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(ucu_neigb_input)  # (None,dim)
    # metapath = Reshape((1, size))(metapath)  # (None,1,dim)
    inputs = concatenate([user_emb, metapath])
    out = (dense_layer_1(inputs))
    out = (dense_layer_2(out))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(ucu_neigb_input)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])
        tmp_out = (dense_layer_1(inputs))
        tmp_out = (dense_layer_2(tmp_out))
        out = concatenate([out, tmp_out])

    atten = Lambda(lambda x: K.softmax(x), name='ucu_attention_softmax')(out)
    out = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([ucu_neigb_input, atten])  # (None,dim)
    return out


def user_unu_emb(user_emb, user_unu, neigb_num, att_size, click_num, dim):
    # gru embedding
    path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': 0})(user_unu)
    output = GRU(dim)(path_input)  # 可能有问题

    for i in range(1, neigb_num):
        path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': i})(user_unu)
        tmp_output = GRU(dim)(path_input)
        output = concatenate([output, tmp_output])
    unu_neigb_input = Reshape((neigb_num, dim))(output)

    # attention embedding
    size = unu_neigb_input.shape[2]
    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='unu_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='unu_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(unu_neigb_input)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([user_emb, metapath])
    out = (dense_layer_1(inputs))
    out = (dense_layer_2(out))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(unu_neigb_input)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])
        tmp_out = (dense_layer_1(inputs))
        tmp_out = (dense_layer_2(tmp_out))
        out = concatenate([out, tmp_out])

    atten = Lambda(lambda x: K.softmax(x), name='unu_attention_softmax')(out)
    out = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([unu_neigb_input, atten])
    return out


def user_udu_emb(user_emb, user_udu, neigb_num, att_size, click_num, dim):
    # gru embedding
    path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': 0})(user_udu)
    output = GRU(dim)(path_input)  # 可能有问题

    for i in range(1, neigb_num):
        path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': i})(user_udu)
        tmp_output = GRU(dim)(path_input)
        output = concatenate([output, tmp_output])
    udu_neigb_input = Reshape((neigb_num, dim))(output)

    # attention embedding
    size = udu_neigb_input.shape[2]
    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='udu_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='udu_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(udu_neigb_input)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([user_emb, metapath])
    out = (dense_layer_1(inputs))
    out = (dense_layer_2(out))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(udu_neigb_input)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])
        tmp_out = (dense_layer_1(inputs))
        tmp_out = (dense_layer_2(tmp_out))
        out = concatenate([out, tmp_out])

    atten = Lambda(lambda x: K.softmax(x), name='udu_attention_softmax')(out)
    out = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([udu_neigb_input, atten])
    return out


def user_uou_emb(user_emb, user_uou, neigb_num, att_size, click_num, dim):
    # gru embedding
    path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': 0})(user_uou)
    output = GRU(dim)(path_input)  # 可能有问题

    for i in range(1, neigb_num):
        path_input = Lambda(slice, output_shape=(click_num, dim), arguments={'index': i})(user_uou)
        tmp_output = GRU(dim)(path_input)
        output = concatenate([output, tmp_output])
    uou_neigb_input = Reshape((neigb_num, dim))(output)

    # attention embedding
    size = uou_neigb_input.shape[2]
    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='uou_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='uou_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(uou_neigb_input)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([user_emb, metapath])
    out = (dense_layer_1(inputs))
    out = (dense_layer_2(out))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(uou_neigb_input)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])
        tmp_out = (dense_layer_1(inputs))
        tmp_out = (dense_layer_2(tmp_out))
        out = concatenate([out, tmp_out])

    atten = Lambda(lambda x: K.softmax(x), name='uou_attention_softmax')(out)
    out = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([uou_neigb_input, atten])
    return out


def user_attention(news_emb, user_neigb, att_size):
    latent_size = news_emb.shape[1]
    neigb_num, size = user_neigb.shape[1], user_neigb.shape[2]

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_attention_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_attention_layer_2')
    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(user_neigb)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([news_emb, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(user_neigb)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([news_emb, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='user_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([user_neigb, atten])  # (None,dim)
    return output


def get_model(news_emb_mat, neighbour_num, neighour_kind, click_num, dim):
    news_input = Input(shape=(1,), dtype='int32')
    user_input = Input(shape=(click_num,), dtype='int32')
    news_ncn_input = Input(shape=(neighbour_num,), dtype='int32')
    news_ntn_input = Input(shape=(neighbour_num,), dtype='int32')
    news_nan_input = Input(shape=(neighbour_num,), dtype='int32')
    news_nun_input = Input(shape=(neighbour_num,), dtype='int32')
    user_neigb_ucu_input = Input(shape=(neighbour_num * click_num,), dtype='int32')
    user_neigb_unu_input = Input(shape=(neighbour_num * click_num,), dtype='int32')
    user_neigb_udu_input = Input(shape=(neighbour_num * click_num,), dtype='int32')
    user_neigb_uou_input = Input(shape=(neighbour_num * click_num,), dtype='int32')

    newsEmb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    news_original = newsEmb(news_input)  # (None,1,dim)
    news_original = Lambda(lambda x: x[:, 0, :])(news_original)# (None,dim)

    user_original_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    user_original = user_original_Emb(user_input)  # (None,click_num,dim)
    user_original = user_original_emb(user_original, dim)  # (None,dim)
    # user_original = Reshape((1, dim))(user_original)  # (None,1,dim)

    news_ncn_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    ncn_neigb = news_ncn_Emb(news_ncn_input)

    news_ntn_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    ntn_neigb = news_ntn_Emb(news_ntn_input)

    news_nan_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    nan_neigb = news_nan_Emb(news_nan_input)

    news_nun_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    nun_neigb = news_nun_Emb(news_nun_input)  # (None,neighbour_num,dim)
    # news_representation = Lambda(lambda x: x[:, 0, :])(news_representation)

    ncn_emb = news_ncn(news_original, ncn_neigb, dim)
    ntn_emb = news_ntn(news_original, ntn_neigb, dim)
    nan_emb = news_nan(news_original, nan_neigb, dim)
    nun_emb = news_nun(news_original, nun_neigb, dim)  # (None,dim)
    # print("nun_emb shape", nun_emb.shape)

    news_t_kind = concatenate([ncn_emb, ntn_emb, nan_emb, nun_emb])
    news_t = Reshape((neighour_kind, dim))(news_t_kind)  # (None,neighour_kind,dim)
    # print("news_t_kind shape",news_t_kind.shape)
    news_emb = news_attention(user_original, news_t, dim)# (None,dim)
    # news_emb = Reshape((dim,))(news_emb)  # (None,dim)
    # print("news_emb shape",news_emb.shape)

    user_neigb_ucu_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    user_neigb_ucu = user_neigb_ucu_Emb(user_neigb_ucu_input)  # (None,neighbour_num*click_num,dim)
    user_ucu = Reshape((neighbour_num, click_num, dim))(Flatten()(user_neigb_ucu))  # (None,neighbour_num,click_num,dim)
    user_neigb_ucu_emb = user_ucu_emb(user_original, user_ucu, neighbour_num, dim, click_num, dim)  # (None,dim)
    # user_neigb_ucu_emb = Reshape((1, dim))(user_neigb_ucu_emb)  # (None,1,dim)
    # print("user_neigb_ucu_emb shape",user_neigb_ucu_emb.shape)

    user_neigb_unu_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    user_neigb_unu = user_neigb_unu_Emb(user_neigb_unu_input)
    user_unu = Reshape((neighbour_num, click_num, dim))(Flatten()(user_neigb_unu))
    user_neigb_unu_emb = user_unu_emb(user_original, user_unu, neighbour_num, dim, click_num, dim)
    # user_neigb_unu_emb = Reshape((1, dim))(user_neigb_unu_emb)

    user_neigb_udu_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    user_neigb_udu = user_neigb_udu_Emb(user_neigb_udu_input)
    user_udu = Reshape((neighbour_num, click_num, dim))(Flatten()(user_neigb_udu))
    user_neigb_udu_emb = user_udu_emb(user_original, user_udu, neighbour_num, dim, click_num, dim)
    # user_neigb_udu_emb = Reshape((1, dim))(user_neigb_udu_emb)

    user_neigb_uou_Emb = Embedding(news_size, dim, weights=[news_emb_mat], trainable=False)
    user_neigb_uou = user_neigb_uou_Emb(user_neigb_uou_input)
    user_uou = Reshape((neighbour_num, click_num, dim))(Flatten()(user_neigb_uou))
    user_neigb_uou_emb = user_uou_emb(user_original, user_uou, neighbour_num, dim, click_num, dim)
    # user_neigb_uou_emb = Reshape((1, dim))(user_neigb_uou_emb)

    user_neigb = concatenate([user_neigb_ucu_emb, user_neigb_unu_emb, user_neigb_udu_emb, user_neigb_uou_emb])
    user_neigb = Reshape((neighour_kind, dim))(user_neigb)  # (None,neighour_kind,dim)

    user_emb = user_attention(news_original, user_neigb, dim)  # (None,dim)
    # user_emb = Reshape((1, dim))(user_emb)  # (None,1,dim)

    # news=news_original
    # user=user_original
    news = concatenate([news_original, news_emb])
    user = concatenate([user_original, user_emb])
    # news = Lambda(lambda x: x[:, 0, :])(news)
    # user = Lambda(lambda x: x[:, 0, :])(user)
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation='relu',
                      name='news_layer%d' % idx)
        news = layer(news)
    # layer = Dense(dim,
    #               kernel_regularizer=l2(0.001),
    #               kernel_initializer='glorot_normal',
    #               activation='relu',
    #               name='news_layer')
    # news = layer(news)  # 是否多加几层全连接层#(None,dim)?

    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation='relu',
                      name='user_layer%d' % idx)
        user = layer(user)
    # layer = Dense(dim,
    #               kernel_regularizer=l2(0.001),
    #               kernel_initializer='glorot_normal',
    #               activation='relu',
    #               name='user_layer')
    # user = layer(user)  # (None,dim)?
    print('user shape', user.shape)

    dot = Dot(axes=(1, 1), normalize=True)
    score = dot([news, user])
    model = Model(inputs=[news_input, user_input, news_ncn_input, news_ntn_input, news_nan_input, news_nun_input,
                          user_neigb_ucu_input, user_neigb_unu_input, user_neigb_udu_input, user_neigb_uou_input],
                  outputs=[score])
    return model



def get_user_neigb(u, udu_neigb_list, user_clicked_list_train, user_udu_neigb_input, user_neigb_num, clicked_news_num,
                   k, tmp):
    udu_num = len(udu_neigb_list[u])
    t = np.array([])
    # t=[]
    if udu_num >= user_neigb_num + 1:
        for uid in udu_neigb_list[u][1:][:user_neigb_num]:
            num_u = len(user_clicked_list_train[uid])
            if num_u >= clicked_news_num:
                t = np.concatenate((t, np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]), axis=0)
                # t+=np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]
            else:
                t = np.concatenate((t, user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)), axis=0)
                # t+=user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)
    else:
        for uid in udu_neigb_list[u]:
            num_u = len(user_clicked_list_train[uid])
            if num_u >= clicked_news_num:
                t = np.concatenate((t, np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]), axis=0)
                # t += np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]
            else:
                t = np.concatenate((t, user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)), axis=0)
                # t += user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)
        for i in range(user_neigb_num - udu_num):
            tmp_t = tmp + 1
            t = np.concatenate((t, tmp_t), axis=0)
            # t+=tmp_t
    # print(t)
    user_udu_neigb_input[k] = t-1
    # user_udu_neigb_input[k]=user_udu_neigb_input[k]-1
    return user_udu_neigb_input

def get_user_neigb_test(u, udu_neigb_list, user_clicked_list_train, user_udu_neigb_input, user_neigb_num, clicked_news_num,
                   k, tmp):
    udu_num = len(udu_neigb_list[u])
    t = np.array([])
    # t=[]
    if udu_num >= user_neigb_num + 1:
        count=0
        for uid in udu_neigb_list[u][1:]:
            num_u = len(user_clicked_list_train[uid])
            if num_u==0:continue
            if num_u >= clicked_news_num:
                t = np.concatenate((t, np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]), axis=0)
                # for nid in np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]:t.append(nid)
            else:
                t = np.concatenate((t, user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)), axis=0)
                # for nid in list(user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)):t.append(nid)
            count+=1
            if count==user_neigb_num:break
        if count < user_neigb_num:
            for i in range(user_neigb_num - count):
                # for nid in (tmp + 1): t.append(nid)
                tmp_t=tmp+1
                t = np.concatenate((t, tmp_t), axis=0)
    else:
        count = 0
        for uid in udu_neigb_list[u][:]:
            num_u = len(user_clicked_list_train[uid])
            if num_u==0:continue
            if num_u >= clicked_news_num:
                t = np.concatenate((t, np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]), axis=0)
                # for nid in np.random.permutation(user_clicked_list_train[uid])[:clicked_news_num]: t.append(nid)
            else:
                t = np.concatenate((t, user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)), axis=0)
                # for nid in list(user_clicked_list_train[uid][:] + [user_clicked_list_train[uid][0]] * (clicked_news_num - num_u)): t.append(nid)
            count += 1

        for i in range(user_neigb_num - count):
            # for nid in (tmp + 1):t.append(nid)
            tmp_t = tmp + 1
            t = np.concatenate((t, tmp_t), axis=0)

    user_udu_neigb_input[k] = t
    user_udu_neigb_input[k] = user_udu_neigb_input[k] - 1
    return user_udu_neigb_input

def get_train_instances(train_list, user_clicked_list, user_clicked_list_train, ncn_neigb_list, nun_neigb_list,
                        ntn_neigb_list, nan_neigb_list,
                        ucu_neigb_list, unu_neigb_list, udu_neigb_list, uou_neigb_list, batch_size, news_neigb_num,
                        user_neigb_num, clicked_news_num, negatives_num, shuffle):
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1

    # print(num_batches_per_epoch)
    def data_generator():
        data_size = len(train_list)
        while True:
            if shuffle == True:
                np.random.shuffle(train_list)
            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, data_size)

                if end_idx - start_idx == batch_size:
                    news_input = np.zeros((batch_size * (negatives_num + 1),))
                    news_ncn_neigb_input = np.zeros((batch_size * (negatives_num + 1), news_neigb_num,))
                    news_nun_neigb_input = np.zeros((batch_size * (negatives_num + 1), news_neigb_num,))
                    news_ntn_neigb_input = np.zeros((batch_size * (negatives_num + 1), news_neigb_num,))
                    news_nan_neigb_input = np.zeros((batch_size * (negatives_num + 1), news_neigb_num,))

                    user_clicked_input = np.zeros((batch_size * (negatives_num + 1), clicked_news_num,))
                    user_ucu_neigb_input = np.zeros(
                        (batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_unu_neigb_input = np.zeros(
                        (batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_udu_neigb_input = np.zeros(
                        (batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_uou_neigb_input = np.zeros(
                        (batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    labels_input = np.zeros(batch_size * (negatives_num + 1))
                else:
                    t_batch_size = end_idx - start_idx
                    news_input = np.zeros((t_batch_size * (negatives_num + 1),))
                    news_ncn_neigb_input = np.zeros((t_batch_size * (negatives_num + 1), news_neigb_num,))
                    news_nun_neigb_input = np.zeros((t_batch_size * (negatives_num + 1), news_neigb_num,))
                    news_ntn_neigb_input = np.zeros((t_batch_size * (negatives_num + 1), news_neigb_num,))
                    news_nan_neigb_input = np.zeros((t_batch_size * (negatives_num + 1), news_neigb_num,))
                    user_clicked_input = np.zeros((t_batch_size * (negatives_num + 1), clicked_news_num,))
                    user_ucu_neigb_input = np.zeros(
                        (t_batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_unu_neigb_input = np.zeros(
                        (t_batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_udu_neigb_input = np.zeros(
                        (t_batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    user_uou_neigb_input = np.zeros(
                        (t_batch_size * (negatives_num + 1), user_neigb_num * clicked_news_num,))
                    labels_input = np.zeros(t_batch_size * (negatives_num + 1))
                k = 0

                for u, n in train_list[start_idx:end_idx]:

                    # positive
                    idx=user_clicked_list_train[u].index(n)
                    num_u = len(user_clicked_list_train[u][:idx])
                    if num_u >= clicked_news_num:
                        # user_clicked_input[k][:] = np.random.permutation(user_clicked_list_train[u])[:clicked_news_num]
                        user_clicked_input[k][:] = user_clicked_list_train[u][idx-clicked_news_num:idx]
                    else:
                        user_clicked_input[k][:] = [user_clicked_list_train[u][0]] * (
                                clicked_news_num - num_u)+user_clicked_list_train[u][:idx]
                    user_clicked_input[k] = user_clicked_input[k] - 1
                    tmp = user_clicked_input[k]

                    user_ucu_neigb_input = get_user_neigb(u, ucu_neigb_list, user_clicked_list_train,
                                                          user_ucu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    tmp_ucu = user_ucu_neigb_input[k]
                    user_unu_neigb_input = get_user_neigb(u, unu_neigb_list, user_clicked_list_train,
                                                          user_unu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    tmp_unu = user_unu_neigb_input[k]
                    user_udu_neigb_input = get_user_neigb(u, udu_neigb_list, user_clicked_list_train,
                                                          user_udu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    tmp_udu = user_udu_neigb_input[k]
                    user_uou_neigb_input = get_user_neigb(u, uou_neigb_list, user_clicked_list_train,
                                                          user_uou_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)
                    tmp_uou = user_uou_neigb_input[k]

                    num_ncn = len(ncn_neigb_list[n])
                    if num_ncn >= news_neigb_num + 1:
                        news_ncn_neigb_input[k][:] = np.random.permutation(ncn_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_ncn_neigb_input[k][:] = ncn_neigb_list[n] + [ncn_neigb_list[n][0]] * (
                                news_neigb_num - num_ncn)
                    news_ncn_neigb_input[k] = news_ncn_neigb_input[k] - 1

                    num_nun = len(nun_neigb_list[n])
                    if num_nun >= news_neigb_num + 1:
                        news_nun_neigb_input[k][:] = np.random.permutation(nun_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_nun_neigb_input[k][:] = nun_neigb_list[n] + [nun_neigb_list[n][0]] * (
                                news_neigb_num - num_nun)
                    news_nun_neigb_input[k] = news_nun_neigb_input[k] - 1

                    num_ntn = len(ntn_neigb_list[n])
                    if num_ntn >= news_neigb_num + 1:
                        news_ntn_neigb_input[k][:] = np.random.permutation(ntn_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_ntn_neigb_input[k][:] = ntn_neigb_list[n] + [ntn_neigb_list[n][0]] * (
                                news_neigb_num - num_ntn)
                    news_ntn_neigb_input[k] = news_ntn_neigb_input[k] - 1

                    num_nan = len(nan_neigb_list[n])
                    if num_nan >= news_neigb_num + 1:
                        news_nan_neigb_input[k][:] = np.random.permutation(nan_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_nan_neigb_input[k][:] = nan_neigb_list[n] + [nan_neigb_list[n][0]] * (
                                news_neigb_num - num_nan)
                    news_nan_neigb_input[k] = news_nan_neigb_input[k] - 1

                    news_input[k] = n - 1

                    labels_input[k] = 1.0
                    k += 1
                    # negative
                    for t in range(negatives_num):
                        j = np.random.randint(1, news_size + 1)
                        while j in user_clicked_list[u]:  # 是否是大范围的user_click
                            j = np.random.randint(1, news_size + 1)

                        user_clicked_input[k][:] = tmp
                        user_unu_neigb_input[k][:] = tmp_unu
                        user_ucu_neigb_input[k][:] = tmp_ucu
                        user_udu_neigb_input[k][:] = tmp_udu
                        user_uou_neigb_input[k][:] = tmp_uou

                        num_ncn = len(ncn_neigb_list[j])
                        if num_ncn >= news_neigb_num + 1:
                            news_ncn_neigb_input[k][:] = np.random.permutation(ncn_neigb_list[j][1:])[:news_neigb_num]
                        else:
                            news_ncn_neigb_input[k][:] = ncn_neigb_list[j] + [ncn_neigb_list[j][0]] * (
                                    news_neigb_num - num_ncn)
                        news_ncn_neigb_input[k] = news_ncn_neigb_input[k] - 1

                        num_nun = len(nun_neigb_list[j])
                        if num_nun >= news_neigb_num + 1:
                            news_nun_neigb_input[k][:] = np.random.permutation(nun_neigb_list[j][1:])[:news_neigb_num]
                        else:
                            news_nun_neigb_input[k][:] = nun_neigb_list[j] + [nun_neigb_list[j][0]] * (
                                    news_neigb_num - num_nun)
                        news_nun_neigb_input[k] = news_nun_neigb_input[k] - 1

                        num_ntn = len(ntn_neigb_list[j])
                        if num_ntn >= news_neigb_num + 1:
                            news_ntn_neigb_input[k][:] = np.random.permutation(ntn_neigb_list[j][1:])[:news_neigb_num]
                        else:
                            news_ntn_neigb_input[k][:] = ntn_neigb_list[j] + [ntn_neigb_list[j][0]] * (
                                    news_neigb_num - num_ntn)
                        news_ntn_neigb_input[k] = news_ntn_neigb_input[k] - 1

                        num_nan = len(nan_neigb_list[j])
                        if num_nan >= news_neigb_num + 1:
                            news_nan_neigb_input[k][:] = np.random.permutation(nan_neigb_list[j][1:])[:news_neigb_num]
                        else:
                            news_nan_neigb_input[k][:] = nan_neigb_list[j] + [nan_neigb_list[j][0]] * (
                                    news_neigb_num - num_nan)
                        news_nan_neigb_input[k] = news_nan_neigb_input[k] - 1

                        news_input[k] = j - 1

                        labels_input[k] = 0.0
                        k += 1

                yield (
                    [news_input, user_clicked_input, news_ncn_neigb_input, news_ntn_neigb_input, news_nan_neigb_input,
                     news_nun_neigb_input, user_ucu_neigb_input, user_unu_neigb_input, user_udu_neigb_input,
                     user_uou_neigb_input], labels_input)

    return num_batches_per_epoch, data_generator()


def generate_test_data(test_list, user_clicked_list, user_clicked_list_test, ncn_neigb_list, nun_neigb_list,
                       ntn_neigb_list, nan_neigb_list,
                       ucu_neigb_list, unu_neigb_list, udu_neigb_list, uou_neigb_list, batch_size, news_neigb_num,
                       user_neigb_num, clicked_news_num, negatives_num, shuffle):
    h_tmp = [0] * (user_size + 1)
    for i in test_list:
        h_tmp[i[0]] += 1
    all_test_list = []
    for uid, n_num in enumerate(h_tmp):
        if n_num != 0:
            test_user_num.append(n_num + negatives_num)
            for t in range(negatives_num):
                j = np.random.randint(1, news_size + 1)
                while j in user_clicked_list[uid]:
                    j = np.random.randint(1, news_size + 1)
                all_test_list.append([uid, j])
                test_label.append(0)
                test_nid.append(j)
            for i in test_list:
                if uid == i[0]:
                    all_test_list.append(i)
                    test_label.append(1)
                    test_nid.append(i[1])
            # for t in range(negatives_num):
            #     j = np.random.randint(1, news_size + 1)
            #     while j in user_clicked_list[uid]:
            #         j = np.random.randint(1, news_size + 1)
            #     all_test_list.append([uid,j])
            #     test_label.append(0)
            #     test_nid.append(j)

    num_batches_per_epoch = int((len(all_test_list) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(all_test_list)

        while True:
            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, data_size)

                if end_idx - start_idx == batch_size:
                    news_input = np.zeros((batch_size,))
                    news_ncn_neigb_input = np.zeros((batch_size, news_neigb_num,))
                    news_nun_neigb_input = np.zeros((batch_size, news_neigb_num,))
                    news_ntn_neigb_input = np.zeros((batch_size, news_neigb_num,))
                    news_nan_neigb_input = np.zeros((batch_size, news_neigb_num,))

                    user_clicked_input = np.zeros((batch_size, clicked_news_num,))
                    user_ucu_neigb_input = np.zeros(
                        (batch_size, user_neigb_num * clicked_news_num,))
                    user_unu_neigb_input = np.zeros(
                        (batch_size, user_neigb_num * clicked_news_num,))
                    user_udu_neigb_input = np.zeros(
                        (batch_size, user_neigb_num * clicked_news_num,))
                    user_uou_neigb_input = np.zeros(
                        (batch_size, user_neigb_num * clicked_news_num,))
                    labels_input = np.zeros(batch_size)
                else:
                    t_batch_size = end_idx - start_idx
                    news_input = np.zeros((t_batch_size,))
                    news_ncn_neigb_input = np.zeros((t_batch_size, news_neigb_num,))
                    news_nun_neigb_input = np.zeros((t_batch_size, news_neigb_num,))
                    news_ntn_neigb_input = np.zeros((t_batch_size, news_neigb_num,))
                    news_nan_neigb_input = np.zeros((t_batch_size, news_neigb_num,))
                    user_clicked_input = np.zeros((t_batch_size, clicked_news_num,))
                    user_ucu_neigb_input = np.zeros(
                        (t_batch_size, user_neigb_num * clicked_news_num,))
                    user_unu_neigb_input = np.zeros(
                        (t_batch_size, user_neigb_num * clicked_news_num,))
                    user_udu_neigb_input = np.zeros(
                        (t_batch_size, user_neigb_num * clicked_news_num,))
                    user_uou_neigb_input = np.zeros(
                        (t_batch_size, user_neigb_num * clicked_news_num,))
                    labels_input = np.zeros(t_batch_size)

                k = 0

                for u, n in all_test_list[start_idx:end_idx]:

                    # num_u = len(user_clicked_list_test[u])
                    # if num_u >= clicked_news_num:
                    #     user_clicked_input[k][:] = np.random.permutation(user_clicked_list_test[u])[:clicked_news_num]
                    # else:
                    #     user_clicked_input[k][:] = [user_clicked_list_test[u][0]] * (
                    #             clicked_news_num - num_u)+user_clicked_list_test[u][:]
                    num_u = len(user_clicked_list_train[u])
                    if num_u >= clicked_news_num:
                        user_clicked_input[k][:] = user_clicked_list_train[u][- clicked_news_num:]
                    else:
                        user_clicked_input[k][:] = [user_clicked_list_train[u][0]] * (
                                clicked_news_num - num_u) + user_clicked_list_train[u][:]
                    user_clicked_input[k] = user_clicked_input[k] - 1
                    tmp = user_clicked_input[k]

                    user_ucu_neigb_input = get_user_neigb_test(u, ucu_neigb_list, user_clicked_list_test,
                                                          user_ucu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    user_unu_neigb_input = get_user_neigb_test(u, unu_neigb_list, user_clicked_list_test,
                                                          user_unu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    user_udu_neigb_input = get_user_neigb_test(u, udu_neigb_list, user_clicked_list_test,
                                                          user_udu_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    user_uou_neigb_input = get_user_neigb_test(u, uou_neigb_list, user_clicked_list_test,
                                                          user_uou_neigb_input, user_neigb_num,
                                                          clicked_news_num, k, tmp)

                    num_ncn = len(ncn_neigb_list[n])
                    if num_ncn >= news_neigb_num + 1:
                        news_ncn_neigb_input[k][:] = np.random.permutation(ncn_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_ncn_neigb_input[k][:] = ncn_neigb_list[n] + [ncn_neigb_list[n][0]] * (
                                news_neigb_num - num_ncn)
                    news_ncn_neigb_input[k] = news_ncn_neigb_input[k] - 1

                    num_nun = len(nun_neigb_list[n])
                    if num_nun >= news_neigb_num + 1:
                        news_nun_neigb_input[k][:] = np.random.permutation(nun_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_nun_neigb_input[k][:] = nun_neigb_list[n] + [nun_neigb_list[n][0]] * (
                                news_neigb_num - num_nun)
                    news_nun_neigb_input[k] = news_nun_neigb_input[k] - 1

                    num_ntn = len(ntn_neigb_list[n])
                    if num_ntn >= news_neigb_num + 1:
                        news_ntn_neigb_input[k][:] = np.random.permutation(ntn_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_ntn_neigb_input[k][:] = ntn_neigb_list[n] + [ntn_neigb_list[n][0]] * (
                                news_neigb_num - num_ntn)
                    news_ntn_neigb_input[k] = news_ntn_neigb_input[k] - 1

                    num_nan = len(nan_neigb_list[n])
                    if num_nan >= news_neigb_num + 1:
                        news_nan_neigb_input[k][:] = np.random.permutation(nan_neigb_list[n][1:])[:news_neigb_num]
                    else:
                        news_nan_neigb_input[k][:] = nan_neigb_list[n] + [nan_neigb_list[n][0]] * (
                                news_neigb_num - num_nan)
                    news_nan_neigb_input[k] = news_nan_neigb_input[k] - 1

                    news_input[k] = n - 1

                    labels_input[k] = test_label[start_idx + k]
                    k += 1
                # for i in user_unu_neigb_input:
                #     if min(i)<0:
                #         print(i)

                yield (
                    [news_input, user_clicked_input, news_ncn_neigb_input, news_ntn_neigb_input, news_nan_neigb_input,
                     news_nun_neigb_input, user_ucu_neigb_input, user_unu_neigb_input, user_udu_neigb_input,
                     user_uou_neigb_input], labels_input)

    return num_batches_per_epoch, data_generator()


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score)


def hr_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / len(test_list)


def div_score(nid_list, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    nid_list = np.take(nid_list, order[:k])
    res = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            t = news_user_list[nid_list[i]] + news_user_list[nid_list[j]]
            co_num = len(set(news_user_list[nid_list[i]]) & set(news_user_list[nid_list[j]]))
            if len(set(t)) == 0 or co_num == 0:
                res += 1
            # print(len(set(t)))
            else:
                res += 1 - co_num / len(set(t))

    return res / (k * (k - 1) / 2)


learning_rate = 0.001
# news_neigb_num=5
# clicked_news_num=10
neighour_kind = 4
negatives_num = 4
negatives_num_test = 20
epochs = 50
batch_size = 64

logging.warning('dataset ready')
dataset = loadDataset(prefix)
train_list = dataset.train
test_list = dataset.test
user_clicked_list = dataset.user_clicked
user_clicked_list_train = dataset.user_clicked_train
user_clicked_list_test = dataset.user_clicked_test
# news_neigb_list=dataset.news_neigb_list 1
news_user_list = dataset.news_user_list
nun_neigb_list = dataset.nun_neigb_list
ncn_neigb_list = dataset.ncn_neigb_list
nan_neigb_list = dataset.nan_neigb_list
ntn_neigb_list = dataset.ntn_neigb_list
unu_neigb_list = dataset.unu_neigb_list
ucu_neigb_list = dataset.ucu_neigb_list
udu_neigb_list = dataset.udu_neigb_list
uou_neigb_list = dataset.uou_neigb_list
# a=[]
# for i in uou_neigb_list[1:]:
#     a.append(min(i))
# print(min(a))

logging.warning('model ready')
model = get_model(news_emb_mat, neighbour[1], neighour_kind, clicked_num[-1], emb_dim[3])
# model=load_model('D:/data/clone/METoNR/train_model/model_epoch11.hdf5')
logging.warning('model readied')
logging.warning('model compile ready')
model.compile(optimizer=Adam(lr=learning_rate, decay=1e-4), loss='binary_crossentropy')
logging.warning('model compile readied')
# model.summary()

for epoch in range(12,epochs):
    test_label = []
    test_user_num = []
    test_nid = []
    t1 = time()
    train_steps, train_batches = get_train_instances(train_list, user_clicked_list, user_clicked_list_train,
                                                     ncn_neigb_list, nun_neigb_list, ntn_neigb_list, nan_neigb_list,
                                                     ucu_neigb_list, unu_neigb_list, udu_neigb_list, uou_neigb_list,
                                                     batch_size, neighbour[1], neighbour[1], clicked_num[-1],
                                                     negatives_num, True)

    t = time()
    cur_model_path = model_path % epoch
    checkpoint = ModelCheckpoint(cur_model_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    print('[%.1f s] epoch %d train_steps %d' % (t - t1, epoch, train_steps))
    hist = model.fit_generator(train_batches, train_steps, epochs=1, verbose=1, callbacks=[checkpoint])
    print('training time %.1f s' % (time() - t))
    t2 = time()
    test_steps, test_batches = generate_test_data(test_list, user_clicked_list, user_clicked_list_test,
                                                  ncn_neigb_list, nun_neigb_list, ntn_neigb_list, nan_neigb_list,
                                                  ucu_neigb_list, unu_neigb_list, udu_neigb_list, uou_neigb_list,
                                                  batch_size, neighbour[1], neighbour[1], clicked_num[-1],
                                                  negatives_num_test, True)
    score = model.predict_generator(test_batches, test_steps, verbose=0)
    # print(type(score))
    # print(score)
    # print(len(score))
    # print(len(test_label))
    # all_auc=[]
    all_mrr3, all_mrr5, all_mrr10, all_mrr20 = [], [], [], []
    all_ndcg3, all_ndcg5, all_ndcg10, all_ndcg20 = [], [], [], []
    all_hr3, all_hr5, all_hr10, all_hr20 = [], [], [], []
    all_div3, all_div5, all_div10, all_div20 = [], [], [], []
    k = 0
    for i in test_user_num:
        all_hr3.append(hr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=3))
        all_hr5.append(hr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=5))
        # all_hr10.append(hr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=10))
        all_hr20.append(hr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=20))
        all_mrr3.append(mrr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=3))
        all_mrr5.append(mrr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=5))
        # all_mrr10.append(mrr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=10))
        all_mrr20.append(mrr_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=20))
        all_ndcg3.append(ndcg_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=3))
        all_ndcg5.append(ndcg_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=5))
        # all_ndcg10.append(ndcg_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=10))
        all_ndcg20.append(ndcg_score(np.array(test_label)[k:k + i], score[k:k + i, 0], k=20))
        all_div3.append(div_score(np.array(test_nid)[k:k + i], score[k:k + i, 0], k=3))
        all_div5.append(div_score(np.array(test_nid)[k:k + i], score[k:k + i, 0], k=5))
        # all_div10.append(div_score(np.array(test_nid)[k:k + i], score[k:k + i, 0], k=10))
        all_div20.append(div_score(np.array(test_nid)[k:k + i], score[k:k + i, 0], k=20))
        k += i
    tun = np.array(test_user_num) - np.array(negatives_num_test)
    print('Iteration %d [%.1f s]: [%.1f s]' % (epoch, t2 - t1, time() - t2))
    # print('hr3 = %.6f, mrr3 = %.6f, ndcg3 = %.6f, div3 = %.6f' % (
    #     np.sum(all_hr3), np.mean(all_mrr3 / tun), np.mean(all_ndcg3), np.mean(all_div3)))
    test3 = 'hr3 = %.6f, mrr3 = %.6f, ndcg3 = %.6f, div3 = %.6f\n' % (
        np.sum(all_hr3), np.mean(all_mrr3 / tun), np.mean(all_ndcg3), np.mean(all_div3))
    # print('hr5 = %.6f,mrr5 = %.6f,ndcg5 = %.6f,div5 = %.6f' % (
    #     np.sum(all_hr5), np.mean(all_mrr5 / tun), np.mean(all_ndcg5), np.mean(all_div5)))
    test5='hr5 = %.6f,mrr5 = %.6f,ndcg5 = %.6f,div5 = %.6f\n' % (
        np.sum(all_hr5), np.mean(all_mrr5 / tun), np.mean(all_ndcg5), np.mean(all_div5))
    # test10 = 'hr10 = %.6f,mrr10 = %.6f,ndcg10 = %.6f,div10 = %.6f\n' % (
    #     np.sum(all_hr10), np.mean(all_mrr10 / tun), np.mean(all_ndcg10), np.mean(all_div10))
    # print('hr10 = %.6f,mrr10 = %.6f,ndcg10 = %.6f,div10 = %.6f' % (
    #     np.sum(all_hr10), np.mean(all_mrr10 / tun), np.mean(all_ndcg10), np.mean(all_div10)))
    # print('hr20 = %.6f,mrr20 = %.6f,ndcg20 = %.6f,div20 = %.6f' % (
    #     np.sum(all_hr20), np.mean(all_mrr20 / tun), np.mean(all_ndcg20), np.mean(all_div20)))
    test20='hr20 = %.6f,mrr20 = %.6f,ndcg20 = %.6f,div20 = %.6f\n' % (
        np.sum(all_hr20), np.mean(all_mrr20 / tun), np.mean(all_ndcg20), np.mean(all_div20))
    with open('./evaluate/test.txt', 'a', encoding='utf-8') as f:
        f.writelines([test3])
        f.writelines([test5])
        # f.writelines([test10])
        f.writelines([test20])
