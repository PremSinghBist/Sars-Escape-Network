from utils import *

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    concatenate, Activation, Dense, Embedding, LSTM, Reshape)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"

'''physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.experimental.set_memory_growth(physical_devices[2], True)
tf.config.experimental.set_memory_growth(physical_devices[3], True)
tf.config.experimental.set_memory_growth(physical_devices[4], True)
tf.config.experimental.set_memory_growth(physical_devices[5], True)
tf.config.experimental.set_memory_growth(physical_devices[6], True)
tf.config.experimental.set_memory_growth(physical_devices[7], True) '''


class LanguageModel(object):
    def __init__(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        #physical_devices = tf.config.list_physical_devices('GPU')
        #try:
        #    for device in physical_devices:
        #        tf.config.experimental.set_memory_growth(device, True)
        #except:
        #    # Invalid device or cannot modify virtual devices once initialized.
        #    pass

    def split_and_pad(self, *args, **kwargs):
        raise NotImplementedError('Use LM instantiation instead '
                                  'of base class.')

    def fit(self, X_cat, lengths):
        X, y = self.split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        dirname = '{}/checkpoints/{}'.format(self.cache_dir_,
                                                        self.model_name_)
        mkdir_p(dirname)
        checkpoint = ModelCheckpoint(
            '{}/{}_{}'
            .format(dirname, self.model_name_, self.hidden_dim_) +
            '-{epoch:02d}.hdf5',
            save_best_only=False, save_weights_only=False,
            mode='auto', save_freq='epoch'
            
        )

        self.model_.fit(
            X, y, epochs=self.n_epochs_, batch_size=self.batch_size_,
            shuffle=True, verbose=self.verbose_,
            callbacks=[ checkpoint ],
        )

        return self

    def predict(self, X_cat, lengths):
        #Split_and_pad
        X = self.split_and_pad(X_cat, lengths, self.seq_len_,
                                self.vocab_size_, self.verbose_)[0]

        y_pred = self.model_.predict(
            X, batch_size=self.inference_batch_size_
        )

        return y_pred

    def transform(self, X_cat, lengths, embed_fname=None):
        #Returns  featurized output X, Y  so, zeroth index is X is [X_pre, X_post] padded data
        X = self.split_and_pad(
            X_cat, lengths,
            self.seq_len_, self.vocab_size_, self.verbose_,
        )[0]

        # For now, each character in each sequence becomes a sample.
        n_samples = sum(lengths)
        if type(X) == list:
            for X_i in X:
                assert(X_i.shape[0] == n_samples)
        else:
            assert(X.shape[0] == n_samples)

        # Embed using the output of a hidden layer.
        hidden = tf.keras.backend.function(
            inputs=self.model_.input,
            outputs=self.model_.get_layer('embed_layer').output,
        )

        # Manage batching to avoid overwhelming GPU memory.
        X_embed_cat = []
        n_batches = math.ceil(n_samples / self.inference_batch_size_)
        if self.verbose_:
            tprint('Embedding...')
            prog_bar = tf.keras.utils.Progbar(n_batches)
        for batchi in range(n_batches):
            start = batchi * self.inference_batch_size_
            end = min((batchi + 1) * self.inference_batch_size_, n_samples)
            if type(X) == list:
                X_batch = [ X_i[start:end] for X_i in X ]
            else:
                X_batch = X[start:end]
            X_embed_cat.append(hidden(X_batch))


            if self.verbose_:
                prog_bar.add(1)
        X_embed_cat = np.concatenate(X_embed_cat)
        if self.verbose_:
            tprint('Done embedding.')

        X_embed = np.array([
            X_embed_cat[start:end]
            for start, end in
            iterate_lengths(lengths, self.seq_len_)
        ])

        return X_embed

    def score(self, X_cat, lengths):
        X, y_true = self.split_and_pad(
            X_cat, lengths, self.seq_len_, self.vocab_size_, self.verbose_
        )

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=[ 'accuracy' ]
        )

        metrics = self.model_.evaluate(X, y_true, verbose=self.verbose_ > 0,
                                       batch_size=self.inference_batch_size_)

        for val, metric in zip(metrics, self.model_.metrics_names):
            if self.verbose_:
                tprint('Metric {}: {}'.format(metric, val))

        return metrics[self.model_.metrics_names.index('loss')] * -len(lengths)



class BiLSTMLanguageModel(LanguageModel):
    def __init__(
            self,
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            dff=512,
            n_epochs=1,
            #1000 Orign : inferec: 1500
            batch_size=10,
            inference_batch_size=1024,
            cache_dir='.',
            model_name='bilstm',
            seed=None,
            verbose=False
    ):
        super().__init__(seed=seed,)

        #policy = mixed_precision.Policy('mixed_float16')
        #mixed_precision.set_policy(policy)

        input_pre = Input(shape=(seq_len - 1,))
        input_post = Input(shape=(seq_len - 1,))

        embed = Embedding(vocab_size + 1, embedding_dim,
                          input_length=seq_len - 1)
        x_pre = embed(input_pre)
        x_post = embed(input_post)

        for _ in range(n_hidden - 1):
            lstm = LSTM(hidden_dim, return_sequences=True)
            x_pre = lstm(x_pre)
            x_post = lstm(x_post)
        lstm = LSTM(hidden_dim)
        x_pre = lstm(x_pre)
        x_post = lstm(x_post)

        x = concatenate([ x_pre, x_post ],
                        name='embed_layer')

        #x = Dense(dff, activation='relu')(x)
        x = Dense(vocab_size + 1)(x)
        output = Activation('softmax', dtype='float32')(x)

        self.model_ = Model(inputs=[ input_pre, input_post ],
                            outputs=output)

        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.n_hidden_ = n_hidden
        self.dff_ = dff
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.inference_batch_size_ = inference_batch_size
        self.cache_dir_ = cache_dir
        self.model_name_ = model_name
        self.verbose_ = verbose
    '''
    X_cat = [[26] ,[5], [3], [4], [6] , [27], [26], [7], [2], [5], [3], [6], [27]]
    lengths = (4172 Sequences length) : [1339, 1132, 1204, 1401, 1205, 1246]  
    seq_len = Choosen(Max Seq Len) : 1584
    vocab_size : 25+2
    '''
    def split_and_pad(self, X_cat, lengths, seq_len, vocab_size, verbose):
        if X_cat.shape[0] != sum(lengths):
            raise ValueError('Length dimension mismatch: {} and {}'
                             .format(X_cat.shape[0], sum(lengths)))

        if verbose > 1:
            tprint('Splitting {} seqs...'.format(len(lengths)))
        X_seqs = [
            X_cat[start:end].flatten()
            for start, end in iterate_lengths(lengths, seq_len)
        ]
        X_pre = [
            X_seq[:i] for X_seq in X_seqs for i in range(len(X_seq))
        ]
        X_post = [
            X_seq[i + 1:] for X_seq in X_seqs for i in range(len(X_seq))
        ]
        y = np.array([
            X_seq[i] for X_seq in X_seqs for i in range(len(X_seq))
        ])

        if verbose > 1:
            tprint('Padding {} splitted...'.format(len(X_pre)))
        X_pre = pad_sequences(
            X_pre, maxlen=seq_len - 1,
            dtype='int32', padding='pre', truncating='pre', value=0
        )
        if verbose > 1:
            tprint('Padding {} splitted again...'.format(len(X_pre)))
        X_post = pad_sequences(
            X_post, maxlen=seq_len - 1,
            dtype='int32', padding='post', truncating='post', value=0
        )
        if verbose > 1:
            tprint('Flipping...')
        #Flipping the colums ie . eg: [10, 2, 5] ==> [5, 2, 10]  ie [ACB] => [BCA]
        X_post = np.flip(X_post, 1)
        X = [ X_pre, X_post ]

        if verbose > 1:
            tprint('Done splitting and padding.')
        return X, y

   

    

