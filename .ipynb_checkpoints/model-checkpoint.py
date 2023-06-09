from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation, GRU,concatenate, ZeroPadding1D, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import pickle
import numpy as np

class sp_syllabler:
    def __init__(self,e2i_ortho, ortho_input_size, latent_dim, embed_dim, max_feat=148):
        self.e2i_ortho = e2i_ortho
        self.ortho_input_size = ortho_input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat = max_feat
        self.model = self.build_model()
        
    def build_model(self):
        
        # orthographic and ipa input layers
        ortho_inputs = Input(self.ortho_input_size,)
        # first branch ortho
        x = Embedding(self.max_feat, self.embed_dim, input_length=self.ortho_input_size)(ortho_inputs)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer=regularizers.l2(1e-5)), input_shape=(self.ortho_input_size, 1))(x)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer= regularizers.l2(1e-5) ), input_shape=(self.ortho_input_size, 1))(x)
        # x = TimeDistributed(Dense(3, activation = 'softmax'))(x)
        x = ZeroPadding1D(padding=(0, self.ortho_input_size - x.shape[1]))(x)
        z = TimeDistributed(Dense(3))(x)
        z = Activation('softmax')(z)
        
        model = Model(inputs=[ortho_inputs], outputs=z)
        
        return model
    
    def ignore_class_accuracy(self, to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return ignore_accuracy
    
    def fit(self, x_tr_ortho, y_tr, x_test_ortho, y_test, ep, batch_size, save_filename, verbose=1):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.ignore_class_accuracy(0)])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        ck = ModelCheckpoint(filepath=save_filename, monitor='val_ignore_accuracy', verbose=verbose, save_best_only=True,
                             mode='max')
        Callbacks = [es, ck]
        self.model.fit( x= x_tr_ortho, y=y_tr, epochs=ep, callbacks=Callbacks, batch_size=batch_size,
                                validation_data=(x_test_ortho, y_test), verbose=verbose)
        
    def syllabify(self, word):
        inted_ortho = []
        for c in word.lower():
            inted_ortho += [self.e2i_ortho[c]]
            
        
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size, 1), verbose = 0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(word, indexes)
        return converted
    
    def raw_syllabify(self, word):
        predicted = self.model.predict(word.reshape(1, self.ortho_input_size, 1), verbose=0)[0]
        indexes = self.to_ind(predicted)
        # real_word = ""
        # for i in word:
        #    if i != 0:
        #        real_word += list(self.e2i_ortho.keys())[list(self.e2i_ortho.values()).index(i)]
        # converted = self.insert_syl(real_word, indexes)
        return indexes
    
    def to_ind(self, sequence):
        index_sequence = []
        for ind in sequence:
            index_sequence += [np.argmax(ind)]
        return index_sequence
    
    def insert_syl(self, word, indexes):
        index_list = np.where(np.array(indexes) == 2)[0]
        word_array = [*word]
        for i in range(0, len(index_list)):
            word_array.insert(index_list[i] + i + 1, '-')
        return ''.join(word_array)
    
    
class onc_to_phon:
    def __init__(self, e2i, d2i, input_size, latent_dim, embed_dim):
        self.e2i = e2i
        self.i2d = {value: key for key, value in d2i.items()}
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat_e = len(e2i) + 1
        self.max_feat_d = len(d2i) + 1
        self.model = self.build_model()
        
    def build_model(self):
        
        # orthographic and ipa input layers
        ortho_inputs = Input(self.input_size,)
        # first branch ortho
        x = Embedding(self.max_feat_e, self.embed_dim, input_length=self.input_size)(ortho_inputs)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer=regularizers.l2(1e-5)), input_shape=(self.input_size, 1))(x)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer= regularizers.l2(1e-5) ), input_shape=(self.input_size, 1))(x)
        x = ZeroPadding1D(padding=(0, self.input_size - x.shape[1]))(x)
        z = TimeDistributed(Dense(self.max_feat_d))(x)
        z = Activation('softmax')(z)
        
        model = Model(inputs=[ortho_inputs], outputs=z)
        
        return model
    
    def ignore_class_accuracy(self, to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return ignore_accuracy
    
    def fit(self, x_tr_ortho, y_tr, x_test_ortho, y_test, ep, batch_size, save_filename, verbose=1):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.ignore_class_accuracy(0)])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        ck = ModelCheckpoint(filepath=save_filename, monitor='val_ignore_accuracy', verbose=verbose, save_best_only=True,
                             mode='max')
        Callbacks = [es, ck]
        self.model.fit( x= x_tr_ortho, y=y_tr, epochs=ep, callbacks=Callbacks, batch_size=batch_size,
                                validation_data=(x_test_ortho, y_test), verbose=verbose)
        
    def ipafy(self, word):
        inted_ortho = []
        for c in word:
            try:
                inted_ortho += [self.e2i[c]]
            except:
                for a in c:
                    inted_ortho += [self.e2i[a]]
            
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.input_size, 1), verbose = 0)[0]
        indexes = self.to_ind(predicted)
        converted = [self.i2d[x] for x in indexes if x != 0]
        return converted
    
    def to_ind(self, sequence):
        index_sequence = []
        for ind in sequence:
            index_sequence += [np.argmax(ind)]
        return index_sequence