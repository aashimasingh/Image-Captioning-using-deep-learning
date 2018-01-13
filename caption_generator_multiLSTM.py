from keras.applications import inception_v3
import numpy as np
import io
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import _pickle as pickle
from keras import optimizers
from keras.regularizers import l1_l2
from keras.layers.normalization import BatchNormalization

EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load( io.open( "encoded_images_resnet.p", "rb" ) )
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iterator = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iterator.__next__()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print ("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")


    def data_generator(self, batch_size = 1024):
        partial_caps = []
        next_words = []
        images = []
        print ("Generating data...")
        gen_count = 0
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iterator = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iterator.__next__()
            caps.append(x[1][1])
            imgs.append(x[1][0])


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print ("yielding count: "+str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        image_model = Sequential()
        image_model.add(Dense(EMBEDDING_DIM, input_dim = 2048, activation='relu'))

        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        
        model.add(LSTM(256,return_sequences=True,dropout=0.13,recurrent_dropout=0.13,kernel_regularizer=l1_l2(8.831598074868035e-08,1.3722161194141783e-07),kernel_initializer='he_normal',implementation=2))
        model.add(LSTM(256,return_sequences=True,dropout=0.13,recurrent_dropout=0.13,kernel_regularizer=l1_l2(8.831598074868035e-08,1.3722161194141783e-07),kernel_initializer='he_normal',implementation=2))
        model.add(LSTM(256,return_sequences=False,dropout=0.13,recurrent_dropout=0.13,kernel_regularizer=l1_l2(8.831598074868035e-08,1.3722161194141783e-07),kernel_initializer='he_normal',implementation=2))
        model.add((Dense(self.vocab_size)))
        model.add(Activation('softmax'))

        print ("Model created!")

        if(ret_model==True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self,index):
        return self.index_word[index]