import os
import tensorflow as tf
from tool import data_loader
from tool import tool_kit
from tool import base_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def train(train_data, val_data):
    model = base_model.CNN_Model()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))

    model.summary()

    model_checkpoint = ModelCheckpoint('./model/bird_classify_pre.h5',
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       period=1,
                                       save_weights_only=False,
                                       save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit_generator(train_data, epochs=5, validation_data=val_data, verbose=1,
                                  callbacks=[model_checkpoint])

    return history

def main_start():
    path = './train/source_data'

    train_data = data_loader.pre_process(path)[0]
    val_data = data_loader.pre_process(path)[1]

    tool_kit.analysis(train(train_data, val_data))

