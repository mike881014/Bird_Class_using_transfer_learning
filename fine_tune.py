from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tool import data_loader
from tool import tool_kit
from tool import base_model

def fine_tune(pre_model, train_data, val_data):
    pre_model.add(base_model.Classify_Model(pre_model))

    pre_model.summary()

    model_checkpoint = ModelCheckpoint('./model/bird_classify_fine.h5',
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       period=1,
                                       save_weights_only=False,
                                       save_best_only=True)

    tool_kit.freeze_layer(pre_model)

    pre_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = pre_model.fit_generator(train_data, epochs=10, validation_data=val_data, verbose=1,
                                  callbacks=[model_checkpoint])

    return history

def main_start():
    model = base_model.CNN_Model()

    train_data = data_loader.pre_process('./train/target_data')[0]
    val_data = data_loader.pre_process('./train/target_data')[1]

    tool_kit.analysis(fine_tune(model, train_data, val_data))