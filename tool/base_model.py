import tensorflow as tf

def CNN_Model():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2)
    ])

    return model

def Classify_Model(model):
    top_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=model.output_shape[1:]),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])

    top_model.load_weights("./model/bird_classify_pre.h5", by_name=True)

    return top_model