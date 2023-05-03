import matplotlib.pyplot as plt

def analysis(model):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()

def freeze_layer(pre_model):
    for layer in pre_model.layers[:1]:
        layer.trainable = False