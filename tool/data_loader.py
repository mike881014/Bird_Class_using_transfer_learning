import os
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator

def pre_process(path):
    TRAINING_DIR = path
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        subset='training',
        target_size=(224, 224),
        class_mode='categorical'
    )

    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        subset='validation',
        target_size=(224, 224),
        class_mode='categorical'
    )

    return train_generator, validation_generator

def re_size(source, target):
    root = os.getcwd().replace("tool","")
    path = os.path.join(root,f"train\\{source}")
    dirs = os.listdir(path)

    for i in dirs:
        current = os.path.join(path, i)
        files = os.listdir(current)
        print("#~ Now Process Dir: i")
        for j in files:
            image = Image.open(os.path.join(current, j))
            img = image.convert('RGB')
            if img.size[0] > img.size[1]:
                x = abs((img.size[0] - img.size[1]) / 2)
                y = 0
                w = img.size[1]
            else:
                x = 0
                y = abs((img.size[0] - img.size[1]) / 2)
                w = img.size[0]
            # 第一个参数左上x距离，第二参数左上y距离，第三个参数x+w，第四个参数y+h
            img_c = img.crop([x, y, x + w, y + w])
            image_data = img_c.resize((224, 224))  # 缩放
            image_data.save(os.path.join(root, f"train\\{target}", i, j))  # 保存图片在本地
            print(f"#~ {j} Resize compleate.")

    print("\n#~ Mission Complete\n")