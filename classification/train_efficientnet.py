import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

DATA_DIR = "dataset/smartvision_dataset/classification"

train = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=(224,224), batch_size=32
)
val = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val", image_size=(224,224), batch_size=32
)

base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
base.trainable = False

x = Dense(256, activation="relu")(base.output)
x = Dropout(0.5)(x)
output = Dense(25, activation="softmax")(x)

model = Model(base.input, output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=10)
model.save("models/classification/efficientnetb0.h5")
