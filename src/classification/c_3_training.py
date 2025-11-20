# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: dml-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ENTRENAMIENTO DE MODELOS
# ---
#
# ### Clasificación de galaxias según su morfología

# %% [markdown]
# #### Integrantes
# - `Crespo Neri, Diego Ubaldo` (`ubaldo.crespo@iteso.mx`)
# - `Garibay Zepeda, Julio Andrés` (`julio.garibay@iteso.mx`)
# - `Vázquez Sandoval, Isaac Ernesto` (`isaac.vazquez@iteso.mx`)

# %% [markdown]
# ---

# %% [markdown]
# ### Bibliotecas

# %%
# Básicas
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# Modelos
from keras.callbacks import EarlyStopping
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

# %%
# Swag
style = "/home/toutl/code/.machine.mplstyle"
if os.path.exists(style):
    plt.style.use(style)

# %% [markdown]
# ---

# %% [markdown]
# ### Data:

# %% [markdown]
# ##### 1. Lectura de los datos

# %%
data_folder = Path("../../data/classification/")
# original_folder = data_folder / "original"
processed_folder = data_folder / "processed"

# %%
# Metadata
metadata_df = pd.read_csv(processed_folder / "sample_64x64_10570.csv").astype(str)
display(metadata_df.head())

# %%
# Files
processed_images = np.load(processed_folder / "sample_64x64_10570.npz")
files = processed_images.files
print(files[:10])
print(len(files))

# %%
# Ejemplo
img_name = files[0]
idx = metadata_df.index[metadata_df["image_id"] == img_name][0]
img_class = metadata_df.at[idx, "class"]

plt.imshow(processed_images[img_name], cmap="gray")
plt.title(f"Image {img_name}. Class {img_class}")
plt.axis("off")
plt.show()

# %%
images_array = np.empty((len(metadata_df), 64, 64))

for i, img_name in enumerate(processed_images):
    image = processed_images[img_name]
    if image.max() > 1:
        raise Exception(f"wot?: {image.max()}")
    images_array[i] = image

# %%
# Aseguramos la misma cantidad de etiquetas que de imágenes, y congruencia de tamaños
print(images_array.shape, metadata_df["class"].shape)


# %% [markdown]
# ##### 2. Preparación de datos en entrenamiento y prueba

# %%
# objeto para los datos
@dataclass
class Data:
    x_train: Any
    y_train: Any
    x_valid: Any = None
    y_valid: Any = None
    x_test: Any = None
    y_test: Any = None


# %%
# train vs test
x_train, x_test, y_train, y_test = train_test_split(
    images_array.astype("float32"),
    metadata_df["class"].astype("int8"),
    test_size=0.2,
    random_state=35,
    stratify=metadata_df["class"].astype("int8"),
)

# train vs validation
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=35,
    stratify=y_train,
)

# Se guarda en nuestro objeto
data = Data(
    x_train=x_train[..., None],
    y_train=np.array(y_train),
    x_valid=x_valid[..., None],
    y_valid=np.array(y_valid),
    x_test=x_test[..., None],
    y_test=np.array(y_test),
)


# %% [markdown]
# ---

# %% [markdown]
# ### Modelo 1. CNN:

# %%
def build_binary_cnn():
    model = Sequential()

    model.add(
        Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=data.x_train.shape[1:],
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,  # type: ignore
        metrics=["accuracy"],
    )

    return model


# %%
def visualize_training(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["training", "validation"], loc="lower right")
    plt.show()

    # training vs validation loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["training", "validation"], loc="upper right")
    plt.show()


# %%
# Callbacks para el fit del modelo, funcionan para encontrar el mejor epoch
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# %%
cnn_model = build_binary_cnn()
cnn_model.summary()

# %%
# Checar
display(
    data.x_train.shape,
    data.y_train.shape,
    data.x_valid.shape,
    data.y_valid.shape,
    data.x_test.shape,
    data.y_test.shape,
)

# %%
print(data.x_train.shape)
print(data.x_train.min(), data.x_train.max())

# %%
hist_cnn = cnn_model.fit(
    data.x_train,
    data.y_train,
    batch_size=32,
    epochs=20,
    validation_data=(data.x_valid, data.y_valid),
    callbacks=[early_stop],
)

# %%
visualize_training(hist_cnn)

# %%
score_cnn = cnn_model.evaluate(data.x_test, data.y_test, verbose=0)
print(f"Accuracy CNN: {score_cnn[1] * 100:.2f}%")
