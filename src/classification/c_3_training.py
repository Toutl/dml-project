# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: venvp
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
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import tensorflow as tf

# %%
# Swag
style = "/home/toutl/code/.machine.mplstyle"
if os.path.exists(style):
    plt.style.use(style)

# %% [markdown]
# ---

# %% [markdown]
# ### Data:

# %%
data_folder = Path("../../data/classification/")
# original_folder = data_folder / "original"
processed_folder = data_folder / "processed"

# %%
# Metadata
metadata_df = pd.read_csv(processed_folder / "galaxy_morphology.csv").astype(str)
display(metadata_df.head())

# %%
# Files
processed_images = np.load(processed_folder / "processed_images.npz")
files = processed_images.files
print(np.array(files[:15]))

# %%
imgsused = 15000
data = np.array([processed_images[i] for i in files[:imgsused]])

# %%
data = np.array(data)

# %%
# Ejemplo
img_name = files[119]
idx = metadata_df.index[metadata_df["image_id"] == img_name][0]
img_class = metadata_df.at[idx, "class"]

plt.imshow(processed_images[img_name], cmap="gray")
plt.title(f"Image {img_name}. Class {img_class}")
plt.axis("off")
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ### Modelo 1. CNN:

# %%
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

def build_binary_cnn():
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu",
                     input_shape=(128, 128, 1)))
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
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    
    return model


# %%
# from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Callbacks para el fit del modelo, funcionan para encontrar el mejor epoch
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# %%
cnn_model = build_binary_cnn()
cnn_model.summary()

# %%
metadata_df['class'].value_counts()

# %%
from sklearn.model_selection import train_test_split

y = metadata_df['class'][:imgsused].map({"S":0, "E":1, "A":0})

x_train, x_test ,y_train, y_test = train_test_split(data,y, random_state=1, shuffle=True)

x_test = np.array(x_test).astype("float32")
y_test = np.array(y_test).astype("float32")
x_train =np.array(x_train).astype("float32")
y_train= np.array(y_train).astype("float32")

# %%
y = np.array(y)
display(data.shape,y[:,np.newaxis].shape)

# %%
hist_cnn = cnn_model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# %%
plt.plot(hist_cnn.history['accuracy'])
plt.plot(hist_cnn.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='lower right')
plt.show()

# training vs validation loss
plt.plot(hist_cnn.history['loss'])
plt.plot(hist_cnn.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# %%

# %%
