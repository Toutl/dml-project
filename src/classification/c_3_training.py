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

# %% [markdown]
# - `Crespo Neri, Diego Ubaldo` (`ubaldo.crespo@iteso.mx`)
# - `Garibay Zepeda, Julio Andrés` (`julio.garibay@iteso.mx`)
# - `Vázquez Sandoval, Isaac Ernesto` (`isaac.vazquez@iteso.mx`)

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Bibliotecas

# %%
# Básicas
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# Swag
style = "/home/toutl/code/.machine.mplstyle"
if os.path.exists(style):
    plt.style.use(style)

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Data:

# %% [markdown]
# ##### 1. Lectura de los datos

# %%
# Paths de las carpetas donde se encuentran los datos
data_folder = Path("../../data/classification/")
processed_folder = data_folder / "processed"

# %%
# Leemos la metadata (etiquetas e ids)
metadata_df = pd.read_csv(processed_folder / "sample_64x64_10570.csv").astype(str)
display(metadata_df.head())

# %%
# Cargamos las imágenes procesadas
processed_images = np.load(processed_folder / "sample_64x64_10570.npz")
files = processed_images.files
print(files[:10])
print(len(files))

# %%
# Ejemplo de uso de una imágen
img_name = files[0]
idx = metadata_df.index[metadata_df["image_id"] == img_name][0]
img_class = metadata_df.at[idx, "class"]

plt.imshow(processed_images[img_name], cmap="gray")
plt.title(f"Image {img_name}. Class {img_class}")
plt.axis("off")
plt.show()

# %%
# Obtenemos un arreglo con todas las imágenes
images_array = np.empty((len(metadata_df), 64, 64))

for i, img_name in enumerate(processed_images):
    image = processed_images[img_name]

    # Asegurarse que las imágenes estén normalizadas
    if image.max() > 1 or image.min() < 0:
        raise Exception(f"ERROR.\nMax: {image.max()}. Min: {image.min()}")

    images_array[i] = image

# %%
# Revisando la cantidad de etiquetas respecto a la cantidad de imágenes, y congruencia de tamaños
print(images_array.shape, metadata_df["class"].shape)


# %% [markdown]
# ---

# %% [markdown]
# ##### 2. Separación de datos en entrenamiento y prueba

# %%
# Objeto que encapsule los datos
class Data:
    def __init__(self, X, y, random_state=35):
        # Train/test
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        # Train/valid
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )

        # Agregar canal
        self.x_train = x_train[..., None]
        self.y_train = np.asarray(y_train)

        self.x_valid = x_valid[..., None]
        self.y_valid = np.asarray(y_valid)

        self.x_test = x_test[..., None]
        self.y_test = np.asarray(y_test)

        # Crear versiones planas
        self.x_train_flat = self.x_train.reshape(len(self.x_train), -1)
        self.x_valid_flat = self.x_valid.reshape(len(self.x_valid), -1)
        self.x_test_flat = self.x_test.reshape(len(self.x_test), -1)


# %%
# Se genera el objeto para los datos
data = Data(
    X=images_array.astype("float32"),
    y=metadata_df["class"].astype("int8"),
)

# %%
# Checar
display(
    data.x_train.shape,
    data.x_train_flat.shape,
    data.y_train.shape,
    data.x_valid_flat.shape,
    data.x_valid.shape,
    data.y_valid.shape,
    data.x_test.shape,
    data.x_test_flat.shape,
    data.y_test.shape,
)

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Modelos:

# %% [markdown]
# #### Introducción

# %% [markdown]
# El primer objetivo a probar será comparar 4 modelos sencillos para una clasificación de imágenes de galaxias como se ha estado trabajando.
#
# La clasificaición será binaria siendo...
# - `0`: S, galaxias con forma espiral.
# - `1`: E, morfología elíptica.
#
# Recordando que como outliers de la clasificación se consideran otros Artefactos u Objetos distintos a galaxias.

# %% [markdown]
# Los modelos a utilizar serán 4:
#
# - **CNN** (Red Neuronal Convolucional): 
#     - Es la favorita. El estado del arte para imágenes.
#     - Representa un sistema visual, capaz de detectar bordes, texturas y formas.
#     - Aprende patrones complejos a costa de una gran cantidad de imágenes.
#
# - **MLP** (Perceptrón MultiCapa): 
#     - Generaliza más que una CNN, tiene ceguera ante la espacialidad.
#     - Util cuando no hay linealidad.
#     - No es realmente utilizada para imágenes.
#
# - **SVC** (Clasificador de Soporte Vectorial): 
#     - No entiende espacialidad ni la estructura de las imágenes.
#     - Bueno para decisiones de frontera.
#     - No muy amigable con conjuntos de datos grandes.
#
# - **LR** (Regresión Logística): 
#     - Útil como base, para reconocer la necesidad de linealidad o estructuras complejas.
#     - Bueno para establecer fronteras.

# %% [markdown]
# Las métricas consideradas para evaluar dichos modelos, deberán ser congruentes para el problema, una clasificación binaria.
#
# Por lo tanto se utilizarán:
#
# - Accuracy:
#   - Indica la proporción de aciertos.
#   - Podría cargar un ligero sesgo debido al que las clases tinene un balance de `60:40`, por esto mismo se complementa con otras métricas.
#
# - Precision, Recall y F1-score:
#   - Muestran el comportamiento del modelo con cada clase.
#   - El F1 combina precisión y recall en un solo número.
#
# - ROC–AUC:
#   - Mide la capacidad del modelo para separar las dos clases.

# %% [markdown]
# La estructura del programa es la siguiente:
# 1. Se cargan los datos y separan en conjuntos de entrenamiento, validación y prueba.
# 2. Se entrenará una instancia sencilla de cada modelo.
#     - Luego se compararán en bruto.
# 3. Con técnicas de ajuste de hiperparámetros se buscarán mejores versiones de cada modelo.
#     - Y una comparación se realizará con los que obtengan las mejores métricas de cada uno.
# 4. Finalmente, se entrenará el mejor modelo que se haya obtenido con los datos sobrantes del muestreo y con la totalidad de los obtenidos.

# %% [markdown]
# ---

# %% [markdown]
# #### Básicos
# ---

# %% [markdown]
# ##### 1. CNN

# %%
# Bibliotecas
from keras.callbacks import EarlyStopping
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam


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
def build_binary_cnn(data):
    model = Sequential()

    model.add(Input(shape=data.x_train.shape[1:]))

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=3e-4)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,  # type: ignore
        metrics=["accuracy"],
    )

    return model


# %%
# Callbacks para el fit del modelo, funcionan para encontrar el mejor epoch
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# %%
cnn_model = build_binary_cnn(data)
cnn_model.summary()

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

score_cnn = cnn_model.evaluate(data.x_test, data.y_test, verbose=0)
print(f"Accuracy CNN: {score_cnn[1] * 100:.2f}%")


# %% [markdown]
# ---

# %% [markdown]
# ##### 2. MLP

# %%
def build_binary_mlp(data):
    model = Sequential()

    model.add(Flatten(input_shape=data.x_train.shape[1:]))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(3e-4),  # type: ignore
        metrics=["accuracy"],
    )

    return model


# %%
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# %%
mlp_model = build_binary_mlp(data)
mlp_model.summary()

# %%
hist_mlp = mlp_model.fit(
    data.x_train,
    data.y_train,
    batch_size=32,
    epochs=40,
    validation_data=(data.x_valid, data.y_valid),
    callbacks=[early_stop],
)

# %%
visualize_training(hist_mlp)

score_mlp = mlp_model.evaluate(data.x_test, data.y_test, verbose="0")
print(f"Accuracy CNN: {score_mlp[1] * 100:.2f}%")

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Support Vector Classifier

# %%
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# %%
svc = SVC(kernel="rbf", C=5, gamma=0.001, class_weight="balanced")

svc.fit(data.x_train_flat, data.y_train)

y_pred = svc.predict(data.x_valid_flat)
print(classification_report(data.y_valid, y_pred))

# %%
# utilizando PCA
pca = PCA(n_components=0.95, whiten=True, random_state=42)
x_train_pca = pca.fit_transform(data.x_train_flat)
x_valid_pca = pca.transform(data.x_valid_flat)

svc = SVC(kernel="rbf", C=5, gamma=0.001, class_weight="balanced")

svc.fit(x_train_pca, data.y_train)

y_pred = svc.predict(x_valid_pca)
print(classification_report(data.y_valid, y_pred))

# %% [markdown]
# ---

# %% [markdown]
# ##### 4. Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

# %%
log_reg = LogisticRegression(max_iter=200, solver="liblinear")
log_reg.fit(data.x_train_flat, data.y_train)

# %%
from sklearn.metrics import accuracy_score

preds = log_reg.predict(data.x_test_flat)
acc = accuracy_score(data.y_test, preds)
acc

# %% [markdown]
# #### Con ajuste de hiper-parámetros
# ---

# %% [markdown]
# ##### 1. CNN

# %% [markdown]
# ---

# %% [markdown]
# ##### 2. MLP

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Support Vector Classifier

# %%
# Con CV
pipe = Pipeline(
    [
        ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
        ("svc", SVC(class_weight="balanced", kernel="rbf")),
    ]
)

param_grid = {
    # "svc__C": [4, 5, 6],
    "svc__C": [5],
    # "svc__gamma": [0.0011, 0.0012, 0.00013],
    "svc__gamma": [0.0012],
}

grid = GridSearchCV(pipe, param_grid, scoring="roc_auc", verbose=True, n_jobs=-1)

grid.fit(data.x_train_flat, data.y_train)
print(grid.best_params_)

# %%
y_pred = grid.predict(data.x_test_flat)
print(classification_report(data.y_test, y_pred))

# %% [markdown]
# ---

# %% [markdown]
# ##### 1. CNN

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Conclusiones

# %% [markdown]
#

# %% [markdown]
#
