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
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

# %%
# NN
import keras_tuner as kt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ReLU)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

# %%
# SV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# %%
# LR
from sklearn.linear_model import LogisticRegression

# %%
# Swag
style = "/home/toutl/code/.machine.mplstyle"
if os.path.exists(style):
    plt.style.use(style)

# %%
# Stablish seeds
SEED = 35

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Para keras
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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
images_ids = metadata_df["image_id"].to_list()

images_array = np.empty((len(metadata_df), 64, 64), dtype=np.float32)

for i, img_name in enumerate(images_ids):
    image = processed_images[img_name]

    # Asegurarse que las imágenes estén normalizadas
    if image.max() > 1 or image.min() < 0:
        raise ValueError(f"Image {img_name} not normalized.")

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
    def __init__(self, X, y, random_state=SEED):
        # Splits
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )

        # Agregar canal
        self.X_train = x_train[..., None]
        self.X_valid = x_valid[..., None]
        self.X_test = x_test[..., None]

        self.y_train = np.asarray(y_train)
        self.y_valid = np.asarray(y_valid)
        self.y_test = np.asarray(y_test)

        # Validar shapes
        assert self.X_train.shape[1:] == (64, 64, 1), "Unexpected image shape"
        assert len(self.X_train) == len(self.y_train), "Train feature/label mismatch"
        assert len(self.X_valid) == len(self.y_valid), "Valid feature/label mismatch"
        assert len(self.X_test) == len(self.y_test), "Test feature/label mismatch"

        # Validar labels
        unique_labels = np.unique(y)
        assert set(unique_labels) <= {0, 1}, f"Unexpected labels found: {unique_labels}"

    # Crear versiones planas
    def _flatten(self, X):
        return X.reshape(len(X), -1)

    @property
    def X_train_flat(self):
        return self._flatten(self.X_train)

    @property
    def X_valid_flat(self):
        return self._flatten(self.X_valid)

    @property
    def X_test_flat(self):
        return self._flatten(self.X_test)


# %%
# Se genera el objeto para los datos
data = Data(
    X=images_array.astype("float32"),
    y=metadata_df["class"].astype("int8"),
)

# %%
# Checar
display(data.X_test_flat.shape)


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
# - **MLP** (Perceptrón MultiCapa): 
# - **SVC** (Clasificador de Soporte Vectorial): 
# - **LR** (Regresión Logística): 

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
def visualize_training(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.ylim((0, 1))
    plt.xlim((0, 20))
    plt.legend(["training", "validation"], loc="lower right")
    plt.show()

    # training vs validation loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.ylim((0, 1))
    plt.xlim((0, 20))
    plt.legend(["training", "validation"], loc="upper right")
    plt.show()


# %%
def build_binary_cnn(data):
    model = Sequential()

    model.add(Input(shape=data.X_train.shape[1:]))

    model.add(Conv2D(filters=32, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=3e-4)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,  # type: ignore
        metrics=["accuracy", "AUC"],
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
    data.X_train,
    data.y_train,
    batch_size=32,
    epochs=20,
    validation_data=(data.X_valid, data.y_valid),
    callbacks=[early_stop],
)

# %%
visualize_training(hist_cnn)

score_cnn = cnn_model.evaluate(data.X_test, data.y_test, verbose="0")
print(f"Accuracy CNN: {score_cnn[1] * 100:.2f}%")


# %% [markdown]
# ---

# %% [markdown]
# ##### 2. MLP

# %%
def build_binary_mlp(data: Data):
    model = Sequential()

    model.add(Input(shape=data.X_train.shape[1:]))
    model.add(Flatten())

    model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(3e-4),  # type: ignore
        metrics=["accuracy", "AUC"],
    )

    return model


# %%
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# %%
mlp_model = build_binary_mlp(data)
mlp_model.summary()

# %%
hist_mlp = mlp_model.fit(
    data.X_train,
    data.y_train,
    batch_size=32,
    epochs=40,
    validation_data=(data.X_valid, data.y_valid),
    callbacks=[early_stop],
)

# %%
visualize_training(hist_mlp)

score_mlp = mlp_model.evaluate(data.X_test, data.y_test, verbose="0")
print(f"Accuracy MLP: {score_mlp[1] * 100:.2f}%")

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Support Vector Classifier

# %%
svc_model = SVC(kernel="rbf", C=5, gamma=0.001, class_weight="balanced")

svc_model.fit(data.X_train_flat, data.y_train)

# %%
y_pred = svc_model.predict(data.X_valid_flat)
print(classification_report(data.y_valid, y_pred))

# %% [markdown]
# ---

# %% [markdown]
# ##### 4. Logistic Regression

# %%
logreg_model = LogisticRegression(max_iter=200, solver="liblinear")
logreg_model.fit(data.X_train_flat, data.y_train)

# %%
y_pred = logreg_model.predict(data.X_valid_flat)
print(classification_report(data.y_valid, y_pred))


# %% [markdown]
# ---

# %% [markdown]
# ##### - Métricas

# %%
def get_scores(model, data, split="test", threshold=0.5):
    X = getattr(data, f"X_{split}")
    X_flat = getattr(data, f"X_{split}_flat")

    if "keras" in str(type(model)).lower():
        y_score = model.predict(X, verbose=0).reshape(-1)
        y_pred = (y_score >= threshold).astype(int)
        return y_score, y_pred

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_flat)
        y_pred = (y_score >= threshold).astype(int)
        return y_score, y_pred

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_flat)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        return y_score, y_pred

    y_pred = model.predict(X_flat)
    return None, y_pred


# %%
def visualize_metrics(models, data, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 200)

    plt.figure(figsize=(16, 5))

    # 1. ROC Curve
    plt.subplot(1, 3, 1)
    for name, model in models.items():

        y_score, _ = get_scores(model, data)

        if y_score is None:
            continue

        fpr, tpr, _ = roc_curve(data.y_test, y_score)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={model_auc:.3f})")

    plt.plot([0, 1], [0, 1], "w--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # 2. Precision–Recall Curve
    plt.subplot(1, 3, 2)
    for name, model in models.items():

        y_score, _ = get_scores(model, data)

        if y_score is None:
            continue

        precision, recall, _ = precision_recall_curve(data.y_test, y_score)
        plt.plot(recall, precision, label=name)

    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    # 3. F1 vs Threshold
    plt.subplot(1, 3, 3)
    for name, model in models.items():

        y_score, _ = get_scores(model, data)

        if y_score is None:
            continue

        f1_scores = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            f1_scores.append(f1_score(data.y_test, y_pred, zero_division=0))

        plt.plot(thresholds, f1_scores, label=name)

    plt.title("F1 vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def evaluate_models2(models: dict, data: Data, threshold=0.5):
    results = []

    for name, model in models.items():

        # Predicción
        start = time.time()

        if hasattr(model, "predict") and "keras" in str(type(model)).lower():
            y_prob = model.predict(data.X_test, verbose=0).reshape(-1)
            y_pred = (y_prob >= threshold).astype(int)

        else:
            check_is_fitted(model)

            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(data.X_test_flat)
                y_pred = model.predict(data.X_test_flat)

            elif hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(data.X_test_flat)[:, 1]
                y_pred = model.predict(data.X_test_flat)

            else:
                y_prob = None
                y_pred = model.predict(data.X_test_flat)

        runtime = time.time() - start

        # Métricas
        acc = accuracy_score(data.y_test, y_pred)
        prec = precision_score(data.y_test, y_pred, zero_division=0)
        rec = recall_score(data.y_test, y_pred, zero_division=0)
        f1 = f1_score(data.y_test, y_pred, zero_division=0)

        # ROC–AUC si hay probabilidades
        if y_prob is not None:
            try:
                auc = roc_auc_score(data.y_test, y_prob)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan

        # Complejidad aprox (número de parámetros)
        if hasattr(model, "count_params"):
            params = model.count_params()
        elif hasattr(model, "coef_"):
            params = model.coef_.size + model.intercept_.size
        # elif "svc" in name:
        #     params = model.support_vectors_.shape[0]
        else:
            params = np.nan

        results.append(
            {
                "Modelo": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC-AUC": auc,
                "Parámetros": params,
                "Tiempo_inferencia_s": runtime,
            }
        )

    return pd.DataFrame(results)


# %%
def plot_confusion_matrices(models, data, threshold=0.5):
    n = len(models)
    plt.figure(figsize=(4 * n, 4))

    for i, (name, model) in enumerate(models.items(), 1):
        _, y_pred = get_scores(model, data, threshold=threshold)
        cm = confusion_matrix(data.y_test, y_pred)

        plt.subplot(1, n, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# %%
basic_models = {
    "cnn": cnn_model,
    "mlp": mlp_model,
    "svc": svc_model,
    "logreg": logreg_model,
}

# %%
visualize_metrics(models=basic_models, data=data)

results = evaluate_models2(models=basic_models, data=data, threshold=0.4)
results.round(2)

# %%
plot_confusion_matrices(basic_models, data, threshold=0.4)


# %% [markdown]
# **Algunas conclusiones:**
#
# - 
#
# -

# %% [markdown]
# #### Con ajuste de hiper-parámetros
# ---

# %%
def evaluate_models(models, data, threshold=0.5):
    rows = []

    for name, model in models.items():
        y_score, y_pred = get_scores(model, data, threshold=threshold)

        acc  = accuracy_score(data.y_test, y_pred)
        prec = precision_score(data.y_test, y_pred, zero_division=0)
        rec  = recall_score(data.y_test, y_pred, zero_division=0)
        f1   = f1_score(data.y_test, y_pred, zero_division=0)

        # AUC solo cuando hay scores continuos
        if y_score is not None:
            try:
                auc_value = roc_auc_score(data.y_test, y_score)
            except ValueError:
                auc_value = np.nan
        else:
            auc_value = np.nan

        rows.append({
            "Modelo": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc_value
        })

    return pd.DataFrame(rows).sort_values(by="AUC", ascending=False)


# %% [markdown]
# ##### 1. CNN

# %%
def build_cnn_model(hp):
    model = Sequential()

    # HP
    f = hp.Choice("filters", [64, 128])
    d = hp.Choice("dropout", [0.3, 0.4])
    lr = hp.Choice("lr", [1e-4, 1e-3])

    model.add(Input(shape=data.X_train.shape[1:]))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=f, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(d))
    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),  # type: ignore
        metrics=["accuracy", "AUC"],
    )

    return model



# %%
cnn_tuner = kt.RandomSearch(
    build_cnn_model,
    objective="val_accuracy",
    max_trials=10,
    directory="cnn_tuning",
    project_name="cnn_binary",
)

cnn_tuner.search(
    data.X_train, data.y_train, validation_data=(data.X_valid, data.y_valid), epochs=10
)

# %%
best_cnn = cnn_tuner.get_best_models(num_models=1)[0]
best_cnn.summary()

# %%
best_hp = cnn_tuner.get_best_hyperparameters(1)[0]
best_hp.values

# %%
evaluate_models({"best_cnn": best_cnn}, data).round(3)


# %% [markdown]
# ---

# %% [markdown]
# ##### 2. MLP

# %%
def build_mlp_model(hp):
    model = Sequential()

    # HP
    u1 = hp.Choice("units1", [128, 256])
    u2 = hp.Choice("units2", [128, 256])
    d = hp.Choice("dropout", [0.3, 0.4])
    lr = hp.Choice("lr", [1e-4, 1e-3])

    model.add(Input(shape=data.X_train.shape[1:]))
    model.add(Flatten())

    model.add(Dense(units=u1, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(units=u2, activation="relu"))
    model.add(Dropout(d))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),  # type: ignore
        metrics=["accuracy", "AUC"],
    )

    return model


# %%
mlp_tuner = kt.RandomSearch(
    build_mlp_model,
    objective="val_accuracy",
    max_trials=10,
    directory="mlp_tuning",
    project_name="cnn_binary",
)

mlp_tuner.search(
    data.X_train, data.y_train, validation_data=(data.X_valid, data.y_valid), epochs=15
)

# %%
best_mlp = mlp_tuner.get_best_models(num_models=1)[0]
best_mlp.summary()

# %%
best_hp = mlp_tuner.get_best_hyperparameters(1)[0]
best_hp.values

# %%
evaluate_models({"best_mlp": best_mlp}, data).round(2)

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Support Vector Classifier

# %%
import warnings
import os

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", module=r".*multiprocessing.queues")

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pkg_resources"

# %%
svc_pipe = Pipeline([
    ("pca", PCA(whiten=True, random_state=35)),
    ("svc", SVC(kernel="rbf", class_weight="balanced"))
])

param_grid = {
    "pca__n_components": [50, 100, 150, 200],
    "svc__C": [5],  # previas iteraciones nos indicaron este valor como el único útil
    "svc__gamma": [0.0011, 0.0012, 0.00013],
}

svc_grid = GridSearchCV(
    svc_pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    n_jobs=-1,
    cv=5,
    verbose=2
)

svc_grid.fit(data.X_train_flat, data.y_train)

# %%
print(f"Best Params: {svc_grid.best_params_}")
print(f"Best Cross-Val AUC: {svc_grid.best_score_:.4f}")

# %%
y_pred = svc_grid.predict(data.X_test_flat)
y_proba = svc_grid.decision_function(data.X_test_flat)

print(classification_report(data.y_test, y_pred))

print(f"\nTest Set ROC AUC: {roc_auc_score(data.y_test, y_proba):.4f}")

# %%
best_svc = svc_grid.best_estimator_
evaluate_models({"best_svc": best_svc}, data).round(2)

# %% [markdown]
# ---

# %% [markdown]
# ##### 4. Logistic Regression

# %%
warnings.filterwarnings('ignore')

# %%
logreg_pipe = Pipeline(
    [
        ("pca", PCA(random_state=35)),
        ("logreg", LogisticRegression(solver="liblinear", max_iter=1000)),
    ]
)

param_grid = {
    "pca__n_components": [0.95, 0.9, 0.85],
    "logreg__C": [0.005, 0.01, 0.015],
    "logreg__penalty": ["l1"],
}

logreg_grid = GridSearchCV(
    logreg_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=2
)

logreg_grid.fit(data.X_train_flat, data.y_train)

# %%
print(f"Best Params: {logreg_grid.best_params_}\n")

y_pred = logreg_grid.predict(data.X_test_flat)
print(classification_report(data.y_test, y_pred))

# %%
best_logreg = logreg_grid.best_estimator_
evaluate_models({"best_logreg": best_logreg}, data).round(2)

# %% [markdown]
# #### General evaluación

# %%
best_models = {
    "best_cnn": best_cnn,
    "best_mlp": best_mlp,
    "best_svc": best_svc,
    "best_logreg": best_logreg,
}

# %%
evaluate_models(best_models, data).round(3)

# %%
visualize_metrics(models=best_models, data=data)

# %% [markdown]
# #

# %% [markdown]
# Las conclusiones de los modelos utilizados fueron: 
#
# - **CNN** (Red Neuronal Convolucional): 
#     - Es la favorita. El estado del arte para imágenes.
#     - Representa un sistema visual, capaz de detectar bordes, texturas y formas.
#     - Aprende patrones complejos a costa de una gran cantidad de imágenes (lo cual cumplimos).
#     - Tuvo la mejor capacidad para distinguir las clases de las imagenes basandonos en las graficas de las metricas.
#     - la cantidad de parametros fueron alrededor de 0.5m
#     - y su tiempo fue 10 veces mas a comparacion de mlp siendo la mas tardada seguida de svc
#      
# - **MLP** (Perceptrón MultiCapa): 
#     - Generaliza más que una CNN, tiene ceguera ante la espacialidad.
#     - Util cuando no hay linealidad.
#     - No es realmente utilizada para imágenes.
#     - Pareciera que aprendio bien los datos pero sigue siendo peor que la cnn porque no entiende d eespacialidad
#
# - **SVC** (Clasificador de Soporte Vectorial): 
#     - No entiende espacialidad ni la estructura de las imágenes.
#     - Bueno para decisiones de frontera.
#     - No muy amigable con conjuntos de datos grandes.
#     - Fue de las mas tardado entre los 4 modelos 
#
# - **LR** (Regresión Logística): 
#     - Útil como base, para reconocer la necesidad de linealidad o estructuras complejas.
#     - Bueno para establecer fronteras.
#     - Fue muy mediocre al identificar las clases y el peor en comparacion a todos los modelos
#     - Fue la que menos tardo haciendo inferencia, esto puede ayudar como un primer estimador.

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Prueba con nuevos datos

# %%
# Leemos la metadata (etiquetas e ids)
metadata2_df = pd.read_csv(processed_folder / "sample_64x64_10570_v2.csv").astype(str)
display(metadata2_df.head())

# %%
# Cargamos las imágenes procesadas
processed_images_2 = np.load(processed_folder / "sample_64x64_10570_v2.npz")
files2 = processed_images_2.files
print(files2[:10])
print(len(files2))

# %%
# Ejemplo de uso de una imágen
img_name = files2[0]
idx = metadata2_df.index[metadata2_df["image_id"] == img_name][0]
img_class = metadata2_df.at[idx, "class"]

plt.imshow(processed_images_2[img_name], cmap="gray")
plt.title(f"Image {img_name}. Class {img_class}")
plt.axis("off")
plt.show()

# %%
# Obtenemos un arreglo con todas las imágenes
images_ids_2 = metadata2_df["image_id"].to_list()

images_array_2 = np.empty((len(metadata2_df), 64, 64), dtype=np.float32)

for i, img_name in enumerate(images_ids_2):
    image = processed_images_2[img_name]

    # Asegurarse que las imágenes estén normalizadas
    if image.max() > 1 or image.min() < 0:
        raise ValueError(f"Image {img_name} not normalized.")

    images_array_2[i] = image

# %%
# Revisando la cantidad de etiquetas respecto a la cantidad de imágenes, y congruencia de tamaños
print(images_array_2.shape, metadata2_df["class"].shape)

# %% [markdown]
# ---

# %% [markdown]
# ##### 2. Separación de datos en entrenamiento y prueba

# %%
# Se genera el objeto para los datos
data2 = Data(
    X=images_array_2.astype("float32"),
    y=metadata2_df["class"].astype("int8"),
)

# %%
# Checar
display(data2.X_test_flat.shape)

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Prueba con el mejor modelo

# %%
mejor_modelo = best_cnn

print(evaluate_models({"mejor_modelo": mejor_modelo}, data2).round(2))

# %%
visualize_metrics(models={"mejor_modelo": mejor_modelo}, data=data2)

# %% [markdown]
# ##### Conclusiones

# %% [markdown]
# La comparación entre modelos mostró que la CNN fue claramente la mejor opción para clasificar las imágenes. Sus resultados finales (Accuracy 0.78, F1 0.73, ROC-AUC 0.85) indican que aprendió patrones visuales relevantes y generalizó razonablemente bien en datos nuevos, aun con un costo computacional mayor y un tiempo de inferencia más largo.
#
# El MLP y el SVC mostraron limitaciones claras: ambos pierden información espacial al trabajar con imágenes aplanadas, lo que redujo su capacidad predictiva y, en el caso del SVC, elevó el tiempo de cómputo. La Regresión Logística sirvió únicamente como línea base; fue la más rápida pero también la más débil en todas las métricas.
#
# En conjunto, los resultados confirman que la arquitectura convolucional es la más adecuada para este tipo de datos y ofrece el mejor balance entre desempeño y estabilidad, incluso sin posibilidad de realizar más pruebas.

# %% [markdown]
# **Siguientes pasos:** Una mejora natural sería explorar técnicas de aumento de datos, ajustar la arquitectura con otras variantes que puedan ser más eficientes y optimizar el preprocesamiento para reducir el tiempo de inferencia sin sacrificar desempeño. Estos ajustes permitirían refinar la generalización del modelo en futuros experimentos.

# %% [markdown]
# ---
