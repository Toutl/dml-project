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
print(files[:10])

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
input_size = 2
output_size = 1

model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])

model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit(metadata_df["class"], processed_images, epochs=100, verbose="")

# %%
