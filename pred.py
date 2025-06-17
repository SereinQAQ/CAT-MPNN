import os
import random

# Temporary suppress tf logs
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import warnings
import argparse
import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import RDLogger
import camodels
from dataset import graphs_from_smiles, process_me_attributes
from tool import (
    preparedataset,
    manual_iter,
    RSquared,
    ModelCheckpointWithCleanup,
    getdateset,
)
from tool import RSquaredLoss


# tf.compat.v1.disable_eager_execution()

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)

ap = argparse.ArgumentParser()
ap.add_argument(
    "--dataset",
    type=str,
    default=".\\mejpg",
    help="path to input dataset of house images",
)
ap.add_argument(
    "--model", type=str, default="densenet", help="name to input dataset of model"
)
ap.add_argument("--batchsize", type=int, default=32, help="batchsize")
ap.add_argument("--trainsize", type=float, default=0.8, help="trainsize")
args = vars(ap.parse_args())

print("[INFO] loading attributes...")

batchsize = args["batchsize"]
train_size = args["trainsize"]

inputPath = os.path.sep.join([args["dataset"], "DESss.xlsx"])
df = datasets.load_me_attributes(inputPath)
df_hba = df["D1"]
df_hbd = df["D2"]
print(df["HBA:HBD"])
df = df[["HBA:HBD", "T", "Density"]]
AttrX = process_me_attributes(df, continuous=["HBA:HBD", "T"], label=False)
df = df["Density"]
df = np.log10(df)
# df = process_me_attributes(df, continuous=["Density"], label=False)
# Dataset_split

# train_index = permuted_indices[: int(datacount * train_size)]
# valid_index = permuted_indices[int(datacount * train_size) : int(datacount * 0.99)]
# test_index = permuted_indices[int(datacount * 0.99) :]


Attrx_valid = AttrX
hbax_valid = graphs_from_smiles(df_hba)
hbdx_valid = graphs_from_smiles(df_hbd)
y_valid = df


valtensor = (
    tf.data.Dataset.from_tensor_slices((Attrx_valid, hbax_valid, hbdx_valid, y_valid))
    .batch(batchsize)
    .map(preparedataset, -1)
)


validdatasets = manual_iter(valtensor)


tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-4
)

model = camodels.Finalmodel(batchsize=batchsize, num_heads=8, d_model=16)

# pretrained_model_path = "best_mpnn_model.h5"
# model = keras.models.load_model(pretrained_model_path)

# for layer in model.layers:
#     try:
#         # 获取层的权重
#         weights = layer.get_weights()
#         print(
#             f"Layer '{layer.name}' loaded successfully with weights shape: {[w.shape for w in weights]}"
#         )
#     except Exception as e:
#         print(f"Error loading layer '{layer.name}': {e}")

keras.utils.plot_model(model, show_dtype=True, show_shapes=True)
model.compile(
    # loss=RSquaredLoss(),
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    metrics=[
        keras.metrics.MeanAbsoluteError(name="mae"),
        keras.metrics.RootMeanSquaredError(name="rmse"),
        RSquared(name="r2"),
    ],
)
checkpoint_callback = ModelCheckpointWithCleanup(
    filepath="model_epoch_{epoch:02d}.h5",  # 保存路径
    monitor="val_loss",  # 监控的指标
    save_best_only=True,  # 仅保存最佳模型
    mode="min",  # 监控指标的变化方向
    max_to_keep=3,  # 仅保留最近的3个模型
)

for layer in model.layers:
    if layer.name.startswith("message_passing"):
        layer.trainable = False
    if layer.name.startswith("transformer_encoder_readout"):
        layer.trainable = False

reduce_lr.set_model(model)
tensorboard_callback.set_model(model)
checkpoint_callback.set_model(model)

epochs = 30
train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_loss_metric = tf.keras.metrics.Mean(name="val_loss")


history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_mae": [],
    "val_mae": [],
    "train_rmse": [],
    "val_rmse": [],
    "train_r2": [],
    "val_r2": [],
}


# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")

#     train_dataset = tqdm(traindatasets, desc=f"Epoch {epoch + 1}/{epochs} - Training")

#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         with tf.GradientTape() as tape:
#             logits = model(x_batch_train, training=True)
#             loss_value = model.compiled_loss(y_batch_train, logits)

#         grads = tape.gradient(loss_value, model.trainable_weights)
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # 更新训练指标
#         train_loss_metric.update_state(loss_value)
#         model.compiled_metrics.update_state(y_batch_train, logits)

#         train_dataset.set_postfix({"Training loss": train_loss_metric.result().numpy()})
#     if epoch > 20:
#         print("**************")
#         print(y_batch_train)
#         print("**************")
#         print(logits)
#     train_loss = train_loss_metric.result()
#     train_metrics = {metric.name: metric.result().numpy() for metric in model.metrics}
#     train_loss_metric.reset_states()
#     for metric in model.metrics:
#         metric.reset_states()

#     print(f"Training loss over epoch: {train_loss:.4f}")
#     for name, result in train_metrics.items():
#         print(f"Training {name} over epoch: {result:.4f}")

#     val_dataset = tqdm(validdatasets, desc=f"Epoch {epoch + 1}/{epochs} - Validation")

#     for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
#         val_logits = model(x_batch_val, training=False)
#         val_loss_value = model.compiled_loss(y_batch_val, val_logits)

#         val_loss_metric.update_state(val_loss_value)
#         model.compiled_metrics.update_state(y_batch_val, val_logits)

#         val_dataset.set_postfix({"Validation loss": val_loss_metric.result().numpy()})

#     val_loss = val_loss_metric.result()
#     val_metrics = {metric.name: metric.result().numpy() for metric in model.metrics}
#     val_loss_metric.reset_states()
#     for metric in model.metrics:
#         metric.reset_states()

#     print(f"Validation loss: {val_loss:.4f}")
#     for name, result in val_metrics.items():
#         print(f"Validation {name}: {result:.4f}")

#     # 调用回调函数
#     logs = {
#         "val_loss": val_loss.numpy(),
#         **{f"val_{name}": result for name, result in val_metrics.items()},
#         "lr": model.optimizer.lr.numpy(),
#     }
#     reduce_lr.on_epoch_end(epoch, logs)
#     tensorboard_callback.on_epoch_end(epoch, logs)
#     checkpoint_callback.on_epoch_end(epoch, logs)
#     history["epoch"].append(epoch + 1)
#     history["train_loss"].append(train_loss.numpy())
#     history["val_loss"].append(val_loss.numpy())
#     history["train_mae"].append(train_metrics["mae"])
#     history["val_mae"].append(val_metrics["mae"])
#     history["train_rmse"].append(train_metrics["rmse"])
#     history["val_rmse"].append(val_metrics["rmse"])
#     history["train_r2"].append(train_metrics["r2"])
#     history["val_r2"].append(val_metrics["r2"])
#     model.save("./savemodel/model_epoch_{}".format(epoch), save_format="tf")
#     if epoch + 1 == 8:
#         for layer in model.layers:
#             layer.trainable = True
#         model.compile(
#             # loss=RSquaredLoss(),
#             loss="mse",
#             optimizer=keras.optimizers.Adam(learning_rate=1.5e-4),
#             metrics=[
#                 keras.metrics.MeanAbsoluteError(name="mae"),
#                 keras.metrics.RootMeanSquaredError(name="rmse"),
#                 RSquared(name="r2"),
#             ],
#         )

# history_df = pd.DataFrame(history)
# history_df.to_csv("training_history.csv", index=False)
# print(f"Training history saved to training_history.csv")


import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load the trained model
model = camodels.Finalmodel(batchsize=batchsize, num_heads=8, d_model=16)

model.load_weights("beat.h5", by_name=True)

val_dataset = tqdm(validdatasets)
y_test = []
y_pred = []
for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
    val_logits = model(x_batch_val)
    y_test.append(y_batch_val.numpy())
    y_pred.append(val_logits.numpy())


y_test = np.concatenate(y_test, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
print(y_pred)
# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="#EEC186", alpha=0.8, label="Predicted")
plt.scatter(y_pred, y_test, color="#B8E5FA", alpha=0.8, label="Experimental")
plt.plot(
    [min(y_test), max(y_test)], [min(y_test), max(y_test)], color="black", linewidth=2
)
plt.xlabel("Experimental value")
plt.ylabel("Predicted value")
plt.title("Experimental vs. Predicted values")
plt.text(
    1.05,
    2.8,
    f"$R^2$={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.legend()
plt.show()
