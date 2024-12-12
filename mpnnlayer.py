import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import warnings
import tensorflow as tf
from tensorflow import keras
from keras import layers
from rdkit import RDLogger
import mpnngraphs
# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)


class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim),
            initializer="zeros",
            name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        # atom_features = layers.core.Activation("relu")(atom_features)
        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

    def get_config(self):
        config = super(MessagePassing, self).get_config()
        config.update({"units": self.units, "steps": self.steps})
        return config


class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.batch_size = batch_size
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))

        return self.average_pooling(proj_output)

    def get_config(self):
        config = super(TransformerEncoderReadout, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "batch_size": self.batch_size,
            }
        )
        return config


def MPNNModel(
    n_name,
    atom_dim,
    bond_dim,
    batch_size,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype="float32", name=n_name+"atom_features")
    bond_features = layers.Input(
        (bond_dim), dtype="float32", name=n_name+"bond_features"
    )
    pair_indices = layers.Input((2), dtype="int32", name=n_name+"pair_indices")
    molecule_indicator = layers.Input(
        (), dtype="int32", name=n_name+"molecule_indicator"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64)(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model


def load_pretrained_weights(model, model1, weight_path):
    """
    Load weights from a pre-trained model into a new model.

    :param model: The model into which the weights are to be loaded.
    :param weight_path: Path to the pre-trained weights file.
    :param layer_prefix: Prefix of the layers to match when loading weights.
    """
    pretrained_model = mpnngraphs.MPNNModel(
        atom_dim=29,
        bond_dim=7,
    )
    pretrained_model.load_weights(weight_path, by_name=True)

    for layer in model.layers:
        if layer.name.startswith('message'):
            pretrained_layer = pretrained_model.get_layer("message_passing_3")
            layer.set_weights(pretrained_layer.get_weights())

        if layer.name.startswith("transformer"):
            pretrained_layer = pretrained_model.get_layer("transformer_encoder_readout_3")
            layer.set_weights(pretrained_layer.get_weights())

    for layer in model1.layers:
        if layer.name.startswith("message"):
            pretrained_layer = pretrained_model.get_layer("message_passing_3")
            layer.set_weights(pretrained_layer.get_weights())

        if layer.name.startswith("transformer"):
            pretrained_layer = pretrained_model.get_layer("transformer_encoder_readout_3")
            layer.set_weights(pretrained_layer.get_weights())
