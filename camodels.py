import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, GlobalAveragePooling2D, Add, LeakyReLU
from keras.layers import MultiHeadAttention, Layer, concatenate, LayerNormalization
from keras.applications.densenet import DenseNet169
import mpnnlayer


def create_mlp(dim, regress=False):
    model = Sequential()

    # 添加第一个全连接层和标准化层
    model.add(Dense(8, input_dim=dim, use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dense(32, use_bias=False))
    # # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.01))

    # model.add(Dense(64, use_bias=False))
    # # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.01))

    # model.add(Dense(128, use_bias=False))
    # # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.01))

    # model.add(Dense(16, use_bias=False))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.01))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model


def create_cnn(Iminput, filters=(16, 32, 64), regress=False):
    inputShape = (Iminput[1], Iminput[0], Iminput[2])
    chanDim = -1

    inputs = Input(shape=inputShape)

    for i, f in enumerate(filters):
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


class CrossAttentionLayer(Layer):
    def __init__(self, num_heads, d_model, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )

    def call(self, inputs):
        # Assuming inputs is a list: [query_input, key_value_input]
        query_input, key_value_input = inputs

        # Add a new axis to match the requirements for MultiHeadAttention layer (batch_size, seq_len, embedding_dim)
        query_input = tf.expand_dims(query_input, axis=1)
        key_value_input = tf.expand_dims(key_value_input, axis=1)

        # Apply cross attention
        attention_output = self.multi_head_attention(
            query_input, key_value_input, key_value_input
        )

        # Remove the sequence length axis
        attention_output = tf.squeeze(attention_output, axis=1)

        return attention_output

    def get_config(self):
        config = super(CrossAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "d_model": self.d_model})
        return config


def Finalmodel(batchsize,num_heads=8, d_model=16):

    mpnnhba = mpnnlayer.MPNNModel(
        batch_size=batchsize,
        n_name="hba",
        atom_dim=29,
        bond_dim=7,
    )

    mpnnhbd = mpnnlayer.MPNNModel(
        batch_size=batchsize,
        n_name="hbd",
        atom_dim=29,
        bond_dim=7
    )

    # Load pre-trained weights into the MPNN branches
    mpnnlayer.load_pretrained_weights(mpnnhba, mpnnhbd, "best_mpnn_model.h5")

    mlp = create_mlp(2, regress=False)
    combinedInput = concatenate([mpnnhba.output, mpnnhbd.output])
    # combinedInput = Add()([mpnnhba.output, mpnnhbd.output])

    # 注意力层
    # if len(combinedInput.shape) == 2:
    #     combinedInput = tf.expand_dims(combinedInput, axis=1)

    # if len(mlp.output.shape) == 2:
    #     mlpoutput = tf.expand_dims(mlp.output, axis=1)
    mlpoutput = mlp.output
    # attention_output = CrossAttentionLayer(num_heads, d_model)(
    #     [combinedInput,mlpoutput,]
    # )
    # attention_output = Add()([combinedInput, mlpoutput])
    catoutput = concatenate([combinedInput, mlpoutput])
    attention_output = CrossAttentionLayer(num_heads, d_model)(
        [
            combinedInput,
            catoutput,
        ]
    )
    # normalized_output = LayerNormalization()(residual_output)
    # x = Dense(8, activation="relu")(attention_output)
    # x = Dense(32, activation="relu")(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dense(32, activation="relu")(x)
    # x = Dense(16, activation="relu")(x)
    # x = Dense(1, activation="linear")(x)
    x = Dense(128)(attention_output)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = Dense(256)(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = Dense(64)(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(1, activation="linear")(x)
    finalmodel = Model(inputs=[mlp.input, mpnnhba.input, mpnnhbd.input], outputs=x)

    return finalmodel


# def create_densenet(regress=False):
#     densenetmode = DenseNet201(include_top=False, input_shape=(64, 64, 6))

#     x = densenetmode.output
#     x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
#     x = Dense(16)(x)
#     x = Activation("relu")(x)

#     model = Model(inputs=densenetmode.input, outputs=x)
#     print(model.summary())

#     return model


# class DenseNet:
#     def __init__(self, x):
#         # self.nb_blocks = nb_blocks
#         # self.filters = filters
#         # self.training = training
#         self.x_shape = x
#         self.model = self.Dense_net(x)

#     def DenseLayer(self, x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
#         x = BatchNormalization(axis=3)(x)
#         x = Activation("relu")(x)
#         x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding="same")(x)
#         x = BatchNormalization(axis=3)(x)
#         x = Activation("relu")(x)
#         x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding="same")(x)
#         if drop_rate:
#             x = Dropout(drop_rate)(x)

#         return x

#     def DenseBlock(self, x, nb_layers, growth_rate, drop_rate=0.2):
#         for i in range(nb_layers):
#             conv = self.DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
#             x = concatenate([x, conv], axis=3)

#         return x

#     def TransitionLayer(self, x, compression=0.5, alpha=0.0, is_max=1):
#         nb_filter = int(x.shape.as_list()[-1] * compression)
#         x = BatchNormalization(axis=3)(x)
#         x = Activation("relu")(x)
#         x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding="same")(x)
#         if is_max != 0:
#             x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
#         else:
#             x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

#         return x

#     def Dense_net(self, input_x):
#         growth_rate = 4
#         bnsize = 8
#         inputs = Input(shape=input_x)

#         x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding="same")(inputs)
#         x = BatchNormalization(axis=3)(x)
#         x = Activation("relu")(x)

#         x = self.DenseBlock(x, bnsize, growth_rate, drop_rate=0.2)

#         x = self.TransitionLayer(x)

#         x = self.DenseBlock(x, bnsize, growth_rate, drop_rate=0.2)

#         x = self.TransitionLayer(x)

#         x = self.DenseBlock(x, bnsize, growth_rate, drop_rate=0.2)

#         x = BatchNormalization(axis=3)(x)
#         x = GlobalAveragePooling2D()(x)

#         x = Dense(4, activation="relu")(x)

#         model = Model(inputs, x)

#         return model


class DenseNet:
    def __init__(self, x):
        # self.nb_blocks = nb_blocks
        # self.filters = filters
        # self.training = training
        self.x_shape = x
        self.model = self.Dense_net(x)

    def add_new_last_layer(self, base_model, Iminput, drop_rate=0.5):
        base_model.layers.pop(0)

        newInput = Input(batch_shape=(0, Iminput[1], Iminput[0], Iminput[2]))
        x = base_model(newInput)
        x = Dropout(drop_rate)(x)  # 添加dropout层
        x = GlobalAveragePooling2D()(x)  # 添加Pooling层
        x = Dense(4, activation="relu")(x)  # 添加softmax层
        model = Model(inputs=newInput, outputs=x)
        return model

    def Dense_net(self, Iminput):
        inputs = Input(shape=Iminput)
        x = Conv2D(3, 1, padding="same")(inputs)
        base_model = DenseNet169(
            include_top=False,
            weights="imagenet",
            input_shape=(Iminput[1], Iminput[0], 3),
        )
        # base_model.layers.pop(0)
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)  # 添加Pooling层
        x = Dense(4, activation="relu")(x)  # 添加softmax层
        model = Model(inputs=inputs, outputs=x)
        # model = self.add_new_last_layer(base_model, Iminput)
        #  for layer in model.layers[:595]:   # DenseNet169需要冻结前595层
        #     layer.trainable = False   # 冻结模型全连接层之前的参数
        return model
