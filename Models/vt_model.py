import tensorflow as tf
import keras
from keras import layers
import math
import numpy as np

class vt_model(layers.Layer):

    def __init__(self, image_size=28, patch_size=7, projection_dim=64, vanilla=False, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.vanilla = vanilla
        self.projection_dim = projection_dim
        self.flatten = layers.Reshape((-1,))
        self.proj = None 

    def build(self, input_shape):
        # input_shape
        num_channels = input_shape[-1] * (5 if not self.vanilla else 1)
        patch_vector_size = (self.patch_size ** 2) * num_channels
        
        self.proj = layers.Dense(units=self.projection_dim)
        # Build the Dense layer
        self.proj.build((None, patch_vector_size))
        super().build(input_shape)

    def crop_shift_pad(self, images, mode):
        if mode == "left-up":
            offset_h, offset_w = 0, 0
            target_h = self.image_size - self.half_patch
            target_w = self.image_size - self.half_patch
        elif mode == "left-down":
            offset_h, offset_w = self.half_patch, 0
            target_h = self.image_size - self.half_patch
            target_w = self.image_size - self.half_patch
        elif mode == "right-up":
            offset_h, offset_w = 0, self.half_patch
            target_h = self.image_size - self.half_patch
            target_w = self.image_size - self.half_patch
        else: 
            offset_h, offset_w = self.half_patch, self.half_patch
            target_h = self.image_size - self.half_patch
            target_w = self.image_size - self.half_patch

        cropped = tf.image.crop_to_bounding_box(images, offset_h, offset_w, target_h, target_w)
        padded = tf.image.pad_to_bounding_box(
            cropped, offset_height=offset_h, offset_width=offset_w,
            target_height=self.image_size, target_width=self.image_size
        )
        return padded

    def call(self, images):

        if self.vanilla:
            concat = images
        else:
            s1 = self.crop_shift_pad(images, "left-up")
            s2 = self.crop_shift_pad(images, "left-down")
            s3 = self.crop_shift_pad(images, "right-up")
            s4 = self.crop_shift_pad(images, "right-down")
            concat = tf.concat([images, s1, s2, s3, s4], axis=-1)

        patches = tf.image.extract_patches(
            images=concat,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        batch = tf.shape(patches)[0]
        n_patches = (self.image_size // self.patch_size) ** 2
        patches = tf.reshape(patches, (batch, n_patches, -1))
        tokens = self.proj(patches)
        return tokens

    def compute_output_shape(self, input_shape):
        # Return the output
        num_patches = (self.image_size // self.patch_size) ** 2
        return (input_shape[0], num_patches, self.projection_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'projection_dim': self.projection_dim,
            'vanilla': self.vanilla
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_patches, self.projection_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, patches):
        return patches + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config


class QueryScaler(layers.Layer):
    def __init__(self, init_value=None, **kwargs):
        super().__init__(**kwargs)
        if init_value is None:
            init_value = 1.0
        self.init = float(init_value)

    def build(self, input_shape):
        self.log_tau = self.add_weight(
            name='log_tau',
            shape=(1,),
            initializer=keras.initializers.Constant(math.log(self.init)),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        tau = tf.exp(self.log_tau)
        return x / tau

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_value': self.init
        })
        return config


def mlp(x, hidden_units, dropout_rate=0.1):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(
    input_shape=(28, 28, 1),
    num_classes=19,
    patch_size=7,
    projection_dim=64,
    transformer_layers=4,
    num_heads=4,
    mlp_head_units=(128, 64),
    vanilla=False
):
    #  number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    inputs = layers.Input(shape=input_shape)

    x = layers.RandomRotation(0.02)(inputs)
    x = layers.RandomZoom(0.1)(x)

    x = layers.Rescaling(1.0 / 255.0)(x)

    tokens = vt_model(image_size=input_shape[0], patch_size=patch_size, projection_dim=projection_dim, vanilla=vanilla)(x)

    encoded = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(tokens)

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)

        scaled_q = QueryScaler()(x1)
        attention_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(scaled_q, x1)

        x2 = layers.Add()([attention_out, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        encoded = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits, name='vit_mnist')
    return model