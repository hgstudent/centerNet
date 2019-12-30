from pooling import *
from unet import *
import tensorflow as tf

def model(c = 1, input_size=(512,512,3)):
  inputs = tf.keras.Input(shape=input_size)
  x = conv_module(inputs, size=(7,7), strides_=(2,2))
  x = conv_module(x, size=(3,3), strides_=(2,2))

  x = unet(x)
  feature_map = unet(x)
  #Top left
  tl = tl_pooling(feature_map)
  tl = tf.keras.layers.Conv2D(32, (3,3), padding="same")(tl)
  tl = tf.keras.layers.BatchNormalization()(tl)

  x = tf.keras.layers.Conv2D(32, (1,1), padding="same")(feature_map)

  x = tl + x
  x = tf.keras.layers.ReLU()(x)
  x = conv_module(x)

  #Top left heatmap
  tl_hm = tf.keras.layers.Conv2D(32, (3,3), activation="sigmoid", padding="same")(x)
  tl_hm = tf.keras.layers.Conv2D(c, (1,1), padding="same", name="tl_hm")(tl_hm)

  #Top left embeddings
  tl_em = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
  tl_em = tf.keras.layers.Conv2D(1, (1,1), padding="same")(tl_em)
  tl_em = tf.squeeze(tl_em, [-1], name="tl_em")

  #Bottom right
  br = br_pooling(feature_map)
  br = tf.keras.layers.Conv2D(32, (3,3), padding="same")(br)
  br = tf.keras.layers.BatchNormalization()(br)

  x = tf.keras.layers.Conv2D(32, (1,1), padding="same")(feature_map)

  x = br + x
  x = tf.keras.layers.ReLU()(x)
  x = conv_module(x)

  #Bottom right heatmap
  br_hm = tf.keras.layers.Conv2D(32, (3,3), activation="sigmoid", padding="same")(x)
  br_hm = tf.keras.layers.Conv2D(c, (1,1), padding="same", name="br_hm")(br_hm)

  #Bottom right embeddings
  br_em = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
  br_em = tf.keras.layers.Conv2D(1, (1,1), padding="same")(br_em)
  br_em = tf.squeeze(br_em, [-1], name="br_em")

  #Create model
  model = tf.keras.Model(inputs=inputs, outputs=[tl_hm, tl_em, br_hm, br_em])
  return model
