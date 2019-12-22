from pooling import *
import tensorflow as tf

def model(c = 1):
  inputs = tf.keras.Input(shape=(3,3,1))
  #
  # Backbone to be added
  #
  
  
  
  #Top left
  tl = tl_pooling(inputs)
  tl = tf.keras.layers.Conv2D(32, (3,3), padding="same")(tl)
  tl = tf.keras.layers.BatchNormalization()(tl)

  x = tf.keras.layers.Conv2D(32, (1,1), padding="same")(inputs)

  x = tl + x
  x = tf.keras.layers.ReLU()(x)
  x = conv_module(x)

  tl_hm = tf.keras.layers.Conv2D(32, (3,3), padding="same")(x)
  tl_hm = tf.keras.layers.ReLU()(tl_hm)
  tl_hm = tf.keras.layers.Conv2D(c, (1,1), padding="same", name="tl_hm")(tl_hm)

  tl_em = tf.keras.layers.Conv2D(32, (3,3), padding="same")(x)
  tl_em = tf.keras.layers.ReLU()(tl_em)
  tl_em = tf.keras.layers.Conv2D(1, (1,1), padding="same", name="tl_em")(tl_em)


  #Bottom right
  br = br_pooling(inputs)
  br = tf.keras.layers.Conv2D(32, (3,3), padding="same")(br)
  br = tf.keras.layers.BatchNormalization()(br)

  x = tf.keras.layers.Conv2D(32, (1,1), padding="same")(inputs)

  x = br + x
  x = tf.keras.layers.ReLU()(x)
  x = conv_module(x)

  br_hm = tf.keras.layers.Conv2D(32, (3,3), padding="same")(x)
  br_hm = tf.keras.layers.ReLU()(br_hm)
  br_hm = tf.keras.layers.Conv2D(c, (1,1), padding="same", name="br_hm")(br_hm)

  br_em = tf.keras.layers.Conv2D(32, (3,3), padding="same")(x)
  br_em = tf.keras.layers.ReLU()(br_em)
  br_em = tf.keras.layers.Conv2D(1, (1,1), padding="same", name="br_em")(br_em)

  #Create model
  model = tf.keras.Model(inputs=inputs, outputs=[tl_hm, tl_em, br_hm, br_em])
  return model
