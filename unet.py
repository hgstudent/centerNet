#from https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/
def unet(in_, filters=32):
  c1 = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding='same')(in_)
  c1 = tf.keras.layers.Dropout(0.1)(c1)
  c1 = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding='same')(c1)
  p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
  
  c2 = tf.keras.layers.Conv2D(filters*2, (3, 3),  activation="relu", padding='same')(p1)
  c2 = tf.keras.layers.Dropout(0.1)(c2)
  c2 = tf.keras.layers.Conv2D(filters*2, (3, 3), activation="relu", padding='same')(c2)
  p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
  
  c3 = tf.keras.layers.Conv2D(filters*4, (3, 3), activation="relu", padding='same')(p2)
  c3 = tf.keras.layers.Dropout(0.2)(c3)
  c3 = tf.keras.layers.Conv2D(filters*4, (3, 3), activation="relu", padding='same')(c3)
  p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
  
  c4 = tf.keras.layers.Conv2D(filters*8, (3, 3), activation="relu", padding='same')(p3)
  c4 = tf.keras.layers.Dropout(0.2)(c4)
  c4 = tf.keras.layers.Conv2D(filters*8, (3, 3), activation="relu", padding='same')(c4)
  p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
  
  c5 = tf.keras.layers.Conv2D(filters*16, (3, 3), activation="relu", padding='same')(p4)
  c5 = tf.keras.layers.Dropout(0.3)(c5)
  c5 = tf.keras.layers.Conv2D(filters*16, (3, 3), activation="relu", padding='same')(c5)
  
  u6 = tf.keras.layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, c4])
  c6 = tf.keras.layers.Conv2D(filters*8, (3, 3), activation="relu", padding='same')(u6)
  c6 = tf.keras.layers.Dropout(0.2)(c6)
  c6 = tf.keras.layers.Conv2D(filters*8, (3, 3), activation="relu", padding='same')(c6)
  
  u7 = tf.keras.layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  c7 = tf.keras.layers.Conv2D(filters*4, (3, 3), activation="relu", padding='same')(u7)
  c7 = tf.keras.layers.Dropout(0.2)(c7)
  c7 = tf.keras.layers.Conv2D(filters*4, (3, 3), activation="relu", padding='same')(c7)
  
  u8 = tf.keras.layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  c8 = tf.keras.layers.Conv2D(filters*2, (3, 3), activation="relu", padding='same')(u8)
  c8 = tf.keras.layers.Dropout(0.1)(c8)
  c8 = tf.keras.layers.Conv2D(filters*2, (3, 3), activation="relu", padding='same')(c8)
  
  u9 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
  c9 = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding='same')(u9)
  c9 = tf.keras.layers.Dropout(0.1)(c9)
  c9 = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding='same')(c9)
  return c9
