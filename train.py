import tensorflow as tf
import tensorflow_datasets as tfds

from utils import *
from model import *
from loss import *


voc_train = tfds.load(name="voc", split="train")
optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
shapes    = ([None, None, 3], [None, 4])
p_values  = (tf.constant(0, dtype=tf.float32), tf.constant(-1, dtype=tf.int32))
epochs    = 100
m         = centerNet()

for epoch in range(epochs):
  loss_cmp  = 0
  i         = 0
  
  for voc_example in voc_train.map(preprocess_img).padded_batch(5,  padded_shapes=shapes, padding_values=p_values): 
    
    image, label  = voc_example[0], voc_example[1]
    tl, br        = extract_masks(label[:,:,0:2]), extract_masks(label[:,:,2:4])
    
    with tf.GradientTape() as tape:  
      logits  = m(image)
      loss_   = loss_fn(logits, [tl, br])
      
    loss_cmp += tf.reduce_sum(loss_)
    grads     = tape.gradient(loss_, m.trainable_weights)
    optimizer.apply_gradients(zip(grads, m.trainable_weights))
    i = i + 1
  
  print("Epoch: "       + str(epoch))
  print("Total loss: "  + str(loss_cmp/i))
  print()
