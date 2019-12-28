import tensorflow as tf
import numpy as np

def push_pull(pred_tl, pred_br, mask_tl, mask_br):
  s1 = tf.gather_nd(pred_tl, mask_tl)
  s2 = tf.gather_nd(pred_br, mask_br)
  N = len(s1)
  push, pull = 0, 0
  m = []
  
  for a,b in zip(s1,s2):
    mean = (a+b)/2
    pull += tf.pow(a-mean, 2) + tf.pow(b-mean,2)
    m.append(mean)
  
  m = tf.convert_to_tensor(m)
  for i in range(len(m)):
    mask = np.ones(len(m))
    mask[i] = 0
    mask = tf.convert_to_tensor(mask)

    p = 1 - tf.math.abs(m - tf.gather_nd(m, [i]))
    p = p*mask
    push += tf.reduce_sum(p)

  pull = pull/N
  push = push/(N*(N-1))  

  return push, pull

def extract_masks(gt, dtype_="int32"):
  gt = tf.cast(gt, dtype=dtype_)
  cond = tf.where(tf.math.not_equal(gt, 0))
  c = cond[::2, 0]
  c = tf.reshape(c, (-1, 1))
  c = tf.cast(c, dtype=dtype_)
  c = tf.concat([c,tf.reshape(tf.gather_nd(gt, cond), (2,-1))], axis=-1)
  return c
