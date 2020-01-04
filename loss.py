import tensorflow as tf
import numpy as np

def push_pull(pred_tl, pred_br, mask_tl, mask_br):
  s1 = tf.gather_nd(pred_tl, mask_tl)
  s2 = tf.gather_nd(pred_br, mask_br)
  N = len(s1)
  mean = (s1+s2)/2

  pull = tf.pow(s1-mean, 2) + tf.pow(s2-mean,2)
  pull = tf.reduce_sum(pull)
  
  tmp = tf.transpose(tf.broadcast_to(mean,(N,N)))  
  push = tf.math.abs(mean - tmp)
  push = tf.math.maximum(0, 1 - push) - tf.eye(N)
  push = tf.reduce_sum(push)

  pull = pull/N
  push = push/(N*(N-1+1e-4))
  
  return push, pull

def loss_fn(y_pred, y_true, bce):
  tl_hm_pred = y_pred[0]
  tl_em_pred = y_pred[1]
  br_hm_pred = y_pred[2]
  br_em_pred = y_pred[3]

  tl_mask = y_true[0]
  br_mask = y_true[1]
  N = len(tl_mask)
  
  if( N <= 0):
    return 0

  tl_2d_mask = np.zeros((tl_hm_pred.shape))
  b, h, w = tl_mask[:,0], tl_mask[:,1], tl_mask[:,2]
  tl_2d_mask[b, h, w, 0] = 1
  
  br_2d_mask = np.zeros((br_hm_pred.shape))
  b, h, w = br_mask[:,0], br_mask[:,1], br_mask[:,2]
  br_2d_mask[b, h, w, 0] = 1

  push_loss, pull_loss = push_pull(tl_em_pred, br_em_pred, tl_mask, br_mask)
  
  corner_loss_tl = tf.keras.losses.binary_crossentropy(tl_2d_mask, tl_hm_pred, label_smoothing=0.1)
  corner_loss_tl = tf.reduce_sum(corner_loss_tl)/N

  corner_loss_br = tf.keras.losses.binary_crossentropy(br_2d_mask, br_hm_pred, label_smoothing=0.1)
  corner_loss_br = tf.reduce_sum(corner_loss_br)/N
  
  return 0.1*push_loss, 0.1*pull_loss, corner_loss_tl, corner_loss_br
