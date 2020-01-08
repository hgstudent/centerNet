import tensorflow as tf
import numpy as np

def push_pull(pred_tl, pred_br, mask_tl, mask_br):
  s1    = tf.gather_nd(pred_tl, mask_tl)
  s2    = tf.gather_nd(pred_br, mask_br)
  N     = len(s1)
  mean  = (s1+s2)/2

  pull  = tf.pow(s1-mean, 2) + tf.pow(s2-mean,2)
  pull  = tf.reduce_sum(pull)
  
  tmp   = tf.transpose(tf.broadcast_to(mean,(N,N)))  
  push  = tf.math.abs(mean - tmp)
  push  = tf.math.maximum(0, 1 - push) - tf.eye(N)
  push  = tf.reduce_sum(push)

  pull  = pull/N
  push  = push/(N*(N-1+1e-4))
  
  return push, pull

def focal_loss(hm_pred, hm_true):
    #Inspired by https://github.com/xuannianz/keras-CenterNet
    N         = tf.cast(tf.reduce_sum(hm_true), tf.float32)
    pos_mask  = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask  = tf.cast(tf.less(hm_true, 1), tf.float32)

    pos_loss  = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss  = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_mask

    pos_loss  = tf.reduce_sum(pos_loss)
    neg_loss  = tf.reduce_sum(neg_loss)

    _loss  = (neg_loss + pos_loss)/N
    return _loss

def loss_fn(y_pred, y_true):
  tl_hm_pred  = y_pred[0]
  tl_em_pred  = y_pred[1]
  br_hm_pred  = y_pred[2]
  br_em_pred  = y_pred[3]

  tl_mask     = y_true[0]
  br_mask     = y_true[1]
  N           = len(tl_mask)
  
  if( N <= 0):
    return 0

  push_loss = 0
  pull_loss = 0

  for batch_no in range(int(tl_mask[-1][0])+1):
    ind         = tf.where(tl_mask[:,0] == batch_no)
    ind         = tf.gather_nd(tl_mask, ind)

    ind_        = tf.where(br_mask[:,0] == batch_no)
    ind_        = tf.gather_nd(br_mask, ind_)

    tmp1, tmp2  = push_pull(tl_em_pred, br_em_pred, ind, ind_)
    push_loss   += tmp1
    pull_loss   += tmp2


  tl_2d_mask              = np.zeros((tl_hm_pred.shape))
  b, h, w                 = tl_mask[:,0], tl_mask[:,1], tl_mask[:,2]
  tl_2d_mask[b, h, w, 0]  = 1
  tl_2d_mask              = tf.nn.max_pool2d(tl_2d_mask, ksize=(4,4), strides=1, padding="SAME")
  
  br_2d_mask              = np.zeros((br_hm_pred.shape))
  b, h, w                 = br_mask[:,0], br_mask[:,1], br_mask[:,2]
  br_2d_mask[b, h, w, 0]  = 1
  br_2d_mask              = tf.nn.max_pool2d(br_2d_mask, ksize=(4,4), strides=1, padding="SAME")
  

  corner_loss_tl          = focal_loss(tl_hm_pred, tl_2d_mask)
  corner_loss_br          = focal_loss(br_hm_pred, br_2d_mask)
  
  return 0.1*push_loss, 0.1*pull_loss, corner_loss_tl, corner_loss_br
