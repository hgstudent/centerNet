def extract_masks(gt, dtype_="int32"):
  gt = tf.cast(gt, dtype=dtype_)
  cond = tf.where(tf.math.not_equal(gt, 0))
  c = cond[::2, 0]
  c = tf.reshape(c, (-1, 1))
  c = tf.cast(c, dtype=dtype_)
  c = tf.concat([c,tf.reshape(tf.gather_nd(gt, cond), (2,-1))], axis=-1)
  return c
