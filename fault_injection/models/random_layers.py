# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image preprocessing layers."""

import tensorflow.compat.v2 as tf

import numpy as np
from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.preprocessing import image as image_preprocessing
from keras.utils import control_flow_util
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export

import numbers
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops

ResizeMethod = tf.image.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}

H_AXIS = -3
W_AXIS = -2


def check_fill_mode_and_interpolation(fill_mode, interpolation):
  if fill_mode not in {'reflect', 'wrap', 'constant', 'nearest'}:
    raise NotImplementedError(
        'Unknown `fill_mode` {}. Only `reflect`, `wrap`, '
        '`constant` and `nearest` are supported.'.format(fill_mode))
  if interpolation not in {'nearest', 'bilinear'}:
    raise NotImplementedError('Unknown `interpolation` {}. Only `nearest` and '
                              '`bilinear` are supported.'.format(interpolation))


class MyRandomCrop(base_layer.Layer):
  """Randomly crop the images to target height and width.

  This layer will crop all the images in the same batch to the same cropping
  location.
  By default, random cropping is only applied during training. At inference
  time, the images will be first rescaled to preserve the shorter side, and
  center cropped. If you need to apply random cropping at inference time,
  set `training` to True when calling the layer.

  Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

  Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., target_height, target_width, channels)`.

  Args:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    seed: Integer. Used to create a random seed.
  """

  def __init__(self, height, width, seed=None, **kwargs):
    self.height = height
    self.width = width
    self.seed = seed
    self._rng = make_generator(self.seed)
    super(MyRandomCrop, self).__init__(**kwargs)
    #base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomCrop').set(True)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    inputs = tf.convert_to_tensor(inputs)
    unbatched = inputs.shape.rank == 3

    def random_cropped_inputs():
      """Cropped inputs with stateless random ops."""
      shape = tf.shape(inputs)
      if unbatched:
        crop_size = tf.stack([self.height, self.width, shape[-1]])
      else:
        crop_size = tf.stack([shape[0], self.height, self.width, shape[-1]])
      check = tf.Assert(
          tf.reduce_all(shape >= crop_size),
          [self.height, self.width])
      with tf.control_dependencies([check]):
        limit = shape - crop_size + 1
        offset = stateless_random_ops.stateless_random_uniform(
            tf.shape(shape),
            dtype=crop_size.dtype,
            maxval=crop_size.dtype.max,
            seed=self._rng.make_seeds()[:, 0]) % limit
        return tf.slice(inputs, offset, crop_size)

    # TODO(b/143885775): Share logic with Resize and CenterCrop.
    def resize_and_center_cropped_inputs():
      """Deterministically resize to shorter side and center crop."""
      input_shape = tf.shape(inputs)
      input_height_t = input_shape[H_AXIS]
      input_width_t = input_shape[W_AXIS]
      ratio_cond = (input_height_t / input_width_t > (self.height / self.width))
      # pylint: disable=g-long-lambda
      resized_height = control_flow_util.smart_cond(
          ratio_cond,
          lambda: tf.cast(self.width * input_height_t / input_width_t,
                          input_height_t.dtype), lambda: self.height)
      resized_width = control_flow_util.smart_cond(
          ratio_cond, lambda: self.width,
          lambda: tf.cast(self.height * input_width_t / input_height_t,
                          input_width_t.dtype))
      # pylint: enable=g-long-lambda
      resized_inputs = tf.image.resize(
          images=inputs, size=tf.stack([resized_height, resized_width]))

      img_hd_diff = resized_height - self.height
      img_wd_diff = resized_width - self.width
      bbox_h_start = tf.cast(img_hd_diff / 2, tf.int32)
      bbox_w_start = tf.cast(img_wd_diff / 2, tf.int32)
      if unbatched:
        bbox_begin = tf.stack([bbox_h_start, bbox_w_start, 0])
        bbox_size = tf.stack([self.height, self.width, -1])
      else:
        bbox_begin = tf.stack([0, bbox_h_start, bbox_w_start, 0])
        bbox_size = tf.stack([-1, self.height, self.width, -1])
      outputs = tf.slice(resized_inputs, bbox_begin, bbox_size)
      return outputs

    output = control_flow_util.smart_cond(training, random_cropped_inputs,
                                          resize_and_center_cropped_inputs)
    input_shape = inputs.shape.as_list()
    if unbatched:
      output_shape = [self.height, self.width, input_shape[-1]]
    else:
      output_shape = [input_shape[0], self.height, self.width, input_shape[-1]]
    output.set_shape(output_shape)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    input_shape[H_AXIS] = self.height
    input_shape[W_AXIS] = self.width
    return tf.TensorShape(input_shape)

  def get_config(self):
    config = {
        'height': self.height,
        'width': self.width,
        'seed': self.seed,
    }
    base_config = super(MyRandomCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
HORIZONTAL_AND_VERTICAL = 'horizontal_and_vertical'

class MyRandomFlip(base_layer.Layer):
  """Randomly flip each image horizontally and vertically.

  This layer will flip the images based on the `mode` attribute.
  During inference time, the output will be identical to input. Call the layer
  with `training=True` to flip the input.

  Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

  Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

  Attributes:
    mode: String indicating which flip mode to use. Can be `"horizontal"`,
      `"vertical"`, or `"horizontal_and_vertical"`. Defaults to
      `"horizontal_and_vertical"`. `"horizontal"` is a left-right flip and
      `"vertical"` is a top-bottom flip.
    seed: Integer. Used to create a random seed.
  """

  def __init__(self,
               mode=HORIZONTAL_AND_VERTICAL,
               seed=None,
               **kwargs):
    super(MyRandomFlip, self).__init__(**kwargs)
    #base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomFlip').set(True)
    self.mode = mode
    if mode == HORIZONTAL:
      self.horizontal = True
      self.vertical = False
    elif mode == VERTICAL:
      self.horizontal = False
      self.vertical = True
    elif mode == HORIZONTAL_AND_VERTICAL:
      self.horizontal = True
      self.vertical = True
    else:
      raise ValueError('RandomFlip layer {name} received an unknown mode '
                       'argument {arg}'.format(name=self.name, arg=mode))
    self.seed = seed
    self._rng = make_generator(self.seed)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        flipped_outputs = tf.image.stateless_random_flip_left_right(
            flipped_outputs,
            self._rng.make_seeds()[:, 0])
      if self.vertical:
        flipped_outputs = tf.image.stateless_random_flip_up_down(
            flipped_outputs,
            self._rng.make_seeds()[:, 0])
      return flipped_outputs

    output = control_flow_util.smart_cond(training, random_flipped_inputs,
                                          lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'mode': self.mode,
        'seed': self.seed,
    }
    base_config = super(MyRandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MyRandomRotation(base_layer.Layer):
  """Randomly rotate each image.

  By default, random rotations are only applied during training.
  At inference time, the layer does nothing. If you need to apply random
  rotations at inference time, set `training` to True when calling the layer.

  Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format

  Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format

  Attributes:
    factor: a float represented as fraction of 2 Pi, or a tuple of size 2
      representing lower and upper bound for rotating clockwise and
      counter-clockwise. A positive values means rotating counter clock-wise,
      while a negative value means clock-wise. When represented as a single
      float, this value is used for both the upper and lower bound. For
      instance, `factor=(-0.2, 0.3)` results in an output rotation by a random
      amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in an
      output rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
      - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
        reflecting about the edge of the last pixel.
      - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
        filling all values beyond the edge with the same constant value k = 0.
      - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
        wrapping around to the opposite edge.
      - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
        nearest pixel.
    interpolation: Interpolation mode. Supported values: `"nearest"`,
      `"bilinear"`.
    seed: Integer. Used to create a random seed.
    fill_value: a float represents the value to be filled outside the boundaries
      when `fill_mode="constant"`.
  """

  def __init__(self,
               factor,
               fill_mode='reflect',
               interpolation='bilinear',
               seed=None,
               fill_value=0.0,
               **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.lower = factor[0]
      self.upper = factor[1]
    else:
      self.lower = -factor
      self.upper = factor
    if self.upper < self.lower:
      raise ValueError('Factor cannot have negative values, '
                       'got {}'.format(factor))
    check_fill_mode_and_interpolation(fill_mode, interpolation)
    self.fill_mode = fill_mode
    self.fill_value = fill_value
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)
    super(MyRandomRotation, self).__init__(**kwargs)
    #base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomRotation').set(True)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    inputs = tf.convert_to_tensor(inputs)
    original_shape = inputs.shape
    unbatched = inputs.shape.rank == 3
    # The transform op only accepts rank 4 inputs, so if we have an unbatched
    # image, we need to temporarily expand dims to a batch.
    if unbatched:
      inputs = tf.expand_dims(inputs, 0)

    def random_rotated_inputs():
      """Rotated inputs with random ops."""
      inputs_shape = tf.shape(inputs)
      batch_size = inputs_shape[0]
      img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
      img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
      min_angle = self.lower * 2. * np.pi
      max_angle = self.upper * 2. * np.pi
      angles = self._rng.uniform(
          shape=[batch_size], minval=min_angle, maxval=max_angle)
      return transform(
          inputs,
          get_rotation_matrix(angles, img_hd, img_wd),
          fill_mode=self.fill_mode,
          fill_value=self.fill_value,
          interpolation=self.interpolation)

    output = control_flow_util.smart_cond(training, random_rotated_inputs,
                                          lambda: inputs)
    if unbatched:
      output = tf.squeeze(output, 0)
    output.set_shape(original_shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'factor': self.factor,
        'fill_mode': self.fill_mode,
        'fill_value': self.fill_value,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(MyRandomRotation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def make_generator(seed=None):
  """Creates a random generator.

  Args:
    seed: the seed to initialize the generator. If None, the generator will be
      initialized non-deterministically.

  Returns:
    A generator object.
  """
  if seed is not None:
    return tf.random.Generator.from_seed(seed)
  else:
    return tf.random.Generator.from_non_deterministic_state()


def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
  return _RESIZE_METHODS[interpolation]



def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape


class MyDropout(tf.keras.layers.Layer):
    def __init__(self, rate, seed):
        super(MyDropout, self).__init__()
        self.rate = rate
        self.seed = seed

    def call(self, x, training):
        #seed = [self.seed, self.seed+1]
        seed = self.seed + tf.cast(1000 * tf.reshape(x, [-1])[:2], tf.int32)
        if (not training) or self.rate == 0:
            return x
        keep_prob = 1 - self.rate
        scale = 1 / keep_prob
        x_scale = x * scale

        '''
        random_tensor = stateless_random_ops.stateless_random_uniform(
                        x.shape, seed=seed, dtype=x.dtype)
        '''
        random_shape = x.shape
        random_tensor = tf.random.stateless_uniform(array_ops.shape(x), seed=seed, dtype=x.dtype)
        #tf.random.set_seed(self.seed)
        #random_tensor = tf.random.uniform(array_ops.shape(x), seed=self.seed, dtype=x.dtype)
        keep_mask = tf.cast(random_tensor >= self.rate, dtype=x.dtype)
        return x_scale * keep_mask


def my_dropout(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.
  Note: The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.
  See also: `tf.keras.layers.Dropout` for a dropout layer.
  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.
  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
  expected value is preserved.
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.5, seed = 1).numpy()
  array([[2., 0., 0., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 0., 2., 0., 2.]], dtype=float32)
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.8, seed = 1).numpy()
  array([[0., 0., 0., 5., 5.],
       [0., 5., 0., 5., 0.],
       [5., 0., 5., 0., 5.]], dtype=float32)
  >>> tf.nn.dropout(x, rate = 0.0) == x
  <tf.Tensor: shape=(3, 5), dtype=bool, numpy=
    array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])>
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions. This is useful for dropping whole
  channels from an image or sequence. For example:
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,10])
  >>> tf.nn.dropout(x, rate = 2/3, noise_shape=[1,10], seed=1).numpy()
  array([[0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.]], dtype=float32)
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  if x.get_shape().as_list()[0] == None:
      return x
  seed = [seed, seed+1]
  with ops.name_scope(name, "dropout", [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError("rate must be a scalar tensor or a float in the "
                       "range [0, 1), got %g" % rate)
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going "
                       "to be scaled. Got a %s tensor instead." % x_dtype)
    if is_rate_number and rate == 0:
      # Fast-path: Return the input immediately if rate is non-tensor & is `0`.
      # We trigger this after all error checking
      # and after `x` has been converted to a tensor, to prevent inconsistent
      # tensor conversions/error raising if rate is changed to/from 0.
      #
      # We also explicitly call `random_seed.get_seed` to make sure
      # we don't change the random number generation behavior of
      # stateful random ops by entering a fastpath,
      # despite not generating a random tensor in the fastpath
      random_seed.get_seed(seed)
      return x

    is_executing_eagerly = context.executing_eagerly()
    if not tensor_util.is_tf_type(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if rate_dtype != x_dtype:
        if not rate_dtype.is_compatible_with(x_dtype):
          raise ValueError(
              "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
              (x_dtype.name, rate_dtype.name, rate))
        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger
    # than rate.
    #
    # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = stateless_random_ops.stateless_random_uniform(
        noise_shape, seed=seed, dtype=x_dtype)
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
    # hence a >= comparison is used.
    keep_mask = random_tensor >= rate
    ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret
