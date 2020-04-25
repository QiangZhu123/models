# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A custom module for some common operations used by NASNet.

Functions exposed in this file:
- calc_reduction_layers
- get_channel_index
- get_channel_dim
- global_avg_pool
- factorized_reduction
- drop_path

Classes exposed in this file:
- NasNetABaseCell
- NasNetANormalCell
- NasNetAReductionCell
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
INVALID = 'null'
# The cap for tf.clip_by_value, it's hinted from the activation distribution
# that the majority of activation values are in the range [-6, 6].
CLIP_BY_VALUE_CAP = 6   #激活值裁剪relu6


def calc_reduction_layers(num_cells, num_reduction_layers):#给定了一共有多少个cell，和red cell的个数，将其分割插入
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


@tf.contrib.framework.add_arg_scope
def get_channel_index(data_format=INVALID):
  assert data_format != INVALID
  axis = 3 if data_format == 'NHWC' else 1
  return axis


@tf.contrib.framework.add_arg_scope
def get_channel_dim(shape, data_format=INVALID):#返回通道数
  assert data_format != INVALID
  assert len(shape) == 4      #长度是1
  if data_format == 'NHWC':    #根据数据的大小来返回通道个数
    return int(shape[3])
  elif data_format == 'NCHW':
    return int(shape[1])
  else:
    raise ValueError('Not a valid data_format', data_format)


@tf.contrib.framework.add_arg_scope
def global_avg_pool(x, data_format=INVALID):#全局池化层
  """Average pool away the height and width spatial dimensions of x."""
  assert data_format != INVALID
  assert data_format in ['NHWC', 'NCHW']  #保证图片形状
  assert x.shape.ndims == 4
  if data_format == 'NHWC':                     #是全局方式
    return tf.reduce_mean(x, [1, 2])         
  else:
    return tf.reduce_mean(x, [2, 3])


@tf.contrib.framework.add_arg_scope
def factorized_reduction(net, output_filters, stride, data_format=INVALID):#拆分执行卷积，两个并行的路径
  """Reduces the shape of net without information loss due to striding."""
  assert data_format != INVALID
  if stride == 1:#只需要执行一次卷积
    net = slim.conv2d(net, output_filters, 1, scope='path_conv')
    net = slim.batch_norm(net, scope='path_bn')
    return net
  if data_format == 'NHWC':#调整数据格式
    stride_spec = [1, stride, stride, 1]
  else:
    stride_spec = [1, 1, stride, stride]

  # Skip path 1
  path1 = tf.nn.avg_pool(
      net, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)#平均池化
  path1 = slim.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')#第一个分支，输出一半的通道数

  # Skip path 2
  # First pad with 0's on the right and bottom, then shift the filter to
  # include those 0's that were added.
  if data_format == 'NHWC':
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
    concat_axis = 3
  else:
    pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
    path2 = tf.pad(net, pad_arr)[:, :, 1:, 1:]
    concat_axis = 1

  path2 = tf.nn.avg_pool(
      path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)

  # If odd number of filters, add an additional one to the second path.
  final_filter_size = int(output_filters / 2) + int(output_filters % 2)#第二个路径
  path2 = slim.conv2d(path2, final_filter_size, 1, scope='path2_conv')

  # Concat and apply BN
  final_path = tf.concat(values=[path1, path2], axis=concat_axis)#组合两个路径的结果
  final_path = slim.batch_norm(final_path, scope='final_path_bn')#BN层
  return final_path


@tf.contrib.framework.add_arg_scope
def drop_path(net, keep_prob, is_training=True):
  """Drops out a whole example hiddenstate with the specified probability."""
  if is_training:
    batch_size = tf.shape(net)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.cast(tf.floor(random_tensor), net.dtype)
    keep_prob_inv = tf.cast(1.0 / keep_prob, net.dtype)
    net = net * keep_prob_inv * binary_tensor

  return net


def _operation_to_filter_shape(operation):#输入卷积字符串进行解析‘separable_3x3_4’
  splitted_operation = operation.split('x')#
  filter_shape = int(splitted_operation[0][-1])#卷积的尺寸
  assert filter_shape == int(
      splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):#输入卷积字符串进行解析‘separable_3x3_4’
  splitted_operation = operation.split('_')#分割的个数，X是Xception
  if 'x' in splitted_operation[-1]:#是几层的意思
    return 1
  return int(splitted_operation[-1])#否则按照要求执行


def _operation_to_info(operation):#根据操作的名字来确定卷积的方式separable_3x3_4-> (3, 4)
  """Takes in operation name and returns meta information.

  An example would be 'separable_3x3_4' -> (3, 4).

  Args:
    operation: String that corresponds to convolution operation.

  Returns:
    Tuple of (filter shape, num layers).
  """
  num_layers = _operation_to_num_layers(operation)#分割的要求个数4层
  filter_shape = _operation_to_filter_shape(operation)#卷积大小
  return num_layers, filter_shape


def _stacked_separable_conv(net, stride, operation, filter_size,
                            use_bounded_activation):#一个完整的字符串卷积操作 ，使用的是separable_conv
  """Takes in an operations and parses it to the correct sep operation."""
  num_layers, kernel_size = _operation_to_info(operation)#根据卷积的字符串，提取出卷积的尺寸和个数i.e。4,3
  activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu#指定激活函数
  for layer_num in range(num_layers - 1):#先执行n-1次
    net = activation_fn(net)
    net = slim.separable_conv2d(
        net,
        filter_size,
        kernel_size,
        depth_multiplier=1,
        scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
        stride=stride)
    net = slim.batch_norm(
        net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
    stride = 1
  net = activation_fn(net)
  net = slim.separable_conv2d(
      net,
      filter_size,
      kernel_size,
      depth_multiplier=1,
      scope='separable_{0}x{0}_{1}'.format(kernel_size, num_layers),
      stride=stride)
  net = slim.batch_norm(
      net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers))#最后层要加上BN层
  return net

#池化层字符串的处理
def _operation_to_pooling_type(operation):#池化层字符串的处理
  """Takes in the operation string and returns the pooling type."""
  splitted_operation = operation.split('_')
  return splitted_operation[0]

#池化层字符串的处理
def _operation_to_pooling_shape(operation):
  """Takes in the operation string and returns the pooling kernel shape."""
  splitted_operation = operation.split('_')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)

#池化层字符串的处理
def _operation_to_pooling_info(operation):
  """Parses the pooling operation string to return its type and shape."""
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def _pooling(net, stride, operation, use_bounded_activation):#一个完整的池化字符串执行
  """Parses operation and performs the correct pooling operation on net."""
  padding = 'SAME'
  pooling_type, pooling_shape = _operation_to_pooling_info(operation)#提取池化的大小和类型
  if use_bounded_activation:#特殊激活方式
    net = tf.nn.relu6(net)
  if pooling_type == 'avg':
    net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding=padding)
  elif pooling_type == 'max':
    net = slim.max_pool2d(net, pooling_shape, stride=stride, padding=padding)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return net


class NasNetABaseCell(object):#基cell，为reducecell和normcell的基类
  """NASNet Cell class that is used as a 'layer' in image architectures.

  Args:
    num_conv_filters: The number of filters for each convolution operation.每个卷积滤波个数
    operations: List of operations that are performed in the NASNet Cell in
      order.Cell中的操作列表
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.来确定
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the NASNet cell.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps, use_bounded_activation=False):
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps
    self._use_bounded_activation = use_bounded_activation

  def _reduce_prev_layer(self, prev_layer, curr_layer):#当前层和前一层通道维度进行匹配，对前一层进行卷积处理
    """Matches dimension of prev_layer to the curr_layer."""
    # Set the prev layer to the current layer if it is none
    if prev_layer is None:#如果没有给定前一层，则返回当前层
      return curr_layer
    curr_num_filters = self._filter_size    #当前层通道个数
    prev_num_filters = get_channel_dim(prev_layer.shape)#返回前一层通道数
    
    curr_filter_shape = int(curr_layer.shape[2])#当前层大小
    prev_filter_shape = int(prev_layer.shape[2])#前一层大小
    
    activation_fn = tf.nn.relu6 if self._use_bounded_activation else tf.nn.relu  #激活函数指定
    
    if curr_filter_shape != prev_filter_shape:#如果当前滤波尺寸和浅层不一样大小
      prev_layer = activation_fn(prev_layer)#激活前一层
      prev_layer = factorized_reduction(
          prev_layer, curr_num_filters, stride=2)#当stride=1就是标准卷积，否则执行两个并行的卷积，1*1卷积
    elif curr_num_filters != prev_num_filters:#滤波个数不同
      prev_layer = activation_fn(prev_layer)#激活前一层
      prev_layer = slim.conv2d(
          prev_layer, curr_num_filters, 1, scope='prev_1x1')#执行1*1卷积
      prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')#BN层
    return prev_layer

  def _cell_base(self, net, prev_layer):
    """Runs the beginning of the conv cell before the predicted ops are run."""
    num_filters = self._filter_size#滤波个数64

    # Check to be sure prev layer stuff is setup correctly
    prev_layer = self._reduce_prev_layer(prev_layer, net)#对前一层进行卷积处理，让他和当前层尺寸相当
    #还要对当前层再进行卷积处理一下
    net = tf.nn.relu6(net) if self._use_bounded_activation else tf.nn.relu(net)#激活
    net = slim.conv2d(net, num_filters, 1, scope='1x1')#1*1卷积
    net = slim.batch_norm(net, scope='beginning_bn')#BN层
    # num_or_size_splits=1
    net = [net]
    net.append(prev_layer)#放入当前的状态列表中
    return net

  def __call__(self, net, scope=None, filter_scaling=1, stride=1,
               prev_layer=None, cell_num=-1, current_step=None):#运行函数
    """Runs the conv cell."""
    self._cell_num = cell_num#cell个数，有个net还有一个prev_layer两个输入，但是最开始的时候没有prev_layer
    self._filter_scaling = filter_scaling#通道缩放
    self._filter_size = int(self._num_conv_filters * filter_scaling)#是否改变滤波的个数32*2.0

    i = 0
    with tf.variable_scope(scope):
      net = self._cell_base(net, prev_layer)#[net,pre_layer]将前一层和当前层全部处理一下，再返回，准备作为cell的输入
      for iteration in range(5):#B=5 这个就是一个cell内部
        with tf.variable_scope('comb_iter_{}'.format(iteration)):
          left_hiddenstate_idx, right_hiddenstate_idx = (
              self._hiddenstate_indices[i],
              self._hiddenstate_indices[i + 1])#左边的状态索引，右边的状态索引[0, 1, 0, 1, 0, 1, 3, 2, 2, 0]
          original_input_left = left_hiddenstate_idx < 2            #0，1是指用的初始特征
          original_input_right = right_hiddenstate_idx < 2
          h1 = net[left_hiddenstate_idx]#从状态列表中，根据索引拿出两个状态h1,和h2
          h2 = net[right_hiddenstate_idx]
          '''
          #选中左右两边状态的操作
          ['separable_5x5_2',
                  'separable_3x3_2',
                  'separable_5x5_2',
                  'separable_3x3_2',
                  'avg_pool_3x3',
                  'none',
                  'avg_pool_3x3',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'none']
              '''
          operation_left = self._operations[i]#左边状态的操作
          operation_right = self._operations[i+1]#右边状态的操作
          i += 2#一次做两个，所以索引要往后移动两位
          # Apply conv operations应用卷积
          with tf.variable_scope('left'):
            h1 = self._apply_conv_operation(h1, operation_left,
                                            stride, original_input_left,
                                            current_step)#处理左边输入
          with tf.variable_scope('right'):
            h2 = self._apply_conv_operation(h2, operation_right,
                                            stride, original_input_right,
                                            current_step)

          # Combine hidden states using 'add'.组合两个生成的状态
          with tf.variable_scope('combine'):
            h = h1 + h2
            if self._use_bounded_activation:
              h = tf.nn.relu6(h)

          # Add hiddenstate to the list of hiddenstates we can choose from
          net.append(h)#放入状态列表

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)#所有隐藏状态都进行了选择concat，是当前cell的完整输出

      return net

  def _apply_conv_operation(self, net, operation,
                            stride, is_from_original_input, current_step):#对状态使用指定的卷积
    """Applies the predicted conv operation to net."""
    # Dont stride if this is not one of the original hiddenstates
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = get_channel_dim(net.shape)#获得当前状态的通道数
    filter_size = self._filter_size#滤波大小
    if 'separable' in operation:#根据操作的字符串是否有separable来确定卷积类型
      net = _stacked_separable_conv(net, stride, operation, filter_size,
                                    self._use_bounded_activation)#一次完整的字符串卷积执行
      if self._use_bounded_activation:#边界激活，就是值的裁剪，范围（-CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP）
        net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)
    elif operation in ['none']:#如果没有操作，执行以下处理
      if self._use_bounded_activation:
        net = tf.nn.relu6(net)#直接激活
      # Check if a stride is needed, then use a strided 1x1 here
      if stride > 1 or (input_filters != filter_size):
        if not self._use_bounded_activation:
          net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
        if self._use_bounded_activation:
          net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)
    elif 'pool' in operation:#池化操作
      net = _pooling(net, stride, operation, self._use_bounded_activation)#池化
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
      if self._use_bounded_activation:
        net = tf.clip_by_value(net, -CLIP_BY_VALUE_CAP, CLIP_BY_VALUE_CAP)#进行激活值的裁剪
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'none':#正则化方法droppath
      net = self._apply_drop_path(net, current_step=current_step)
    return net

  def _combine_unused_states(self, net):#输入的是状态列表，要返回的是最终一个cell的输出结果
    """Concatenate the unused hidden states of the cell."""
    used_hiddenstates = self._used_hiddenstates#[1, 1, 1, 0, 0, 0, 0]

    final_height = int(net[-1].shape[2])#最后一个状态特征图的大小
    final_num_filters = get_channel_dim(net[-1].shape)#通道个数
    assert len(used_hiddenstates) == len(net)
    for idx, used_h in enumerate(used_hiddenstates):
      curr_height = int(net[idx].shape[2])
      curr_num_filters = get_channel_dim(net[idx].shape)

      # Determine if a reduction should be applied to make the number of
      # filters match.确定当前选中的隐藏状态是否和最后的输出尺寸匹配，不匹配要进行修改
      should_reduce = final_num_filters != curr_num_filters
      should_reduce = (final_height != curr_height) or should_reduce
      should_reduce = should_reduce and not used_h#没有用过的特征要和最后一个特征concat后输出
      if should_reduce:
        stride = 2 if final_height != curr_height else 1
        with tf.variable_scope('reduction_{}'.format(idx)):
          net[idx] = factorized_reduction(
              net[idx], final_num_filters, stride)#就是调整两个不同大小的特征图尺寸，将所有没有用的状态的尺寸都调整后才能进行concat

    states_to_combine = (
        [h for h, is_used in zip(net, used_hiddenstates) if not is_used])#选择出要用的隐藏状态

    # Return the concat of all the states
    concat_axis = get_channel_index()
    net = tf.concat(values=states_to_combine, axis=concat_axis)#对其进行concat
    return net

  @tf.contrib.framework.add_arg_scope  # No public API. For internal use only.
  def _apply_drop_path(self, net, current_step=None,
                       use_summaries=False, drop_connect_version='v3'):
    """Apply drop_path regularization.

    Args:
      net: the Tensor that gets drop_path regularization applied.
      current_step: a float32 Tensor with the current global_step value,
        to be divided by hparams.total_training_steps. Usually None, which
        defaults to tf.train.get_or_create_global_step() properly casted.
      use_summaries: a Python boolean. If set to False, no summaries are output.
      drop_connect_version: one of 'v1', 'v2', 'v3', controlling whether
        the dropout rate is scaled by current_step (v1), layer (v2), or
        both (v3, the default).

    Returns:
      The dropped-out value of `net`.
    """
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert self._cell_num != -1
        # The added 2 is for the reduction cells
        num_cells = self._total_num_cells
        layer_ratio = (self._cell_num + 1)/float(num_cells)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('layer_ratio', layer_ratio)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        if current_step is None:
          current_step = tf.train.get_or_create_global_step()
        current_step = tf.cast(current_step, tf.float32)
        drop_path_burn_in_steps = self._total_training_steps
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('current_ratio', current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      if use_summaries:
        with tf.device('/cpu:0'):
          tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
      net = drop_path(net, drop_path_keep_prob)
    return net


class NasNetANormalCell(NasNetABaseCell):
  """NASNetA Normal Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps, use_bounded_activation=False):
    operations = ['separable_5x5_2',
                  'separable_3x3_2',
                  'separable_5x5_2',
                  'separable_3x3_2',
                  'avg_pool_3x3',
                  'none',
                  'avg_pool_3x3',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'none']
    used_hiddenstates = [1, 0, 0, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    super(NasNetANormalCell, self).__init__(num_conv_filters, operations,
                                            used_hiddenstates,
                                            hiddenstate_indices,
                                            drop_path_keep_prob,
                                            total_num_cells,
                                            total_training_steps,
                                            use_bounded_activation)


class NasNetAReductionCell(NasNetABaseCell):
  """NASNetA Reduction Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps, use_bounded_activation=False):
    operations = ['separable_5x5_2',
                  'separable_7x7_2',
                  'max_pool_3x3',
                  'separable_7x7_2',
                  'avg_pool_3x3',
                  'separable_5x5_2',
                  'none',
                  'avg_pool_3x3',
                  'separable_3x3_2',
                  'max_pool_3x3']
    used_hiddenstates = [1, 1, 1, 0, 0, 0, 0]
    hiddenstate_indices = [0, 1, 0, 1, 0, 1, 3, 2, 2, 0]#这个是已经预先搜索完成的模型，如果想要调整的话就修改这个列表就可以了
    super(NasNetAReductionCell, self).__init__(num_conv_filters, operations,
                                               used_hiddenstates,
                                               hiddenstate_indices,
                                               drop_path_keep_prob,
                                               total_num_cells,
                                               total_training_steps,
                                               use_bounded_activation)
