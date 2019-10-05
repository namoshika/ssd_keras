"""Some special pupropse layers for SSD."""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.image_data_format() == 'channels_last':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = self.add_weight(
            name=f"{self.name}_gamma", shape=shape,
            initializer=tf.initializers.Constant(init_gamma), trainable=True)

    def call(self, x, mask=None):
        output = tf.math.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.

    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    # References
        https://arxiv.org/abs/1512.02325

    #TODO
        Add possibility not to have variances.
        Add Theano support
    """
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        if K.image_data_format() == 'channels_last':
            self.waxis = 2
            self.haxis = 1
        else:
            self.waxis = 3
            self.haxis = 2
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances, dtype=np.float32)
        self.clip = True
        
        # define prior boxes shapes
        self.box_widths = list()
        self.box_heights = list()
        for ar in self.aspect_ratios:
            if ar == 1 and len(self.box_widths) == 0:
                self.box_widths.append(self.min_size)
                self.box_heights.append(self.min_size)
            elif ar == 1 and len(self.box_widths) > 0:
                self.box_widths.append(np.sqrt(self.min_size * self.max_size))
                self.box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                self.box_widths.append(self.min_size * np.sqrt(ar))
                self.box_heights.append(self.min_size / np.sqrt(ar))
        self.box_widths = 0.5 * np.array(self.box_widths)
        self.box_heights = 0.5 * np.array(self.box_heights)

        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)
    
    def build(self, input_shape):
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = tf.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = tf.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        centers_x, centers_y = tf.meshgrid(linx, liny)
        centers_x = tf.reshape(centers_x, (-1, 1))
        centers_y = tf.reshape(centers_y, (-1, 1))
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = tf.concat((centers_x, centers_y), axis=1)
        prior_boxes = tf.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes = tf.Variable(prior_boxes)
        prior_boxes[:, 0::4].assign(prior_boxes[:, 0::4] - self.box_widths)
        prior_boxes[:, 1::4].assign(prior_boxes[:, 1::4] - self.box_heights)
        prior_boxes[:, 2::4].assign(prior_boxes[:, 2::4] + self.box_widths)
        prior_boxes[:, 3::4].assign(prior_boxes[:, 3::4] + self.box_heights)
        prior_boxes[:, 0::2].assign(prior_boxes[:, 0::2] / img_width)
        prior_boxes[:, 1::2].assign(prior_boxes[:, 1::2] / img_height)
        prior_boxes = tf.reshape(prior_boxes, (-1, 4))
        if self.clip:
            prior_boxes = tf.minimum(tf.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = prior_boxes.shape[0]
        if len(self.variances) == 1:
            variances = tf.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = tf.tile(tf.reshape(self.variances, (1, 4)), (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = tf.concat((prior_boxes, variances), axis=1)
        prior_boxes_tensor = tf.expand_dims(prior_boxes, 0)
        self.prior_boxes_tensor = prior_boxes_tensor

    def call(self, x, mask=None):
        prior_boxes_tensor = tf.tile(self.prior_boxes_tensor, [tf.shape(x)[0], 1, 1])
        return prior_boxes_tensor
