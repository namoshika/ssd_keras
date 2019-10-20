"""Keras implementation of SSD."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class SSD300(keras.Model):
    def __init__(self, input_shape, num_classes):
        """SSD300 architecture.

        # Arguments
            input_shape: Shape of the input image,
                expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
            num_classes: Number of classes including background.

        # References
            https://arxiv.org/abs/1512.02325
        """
        super().__init__()
        # Block 1 (shape: 300 -> 150)
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')
        # Block 2 (shape: 150 -> 75)
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')
        # Block 3 (shape: 75 -> 38)
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')
        # Block 4 (shape: 38 -> 19)
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')
        self.conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')
        self.conv4_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')
        self.pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')
        # Block 5 (shape: 19 -> 19)
        self.conv5_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')
        self.conv5_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')
        self.conv5_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')
        self.pool5 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5')
        # FC6 (shape: 19 -> 19)
        self.fc6 = layers.Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')
        # FC7 (shape: 19 -> 19)
        self.fc7 = layers.Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')
        # Block 6 (shape: 19 -> 10)
        self.conv6_1 = layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')
        self.conv6_2 = layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')
        # Block 7 (shape: 10 -> 5)
        self.conv7_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')
        self.conv7_2_padding = layers.ZeroPadding2D()
        self.conv7_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')
        # Block 8 (shape: 5 -> 3)
        self.conv8_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')
        self.conv8_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')
        # Last Pool (shape: (3, 3, 256) -> 256)
        self.pool6 = layers.GlobalAveragePooling2D(name='pool6')

        # Prediction from conv4_3
        # loc (shape: (38, 38, 3 * 4) = 17328)
        # conf (shape: (38, 38, 3 * 21) = 90972)
        # prior (shape: (38 * 38 * 3, 4 + 21 + 4))
        variances = [0.1, 0.1, 0.2, 0.2]
        aspect_ratios = [1.0, 2.0, 1/2]
        num_priors = len(aspect_ratios)
        self.conv4_3_norm = Normalize(20, name='conv4_3_norm')
        self.conv4_3_boxloc = layers.Conv2D(num_priors * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')
        self.conv4_3_boxloc_flat = layers.Flatten(name='conv4_3_norm_mbox_loc_flat')
        self.conv4_3_boxcnf = layers.Conv2D(num_priors * num_classes, (3, 3), padding='same', name='conv4_3_norm_mbox_conf')
        self.conv4_3_boxcnf_flat = layers.Flatten(name='conv4_3_norm_mbox_conf_flat')
        conv4_3_priorbox = self.get_priorboxes(input_shape, 30.0, None, variances, aspect_ratios, (38, 38))

        # Prediction from fc7
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        self.fc7_boxloc = layers.Conv2D(num_priors * 4, (3, 3), padding='same', name='fc7_mbox_loc')
        self.fc7_boxloc_flat = layers.Flatten(name='fc7_mbox_loc_flat')
        self.fc7_boxcnf = layers.Conv2D(num_priors * num_classes, (3, 3), padding='same', name='fc7_mbox_conf')
        self.fc7_boxcnf_flat = layers.Flatten(name='fc7_mbox_conf_flat')
        fc7_priorbox = self.get_priorboxes(input_shape, 60.0, 114.0, variances, aspect_ratios, (19, 19))

        # Prediction from conv6_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        self.conv6_2_boxloc = layers.Conv2D(num_priors * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')
        self.conv6_2_boxloc_flat = layers.Flatten(name='conv6_2_mbox_loc_flat')
        self.conv6_2_boxcnf = layers.Conv2D(num_priors * num_classes, (3, 3), padding='same', name='conv6_2_mbox_conf')
        self.conv6_2_boxcnf_flat = layers.Flatten(name='conv6_2_mbox_conf_flat')
        conv6_2_priorbox = self.get_priorboxes(input_shape, 114.0, 168.0, variances, aspect_ratios, (10, 10))

        # Prediction from conv7_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        self.conv7_2_boxloc = layers.Conv2D(num_priors * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')
        self.conv7_2_boxloc_flat = layers.Flatten(name='conv7_2_mbox_loc_flat')
        self.conv7_2_boxcnf = layers.Conv2D(num_priors * num_classes, (3, 3), padding='same', name='conv7_2_mbox_conf')
        self.conv7_2_boxcnf_flat = layers.Flatten(name='conv7_2_mbox_conf_flat')
        conv7_2_priorbox = self.get_priorboxes(input_shape, 168.0, 222.0, variances, aspect_ratios, (5, 5))

        # Prediction from conv8_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        self.conv8_2_boxloc = layers.Conv2D(num_priors * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')
        self.conv8_2_boxloc_flat = layers.Flatten(name='conv8_2_mbox_loc_flat')
        self.conv8_2_boxcnf = layers.Conv2D(num_priors * num_classes, (3, 3), padding='same', name='conv8_2_mbox_conf')
        self.conv8_2_boxcnf_flat = layers.Flatten(name='conv8_2_mbox_conf_flat')
        conv8_2_priorbox = self.get_priorboxes(input_shape, 222.0, 276.0, variances, aspect_ratios, (3, 3))

        # Prediction from pool6
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        self.pool6_boxloc_dense = layers.Dense(num_priors * 4, name='pool6_mbox_loc_flat')
        self.pool6_boxcnf_dense = layers.Dense(num_priors * num_classes, name='pool6_mbox_conf_flat')
        pool6_priorbox = self.get_priorboxes(input_shape, 276.0, 330.0, variances, aspect_ratios, (1, 1))
        
        # Gather all predictions
        # mbox_loc shape: [data, sum(layer_width * layer_height * prior_box * 4:(xmin, ymin, xmax, ymax))]
        # mbox_conf shape: [data, sum(layer_width * layer_height * prior_box * classes)]
        self.priorboxes = np.concatenate([
            conv4_3_priorbox,
            fc7_priorbox,
            conv6_2_priorbox,
            conv7_2_priorbox,
            conv8_2_priorbox,
            pool6_priorbox
        ], axis=0)
        self.boxloc1_1_concat = layers.Concatenate(axis=1, name='mbox_loc')
        self.boxcnf1_1_concat = layers.Concatenate(axis=1, name='mbox_conf')
        num_boxes = self.priorboxes.shape[0]
        self.boxloc1_2_reshape = layers.Reshape((num_boxes, 4), name='mbox_loc_final')
        self.boxcnf1_2_reshape = layers.Reshape((num_boxes, num_classes), name='mbox_conf_logits')
        self.boxcnf1_3_act1 = layers.Activation('softmax', name='mbox_conf_final')
        self.boxloc1_3_zeros = layers.Lambda(lambda x: tf.zeros((x, self.priorboxes.shape[0], 1)), name="hard_nega_mining")
        # predictions shape: [data, layer_width * layer_height * prior_box, feature:(xmin, ymin, xmax, ymax, *classes, xmin, ymin, xmax, ymax, *variances)]
        self.concat_all = layers.Concatenate(axis=2, name='predictions')
    
    def call(self, x):
        # Block 1 (shape: 300 -> 150)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        # Block 2 (shape: 150 -> 75)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        # Block 3 (shape: 75 -> 38)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        # Block 4 (shape: 38 -> 19)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x_conv4_3 = x
        x = self.pool4(x)
        # Block 5 (shape: 19 -> 19)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        # FC6 (shape: 19 -> 19)
        x = self.fc6(x)
        # FC7 (shape: 19 -> 19)
        x = self.fc7(x)
        x_fc7 = x
        # Block 6 (shape: 19 -> 10)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x_conv6_2 = x
        # Block 7 (shape: 10 -> 5)
        x = self.conv7_1(x)
        x = self.conv7_2_padding(x)
        x = self.conv7_2(x)
        x_conv7_2 = x
        # Block 8 (shape: 5 -> 3)
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x_conv8_2 = x
        # Last Pool (shape: (3, 3, 256) -> 256)
        x = self.pool6(x)
        x_pool6 = x

        # Prediction from conv4_3
        # loc (shape: (38, 38, 3 * 4) = 17328)
        # conf (shape: (38, 38, 3 * 21) = 90972)
        # prior (shape: (38 * 38 * 3, 4 + 21 + 4))
        aspect_ratios = [1.0, 2.0, 1/2]
        num_priors = len(aspect_ratios)
        x_conv4_3_norm = self.conv4_3_norm(x_conv4_3)
        x_conv4_3_boxloc = self.conv4_3_boxloc(x_conv4_3_norm)
        x_conv4_3_boxloc = self.conv4_3_boxloc_flat(x_conv4_3_boxloc)
        x_conv4_3_boxcnf = self.conv4_3_boxcnf(x_conv4_3_norm)
        x_conv4_3_boxcnf = self.conv4_3_boxcnf_flat(x_conv4_3_boxcnf)

        # Prediction from fc7
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        x_fc7_boxloc = self.fc7_boxloc(x_fc7)
        x_fc7_boxloc = self.fc7_boxloc_flat(x_fc7_boxloc)
        x_fc7_boxcnf = self.fc7_boxcnf(x_fc7)
        x_fc7_boxcnf = self.fc7_boxcnf_flat(x_fc7_boxcnf)

        # Prediction from conv6_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        x_conv6_2_boxloc = self.conv6_2_boxloc(x_conv6_2)
        x_conv6_2_boxloc = self.conv6_2_boxloc_flat(x_conv6_2_boxloc)
        x_conv6_2_boxcnf = self.conv6_2_boxcnf(x_conv6_2)
        x_conv6_2_boxcnf = self.conv6_2_boxcnf_flat(x_conv6_2_boxcnf)

        # Prediction from conv7_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        x_conv7_2_boxloc = self.conv7_2_boxloc(x_conv7_2)
        x_conv7_2_boxloc = self.conv7_2_boxloc_flat(x_conv7_2_boxloc)
        x_conv7_2_boxcnf = self.conv7_2_boxcnf(x_conv7_2)
        x_conv7_2_boxcnf = self.conv7_2_boxcnf_flat(x_conv7_2_boxcnf)

        # Prediction from conv8_2
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        x_conv8_2_boxloc = self.conv8_2_boxloc(x_conv8_2)
        x_conv8_2_boxloc = self.conv8_2_boxloc_flat(x_conv8_2_boxloc)
        x_conv8_2_boxcnf = self.conv8_2_boxcnf(x_conv8_2)
        x_conv8_2_boxcnf = self.conv8_2_boxcnf_flat(x_conv8_2_boxcnf)

        # Prediction from pool6
        aspect_ratios = [1.0, 1.0, 2.0, 1/2, 3.0, 1/3]
        num_priors = len(aspect_ratios)
        x_pool6_boxloc = self.pool6_boxloc_dense(x_pool6)
        x_pool6_boxcnf = self.pool6_boxcnf_dense(x_pool6)
        # Gather all predictions
        # mbox_loc shape: [data, sum(layer_width * layer_height * prior_box * 4:(xmin, ymin, xmax, ymax))]
        x_boxloc1_1_concat = self.boxloc1_1_concat([
            x_conv4_3_boxloc,
            x_fc7_boxloc,
            x_conv6_2_boxloc,
            x_conv7_2_boxloc,
            x_conv8_2_boxloc,
            x_pool6_boxloc
        ])
        # mbox_conf shape: [data, sum(layer_width * layer_height * prior_box * classes)]
        x_boxcnf1_1_concat = self.boxcnf1_1_concat([
            x_conv4_3_boxcnf,
            x_fc7_boxcnf,
            x_conv6_2_boxcnf,
            x_conv7_2_boxcnf,
            x_conv8_2_boxcnf,
            x_pool6_boxcnf
        ])

        x_boxloc1_2_reshape = self.boxloc1_2_reshape(x_boxloc1_1_concat)
        x_boxloc1_3_zeros = self.boxloc1_3_zeros(tf.shape(x_boxloc1_2_reshape)[0])
        x_boxcnf1_2_reshape = self.boxcnf1_2_reshape(x_boxcnf1_1_concat)
        x_boxcnf1_3_act1 = self.boxcnf1_3_act1(x_boxcnf1_2_reshape)
        # predictions shape: [data, layer_width * layer_height * prior_box, feature:(xmin, ymin, xmax, ymax, *classes, xmin, ymin, xmax, ymax, *variances)]
        x_concat_all = self.concat_all([x_boxloc1_2_reshape, x_boxcnf1_3_act1, x_boxloc1_3_zeros])
        return x_concat_all
    
    def get_priorboxes(self, img_size, min_size, max_size, variances, aspect_ratios, input_shape):
        # define prior boxes shapes
        box_widths = list()
        box_heights = list()
        for ar in aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            elif ar != 1:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))

        # 特徴マップ1ピクセル当たりの入力画像におけるピクセル数の半分を求める
        layer_height, layer_width = input_shape
        img_w, img_h = img_size
        half_step_x = img_w / layer_width / 2
        half_step_y = img_h / layer_height / 2
        # define centers of prior boxes
        centers_x = np.linspace(half_step_x, img_w - half_step_x, layer_width)
        centers_y = np.linspace(half_step_y, img_h - half_step_y, layer_height)
        centers_x, centers_y = np.meshgrid(centers_x, centers_y)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(aspect_ratios)
        box_half_widths = np.array(box_widths) / 2
        box_half_heights = np.array(box_heights) / 2

        # デフォルトバウンディングボックスの座標生成 (range: 0～1)
        min_x = centers_x - box_half_widths
        min_y = centers_y - box_half_heights
        max_x = centers_x + box_half_widths
        max_y = centers_y + box_half_heights
        min_x = min_x.reshape(-1, 1) / img_w
        min_y = min_y.reshape(-1, 1) / img_h
        max_x = max_x.reshape(-1, 1) / img_w
        max_y = max_y.reshape(-1, 1) / img_h
        # define prior_boxes
        #   dim: [left, top, right, bottom]
        prior_boxes = np.concatenate((min_x, min_y, max_x, max_y), axis=1)
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = prior_boxes.shape[0]
        variances = np.array(variances)
        variances = np.tile(variances.reshape(1, 4), (num_boxes, 1))
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        # define prior_boxes_tensor
        #   dim: [layer_width * layer_height * prior_box, 8:[xmin, ymin, xmax, ymax, *variances]]
        return prior_boxes.astype(np.float32)

class Normalize(layers.Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # layers.Input shape
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
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        # channel size
        shape = (input_shape[3],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = self.add_weight(
            name=f"{self.name}_gamma", shape=shape,
            initializer=tf.initializers.Constant(init_gamma), trainable=True)

    def call(self, x, mask=None):
        output = self.gamma * tf.math.l2_normalize(x, 3)
        return output
