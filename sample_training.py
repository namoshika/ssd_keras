#%%
# %reload_ext autoreload
# %autoreload 2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from tensorflow.keras.preprocessing import image

import src

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

#%%
img_size = (300, 300)
model = src.SSD300(img_size, num_classes=len(src.voc_classes))
model.build((None, *img_size, 3))
model.load_weights('data/weights_SSD300.hdf5', by_name=True)
priors = model.priorboxes

checkpoint_path = './checkpoints/weights.{epoch:02d}-{val_loss:.2f}.ckpt'
callbks = [keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)]

#%%
optim = keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=optim, loss=src.MultiboxLoss(len(src.voc_classes), neg_pos_ratio=2.0).compute_loss)

#%%
latest_cp = tf.train.latest_checkpoint("./checkpoints")
model.load_weights(latest_cp)

#%%
freeze = [
    'input_1',
    'conv1_1', 'conv1_2', 'pool1',
    'conv2_1', 'conv2_2', 'pool2',
    'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
    # 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
]
for L in model.layers:
    L.trainable = not L.name in freeze

#%%
# load ground truth
gt = pickle.load(open('data/gt_pascal.pkl', 'rb'))
keys = sorted(gt.keys())

num_train = int(round(0.8 * len(keys)))
train_keys, val_keys = keys[:num_train], keys[num_train:]
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

#%%
bbox_util = src.BBoxUtility(src.voc_classes, priors)

#%%
path_prefix = "../VOCdevkit/VOC2007/JPEGImages/"
gen = src.Generator(gt, bbox_util, 20, path_prefix, train_keys, val_keys, img_size, do_crop=False)

#%%
res = model.fit_generator(
    gen.generate(True), gen.train_batches, epochs=30, callbacks=callbks,
    validation_data=gen.generate(False), validation_steps=gen.val_batches)

#%%
plt.plot(res.history["loss"])
plt.plot(res.history["val_loss"])

#%%
def show_pred_with_gt(model: keras.Model, filenames: list, colors: list):
    files = [Path("/workspace/dataset/VOCdevkit/VOC2007/JPEGImages") / name for name in filenames]
    grandtruths = np.array([bbox_util.assign_boxes(gt[name]) for name in filenames], dtype=np.float32)
    grandtruths = list(bbox_util.detection_out(grandtruths))
    
    images = [image.img_to_array(image.load_img(filepath)) for filepath in files]
    inputs = np.array([image.img_to_array(image.load_img(filepath, target_size=(300, 300))) for filepath in files])
    predicts = model.predict(inputs)
    predicts = list(bbox_util.detection_out(predicts))

    # iterate images
    for img, grd, prd in zip(images, grandtruths, predicts):
        plt.figure(figsize=(12, 18))
        grdprd = ((prd, "prd"), (grd, "gt"))
        # show grandtruth and predict
        for i, (res, nam) in enumerate(grdprd):
            plt.subplot(1, 2, i + 1)
            plt.title(nam)
            bbox_util.show_detection(res, img, colors)

show_pred_with_gt(model, train_keys[:10], colors)

#%%
show_pred_with_gt(model, val_keys[:10], colors)

#%%
sample_imgs = [
    './data/fish-bike.jpg', './data/cat.jpg',
    './data/boys.jpg', './data/car_cat.jpg', './data/car_cat2.jpg']
bbox_util.show_pred(model, sample_imgs, colors)
