import functools
import numpy as np
import pandas as pd
import python_voc_parser as voc
import random as rnd
import tensorflow as tf
import typing as typ
from imgaug import augmenters, augmentables
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from .. import core
from . import data

class PascalVocGenerator(object):
    def __init__(self, dir_path, batch_size: int, output_shape: typ.Tuple[int, int],
                 classes: typ.List[str], box_encoder: 'core.BoxEncoder', train_valid_rate=0.8):

        self.dir_path = Path(dir_path)
        self.dir_path = self.dir_path.resolve()
        self.gt, self.images_size = self._load_gt(self.dir_path, classes)
        self.classes = classes
        self.box_encoder = box_encoder
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.seq = augmenters.Sequential([
            augmenters.Fliplr(0.5),
            augmenters.Multiply((0.5, 1.5)),
            augmenters.MultiplySaturation((0.5, 1.5)),
            augmenters.LinearContrast((0.5, 1.5)),
            # augmenters.Affine(scale=(0.8, 1.2), translate_percent=(-0.3, 0.3), mode="edge"),
            # augmenters.Resize({"height": self.output_shape[0], "width": self.output_shape[1]})
        ])

        keys = list(self.gt.items())
        keys_train_sz = int(len(keys) * train_valid_rate)
        self.keys_train = keys[:keys_train_sz]
        self.keys_val = keys[keys_train_sz:]
        self.num_train_batches = -(-len(self.keys_train) // batch_size)
        self.num_val_batches = -(-len(self.keys_val) // batch_size)

    def _make_data(
        self, img_gt_tuple: typ.Tuple[str, np.ndarray],
        augmentation_is_enable: bool = True,
        preprocess_is_enabled: bool = True) -> typ.Tuple[np.ndarray, np.ndarray]:
        
        img, box = self._load_img_gt(*img_gt_tuple)
        if augmentation_is_enable:
            img, box = self._aug_img(img, box)

        img = img.astype(np.float32)
        if preprocess_is_enabled:
            img = preprocess_input(img)

        box_shape = img_gt_tuple[1].shape
        box = box.to_decoded_array(len(self.classes))
        box = box.reshape(-1, box_shape[1])
        box = self.box_encoder.encode([box])
        box = box.astype(np.float32)

        return img, box[0]

    def _load_img_gt(self, img_name: str, gt_array: np.ndarray) -> typ.Tuple[np.ndarray, data.DetectResult]:
        img_size = self.images_size[img_name]
        img_path = self.dir_path / "JPEGImages" / img_name
        img = image.load_img(img_path, target_size=self.output_shape)
        img = image.img_to_array(img, dtype=np.uint8)
        res = data.DetectResult.from_decoded_array(gt_array, self.classes, img_size)
        return img, res

    def _load_gt(self, dir_path: Path, classes: typ.List[str]) -> typ.Dict[str, np.ndarray]:
        """Load grandtruth from directory.

        Arguments
        ---------------------
        dir_path: PascalVOC directory path.
        classes: Label list without background.

        Return
        ---------------------
        dat_final shape: (xmin, ymin, xmax, ymax, *classes[without background])
        """
        path_image = dir_path / "JPEGImages"
        path_imageset = dir_path / "ImageSets/Main/trainval.txt"
        path_annotations = dir_path / "Annotations"
        parser = voc.VocAnnotationsParser(str(path_image), str(path_imageset), str(path_annotations))
        dat = parser.get_annotation_dataframe()

        dat_img = dat.loc[:, ["filename", "width", "height"]]
        dat_loc = dat.loc[:, ["xmin", "ymin", "xmax", "ymax"]] / \
            dat.loc[:, ["width", "height", "width", "height"]].values
        dat_lbl = dat.loc[:, ["class_name"] * len(classes)] == classes
        dat_lbl = dat_lbl.astype(int)
        dat_lbl.columns = [f"class_{label}" for label in classes]

        decoded_arrays = pd.concat([dat_img.loc[:, "filename"], dat_loc, dat_lbl], axis=1)
        decoded_arrays = {f"{group[0]}.jpg": group[1].values[:, 1:].astype(np.float32) for group in decoded_arrays.groupby(["filename"])}
        image_sizes = {f"{item.filename}.jpg": (item.height, item.width) for item in dat_img.itertuples()}
        return decoded_arrays, image_sizes

    def _aug_img(self, image: np.ndarray, grandtruthe: data.DetectResult) -> typ.Tuple[np.ndarray, data.DetectResult]:
        """Apply data augmentation.

        Arguments
        ---------------------
        image shape: (height, width, channel)
        grandtruthe

        Return
        ---------------------
        imgs_aug shape: (height, width, channel)
        bbxs_aug
        """
        bboxes = [box.to_bbox() for box in grandtruthe.boxes]
        bboxes_on_imgs = augmentables.BoundingBoxesOnImage(bboxes, grandtruthe.image_size)
        
        imgs_aug, bbxs_aug = image, bboxes_on_imgs
        imgs_aug, bbxs_aug = self.seq(image=imgs_aug, bounding_boxes=bbxs_aug)
        bbxs_aug = bbxs_aug.remove_out_of_image()
        bbxs_aug = bbxs_aug.clip_out_of_image()
        bbxs_aug = data.DetectResult.from_bbox_on_image(bbxs_aug, self.classes)

        return imgs_aug, bbxs_aug

    def generate(self, train: bool = True, augmentation_is_enable: bool = True, preprocess_is_enabled: bool = True):
        shape_img = tf.TensorShape([*self.output_shape, 3])
        shape_gt = self.box_encoder.encoded_shape
        make_data = functools.partial(
            self._make_data,
            augmentation_is_enable = augmentation_is_enable,
            preprocess_is_enabled = preprocess_is_enabled)

        if train:
            datset_train = lambda: map(make_data, rnd.sample(self.keys_train, len(self.keys_train)))
            datset_train = tf.data.Dataset.from_generator(
                datset_train, output_types=(tf.float32, tf.float32), output_shapes=(shape_img, shape_gt))
            datset_train = datset_train.batch(self.batch_size)
            datset_train = datset_train.repeat()
            datset_train = datset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return datset_train
        else:
            datset_val = lambda: map(make_data, rnd.sample(self.keys_val, len(self.keys_val)))
            datset_val = tf.data.Dataset.from_generator(
                datset_val, output_types=(tf.float32, tf.float32), output_shapes=(shape_img, shape_gt))
            datset_val = datset_val.batch(self.batch_size)
            datset_val = datset_val.repeat()
            datset_val = datset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return datset_val
