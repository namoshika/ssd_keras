# SSD: Single Shot MultiBox Detector on TensorFlow 2.0
Implemented SSD300 in TensorFlow 2.0 (keras API).  
This SSD300 has some improvements over the forked [repository](https://github.com/rykov8/ssd_keras).

1. Port to TF2.0 (Eager Execution + Keras API)
2. Evaluate ROC curves, mAP Score.
3. Remove predatas (gt_pascal.pkl, prior_boxes_ssd300.pkl).
4. Remove output feature which didn't have trainable weights.

# Requirements
* Python v3.7
* TensorFlow v2.0
* python_voc_parser v1.0.0
* imgaug v0.3.0

# Usage
1. Open demo with VSCode or Jupyter.
    * sample_demo.py: You can try predict and training. My work space (VSCode friendly. I'm attached to breakpoint. XD).
    * sample_demo.ipynb: This is same as sample_demo.py.
2. Run `# データセットをダウンロード` cell.  
   Some minutes after... Downloaded the [PascalVOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/) onto `./data`.
   ```
   ./data/VOCdevkit/VOC2007/
                        |- Annotations
                        |- ImageSets
                        |- JPEGImages
                        |- SegmentationClass
                        |- SegmentationObject
   ```
3. Download the [weights_SSD300.hdf5](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA) onto `./data`. This is weights was ported from the original models.
   ```
   ./data/weights_SSD300.hdf5
   ```
4. Run following cells... XD

# Copyright
Copyright (c) 2020 namoshika  

This repository is forked https://github.com/rykov8/ssd_keras.  
Copyright (c) 2016 Andrey Rykov  