import numpy as np
import pickle
import src

voc_classes = [
    'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car',
    'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse','Motorbike',
    'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
]

def test_encode_decode():
    testdat = np.load("data/testdata.npz")
    priors, gts = testdat["priorboxes"], testdat["grandtruths"]
    box_encoder = src.core.BoxEncoder(priors)

    encoded_array = box_encoder.encode([gts])
    decoded_array = box_encoder.decode(encoded_array)

    err_array = decoded_array - gts
    err_array = np.abs(err_array)
    err_array = np.sum(err_array)
    err_array = np.round(err_array, 3)
    assert err_array == 0

def test_parser():
    test_dat = np.load("data/testdata.npz")
    priors, gts = test_dat["priorboxes"], test_dat["grandtruths"]
    
    box_encoder = src.core.BoxEncoder(priors, len(voc_classes))
    parser = src.data.PascalVocGenerator(
        "data/VOCdevkit/VOC2007/", 32, (300, 300), voc_classes, box_encoder)
    gen = parser.generate()

    gen_iter = iter(gen)
    val1 = next(gen_iter)
    val2 = next(gen_iter)
    val3 = next(gen_iter)

    dat_gt = ("000005.jpg", parser.gt["000005.jpg"])
    dat_im, dat_dc = parser._make_data(dat_gt)
    dat_bx = box_encoder.decode(dat_dc[np.newaxis])
    dat_bx = dat_bx[0]
    pass