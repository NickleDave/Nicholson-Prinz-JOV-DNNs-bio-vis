from typing import NamedTuple

import torch
import torchvision.transforms as vis_transforms

from searchnets import transforms as searchnets_transforms


class BndBox(NamedTuple):
    xmin : int
    ymin : int
    xmax : int
    ymax : int

    @classmethod
    def from_dict(cls, bndbox_dict):
        """casts "bndbox" value from nested dict of xml annotation to an instance of this NamedTuple"""
        casted = {field: cls.__annotations__[field](value) for field, value in bndbox_dict.items()}
        return cls(**casted)


SEARCHSTIMS_CLASSES = [
    't',
    'd',
    'dV',
    'dH',
    'dT',
    'dL',
    'dx',
    'do',
]

SEARCHSTIMS_CLASS_INT_MAP = {
    k: (1 if k == 't' else 2)
    for k in SEARCHSTIMS_CLASSES
}


def voc_xml_to_frcnn_targets(xml_dict, class_int_map=SEARCHSTIMS_CLASS_INT_MAP):
    voc_objects = xml_dict['annotation']['object']
    if isinstance(voc_objects, dict):
        voc_objects = [voc_objects]
    bndboxes = [
        BndBox.from_dict(obj['bndbox']) for obj in voc_objects
    ]
    names = [obj['name'] for obj in voc_objects]

    boxes = [
        [bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax]
        for bndbox in bndboxes
    ]
    labels = [class_int_map[name] for name in names]

    targets = {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
    }

    return targets


class VocXMLToFRCNNTargets:
    def __init__(self, class_int_map=SEARCHSTIMS_CLASS_INT_MAP):
        self.class_int_map = class_int_map

    def __call__(self, xml_dict):
        return voc_xml_to_frcnn_targets(xml_dict, self.class_int_map)


def get_transform():
    tfm = vis_transforms.ToTensor()

    target_tfm = vis_transforms.Compose(
        [
            searchnets_transforms.ParseVocXml(),
            VocXMLToFRCNNTargets(),
        ]
    )

    return tfm, target_tfm
