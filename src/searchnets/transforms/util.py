import torchvision.transforms as vis_transforms

from . import functional as F
from . import transforms

# for preprocessing, normalize using values used when training these models on ImageNet for torchvision
# see https://github.com/pytorch/examples/blob/632d385444ae16afe3e4003c94864f9f97dc8541/imagenet/main.py#L197-L198
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(dataset_type,
                   loss_func,
                   pad_size=F.VSD_PAD_SIZE
                   ):
    """helper function that returns transforms and target transforms used
    with torchvision datasets, given the dataset type and the loss function.

    Parameters
    ----------
    dataset_type : str
        one of {'searchstims', 'VOC'}.
        Either a dataset of images generated by the searchstims library, or
        the PascalVOC dataset (as used to create the Visual Search Difficulty dataset).
    loss_func : str
        one of {'CE', 'BCE'}. Cross entropy or binary cross entropy.
    pad_size : int
        size to which images in PascalVOC / Visual Search Difficulty dataset should be padded.
        Images are padded by making an array of zeros and randomly placing the image within it
        so that the entire image is still within the boundaries of (pad size x pad size).
        Default value is specified by searchnets.transforms.functional.VSD_PAD_SIZE.
        Argument has no effect if the dataset_type is not 'VOC'.

    Returns
    -------
    transform, target_transform
    """
    if dataset_type == 'searchstims':
        if loss_func == 'CE':
            transform = vis_transforms.Compose(
                [vis_transforms.ToTensor(),
                 vis_transforms.Normalize(mean=MEAN, std=STD)]
            )
            target_transform = None
        else:
            raise ValueError(
                f"no transforms specified for dataset_type '{dataset_type}' and loss_func '{loss_func}'"
            )

    elif dataset_type == 'VSD':
        # img transform is the same regardless of the loss
        transform = vis_transforms.Compose(
            [vis_transforms.ToTensor(),
             transforms.RandomPad(pad_size=pad_size),
             ]
        )

        # start with list of target transforms that we can extent depending on loss function
        # and then convert to a vis_transforms.Compose instance below
        target_transform = [
                transforms.ParseVocXml(),
                transforms.ClassIntsFromXml(),
        ]

        if loss_func == 'BCE':
            target_transform.append(
                transforms.OneHotFromClassInts(),
            )
        else:
            raise ValueError(
                f"no transforms specified for dataset_type '{dataset_type}' and loss_func '{loss_func}'"
            )

        target_transform = vis_transforms.Compose(target_transform)

    else:
        raise ValueError(
            f'invalid dataset_type: {dataset_type}'
        )

    return transform, target_transform