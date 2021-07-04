import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(num_classes=3,
              anchor_sizes=((32, 64, 128, 256, 512),),
              aspect_ratios=(0.5, 1.0, 2.0),
              rpn_pre_nms_top_n_test=1000,
              rpn_post_nms_top_n_test=1000,
              rpn_score_thresh=None):
    """returns Faster R-CNN model with pre-trained VGG-16 "backbone",
    initialized with specified hyperparameters"""
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)

    # next line, [:-1] because we throw away last max pool layer
    backbone = torchvision.models.vgg16(pretrained=True).features[:-1]

    for param in backbone.parameters():
        param.requires_grad = False

    # next line, [:-2] to get attribute from conv, not ReLu activation
    backbone.out_channels = backbone[-2].out_channels

    # we use the default anchor generator and ROI pooler
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                       rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                       rpn_score_thresh=rpn_score_thresh)
    return model
