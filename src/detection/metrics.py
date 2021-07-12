"""evaluation metrics

adapted from:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
under Apache License
https://github.com/facebookresearch/detectron2/blob/master/LICENSE
"""
import numpy as np
from tqdm import tqdm


def voc_ap(rec, prec):
    """Compute VOC AP given precision and recall."""
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def class_avg_precision(gt_df,
                        pred_df,
                        class_int,
                        overlap_thresh=0.5):
    """compute Average Precision for a specified class

    Parameters
    ----------
    gt_df: pandas.DataFrame
        ground truth ('gt') from dataset.
        Each row is a bounding box + class label from an image.
    pred_df : pandas.DataFrame
        same as ground truth, except each row is a candidate bounding box
        output by the network, and has a confidence score associated with it.
    class_int : int
        integer label for class

    Returns
    -------
    rec : numpy.array
        recall values, as a function of sorted objectness scores in pred_df
    prec : numpy.array
        precision values, as a function of sorted objectness scores in pred_df
    ap : scalar
        Average Precision for class ``class_int``,
        as computed by ``detection.metrics.voc_ap``.
    gt_df : pandas.DataFrame
        ground truth for class_int only, with column 'detected' added,
        that will be 'True' where the
    """
    gt_df = gt_df[gt_df['class'] == class_int].copy()
    # add column we use below when deciding if detection is true positive or false positive
    gt_df['detected'] = False
    pred_df = pred_df[pred_df['class'] == class_int].copy()

    pred_df = pred_df.sort_values(by="score", ascending=False).copy()

    tp = np.zeros(len(pred_df))
    fp = np.zeros(len(pred_df))
    pbar = tqdm(pred_df.itertuples())
    for pred_row_ind, pred_row in enumerate(pbar):
        img_id = pred_row.id
        pred_bbox = np.array(
            [pred_row.xmin, pred_row.ymin, pred_row.xmax, pred_row.ymax]
        ).astype(float)

        gt_df_img_id = gt_df[gt_df.id == img_id].copy()
        bbox_gt = gt_df_img_id.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)

        ovmax = -np.inf  # default overlap
        if bbox_gt.size > 0:  # compute overlap
            # determine the (x, y)-coordinates of the intersection rectangle
            ixmin = np.maximum(bbox_gt[:, 0], pred_bbox[0])
            iymin = np.maximum(bbox_gt[:, 1], pred_bbox[1])
            ixmax = np.minimum(bbox_gt[:, 2], pred_bbox[2])
            iymax = np.minimum(bbox_gt[:, 3], pred_bbox[3])
            # compute the area of intersection rectangle
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection_area = iw * ih

            # union
            union_area = (
                    (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0)
                    + (bbox_gt[:, 2] - bbox_gt[:, 0] + 1.0) * (bbox_gt[:, 3] - bbox_gt[:, 1] + 1.0)
            )

            overlaps = intersection_area / (union_area - intersection_area)
            ovmax = np.max(overlaps)
            gt_bbox_ind = np.argmax(overlaps)

        if ovmax > overlap_thresh:
            if not gt_df_img_id.iloc[gt_bbox_ind].detected:
                tp[pred_row_ind] = 1.0
                detected_idx = gt_df_img_id.index[gt_bbox_ind]
                gt_df.loc[detected_idx, 'detected'] = True
            else:  # was already detected
                fp[pred_row_ind] = 1.0
        else:
            fp[pred_row_ind] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(len(gt_df))  # length of ground truth dataframe = tp + fn
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap, gt_df
