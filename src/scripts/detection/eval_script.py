#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
import json
from pathlib import Path

import joblib
import numpy as np
import pyprojroot
import torch
from tqdm import tqdm

import detection


def add_overlap_vals(gt_row, pred_rows):
    """adds column with computed overlap values
    (Intersection over Union) to the rows of a dataframe ``pred_rows``
    where each row corresponds to a predicted bounding box,
    and overlap is with the ground truth bounding box in ``gt_row``
    """
    pbar = tqdm(pred_rows.itertuples())

    ovmax_vals = []
    for pred_row_ind, pred_row in enumerate(pbar):
        pred_bbox = np.array(
            [pred_row.xmin, pred_row.ymin, pred_row.xmax, pred_row.ymax]
        ).astype(float)

        bbox_gt = gt_row.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)

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
            ovmax_vals.append(ovmax)
    pred_rows['overlap'] = ovmax_vals
    return pred_rows


def eckstein_analysis(gt_df, pred_df):
    gt_target_df = gt_df[gt_df['class'] == 1]  # 1 = target, 2 = distractors

    THRESH = 0.5

    hit_miss_setsize = {
        setsize: {'hits': 0, 'misses': 0} for setsize in gt_target_df['img_set_size'].unique()
    }

    pbar = tqdm(gt_target_df.id.unique())
    for id in pbar:
        gt_id_df = gt_target_df[gt_target_df.id == id].copy()  # will be a single row because only 1 target / image
        setsize = gt_id_df['img_set_size'].values.item()
        # we want all the predicted boxes, not just the ones that are target class
        pred_id_df = pred_df[pred_df.id == id].copy()
        pred_overlap_df = add_overlap_vals(gt_id_df, pred_id_df)
        pred_overlap_df = pred_overlap_df[pred_overlap_df.overlap > THRESH].reset_index()
        if len(pred_overlap_df) < 1:
            # catch this first to prevent crash due to running argmax on 0 rows
            hit_miss_setsize[setsize]['misses'] += 1
        elif pred_overlap_df.iloc[pred_overlap_df['score'].idxmax()]['class'] == 1.0:
            hit_miss_setsize[setsize]['hits'] += 1
        else:
            hit_miss_setsize[setsize]['misses'] += 1
        pbar.set_description(
            f'id: {id}, n pred. bbox: {len(pred_id_df)}, n. overlap with target: {len(pred_overlap_df)}, '
            f'set size {setsize}, hits: {hit_miss_setsize[setsize]["hits"]}, '
            f'misses: {hit_miss_setsize[setsize]["misses"]}'
        )
    hit_rate_setsize = {}
    for setsize, hits_misses in hit_miss_setsize.items():
        setsize = int(setsize)  # cast from np.int64
        hits, misses = hit_miss_setsize[setsize]['hits'], hit_miss_setsize[setsize]['misses']
        hr = hits / (hits + misses)
        hit_rate_setsize[setsize] = hr
    return hit_rate_setsize


def main(args):
    results_dst = Path(args.results_dst)
    results_dir_path = results_dst / args.results_dir
    if not results_dir_path.exists():
        raise NotADirectoryError(
            f'results_dir not found: {results_dir_path}'
        )

    eval_results_dir_path = detection.results_dir.make_results_dir(
        results_root=results_dir_path, prefix='eval_results_'
    )

    args_vars = vars(args)
    print(
        f'running eval with args: {args_vars}'
    )
    with eval_results_dir_path.joinpath(f'args.json').open('w') as fp:
        json.dump(args_vars, fp)

    tfm, target_tfm = detection.transforms.get_transform()

    dataset_test = detection.datasets.SearchstimsDetection(root=args.dataset_root,
                                                           csv_file=args.csv_file,
                                                           split='test',
                                                           transform=tfm,
                                                           target_transform=target_tfm)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=detection.collate.frcnn_collate_fn
    )

    num_classes = 3  # background, target, distractor

    # need to convert args to Tuple[Tuple[Int]]
    anchor_sizes = (tuple(args.anchor_sizes),)
    aspect_ratios = (tuple(args.aspect_ratios),)
    print('building model')
    model = detection.model.get_model(num_classes,
                                      anchor_sizes=anchor_sizes,
                                      aspect_ratios=aspect_ratios,
                                      rpn_pre_nms_top_n_test=args.rpn_pre_nms_top_n_test,
                                      rpn_post_nms_top_n_test=args.rpn_post_nms_top_n_test,
                                      rpn_score_thresh=args.rpn_score_thresh
                                      )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device);

    print('loading model')
    ckpt_path = results_dir_path / args.ckpt_filename
    model.load_state_dict(
        torch.load(ckpt_path)
    )

    print(f'\nevaluating model on test set')
    (mAP,
     ap_class_map,
     gt_df_out,
     gt_df,
     pred_df) = detection.engine.evaluate(model, data_loader_test, device,
                                          overlap_thresh=args.overlap_thresh)
    print(f'\nmAP: {mAP:0.4f}\n')

    ap_class_map_filename = eval_results_dir_path.joinpath(f'ap_class_map.joblib')
    joblib.dump(ap_class_map, ap_class_map_filename)

    for gt_df_class in gt_df_out:
        class_label = gt_df_class['class'].unique()
        assert len(class_label) == 1
        class_label = class_label[0]
        gt_df_class_csv_filename = eval_results_dir_path.joinpath(f'gt_df_class_{class_label}.csv')
        gt_df_class.to_csv(gt_df_class_csv_filename, index=False)

    gt_df.to_csv(eval_results_dir_path / 'gt_df.csv')
    pred_df.to_csv(eval_results_dir_path / 'pred_df.csv')

    hit_rate_setsize = eckstein_analysis(gt_df, pred_df)
    with eval_results_dir_path.joinpath(f'hit_rate_setsize.json').open('w') as fp:
        json.dump(hit_rate_setsize, fp)


STIMS_ROOT = pyprojroot.here() / '..' / 'visual_search_stimuli'

DATASET_ROOT = STIMS_ROOT / 'alexnet_multiple_stims_v2'
CSV_FILE = DATASET_ROOT / 'alexnet_multiple_stims_balanced_split.csv'
RESULTS_DST = pyprojroot.here() / 'results' / 'detection'
CKPT_FILENAME = 'checkpoint.pt'


def get_parser():
    parser = ArgumentParser()

    # paths / filenames
    parser.add_argument('results_dir', type=str,
                        help='name of results dir, of the form "results_%y%m%d_%H%M%S"')
    parser.add_argument('--dataset-root', default=str(DATASET_ROOT), help='path to root of dataset')
    parser.add_argument('--csv-file', default=str(CSV_FILE), help='csv with dataset split')
    parser.add_argument('--results-dst', default=str(RESULTS_DST),
                        help='path to root dir where results directories were created')
    parser.add_argument('--ckpt-filename', default=CKPT_FILENAME,
                        help='filename for saved checkpoint')

    # model parameters
    parser.add_argument('--anchor-sizes', nargs='+', default=[15, 30, 45],
                        help='candidate anchor sizes for faster-rcnn')
    parser.add_argument('--aspect-ratios', nargs='+',
                        default=[1.0], help='image aspect ratios for training faster-rcnn')
    parser.add_argument('--rpn-score-thresh', default=0.0, type=float,
                        help='rpn score threshold for faster-rcnn')
    parser.add_argument('--rpn_pre_nms_top_n_test', default=1000, type=int,
                        help='rpn score threshold for faster-rcnn')
    parser.add_argument('--rpn_post_nms_top_n_test', default=1000, type=int,
                        help='rpn score threshold for faster-rcnn')

    # evaluation
    parser.add_argument('--overlap-thresh', type=float, default=0.5,
                        help='overlap threshold to use when computing mAP for evaluation')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
