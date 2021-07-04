#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
import json
from pathlib import Path

import pyprojroot
import torch

import detection


def main(args):
    results_dst = Path(args.results_dst)
    results_dir_path = detection.results_dir.make_results_dir(results_dst)

    with results_dir_path.joinpath('args.json').open('w') as fp:
        json.dump(vars(args), fp)

    tfm, target_tfm = detection.transforms.get_transform()

    dataset = detection.datasets.SearchstimsDetection(root=args.dataset_root,
                                                      csv_file=args.csv_file,
                                                      split='train',
                                                      transform=tfm,
                                                      target_transform=target_tfm)
    dataset_val = detection.datasets.SearchstimsDetection(root=args.dataset_root,
                                                          csv_file=args.csv_file,
                                                          split='val',
                                                          transform=tfm,
                                                          target_transform=target_tfm)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=detection.collate.frcnn_collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=detection.collate.frcnn_collate_fn
    )

    num_classes = 3  # background, target, distractor

    # need to convert args to Tuple[Tuple[Int]]
    anchor_sizes = (tuple(args.anchor_sizes),)
    aspect_ratios = (tuple(args.aspect_ratios),)
    model = detection.model.get_model(num_classes,
                                      anchor_sizes=anchor_sizes,
                                      aspect_ratios=aspect_ratios,
                                      rpn_score_thresh=args.rpn_score_thresh
                                      )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device);

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(1, args.num_epochs + 1):
        detection.engine.train_one_epoch(model,
                                         optimizer,
                                         data_loader,
                                         device,
                                         epoch,
                                         print_freq=10)
        lr_scheduler.step()
        print(f'\nevaluating model for epoch {epoch}')
        mAP, _, _ = detection.engine.evaluate(model, data_loader_val, device,
                                              overlap_thresh=args.overlap_thresh)
        print(f'\nmAP for epoch {epoch}: {mAP:0.4f}\n')

        ckpt_path = results_dir_path / args.ckpt_filename
        torch.save(model.state_dict(), ckpt_path)

    print('finished training.')


STIMS_ROOT = pyprojroot.here() / '..' / 'visual_search_stimuli'

DATASET_ROOT = STIMS_ROOT / 'alexnet_multiple_stims_v2'
CSV_FILE = DATASET_ROOT / 'alexnet_multiple_stims_v2/alexnet_multiple_stims_balanced_split.csv'
RESULTS_DST = pyprojroot.here() / 'results' / 'detection'
CKPT_FILENAME = 'checkpoint.pt'


def get_parser():
    parser = ArgumentParser()

    # paths / filenames
    parser.add_argument('--dataset-root', default=DATASET_ROOT, help='path to root of dataset')
    parser.add_argument('--csv-file', default=CSV_FILE, help='csv with dataset split')
    parser.add_argument('--results-dst', default=RESULTS_DST, help='where results should be saved')
    parser.add_argument('--ckpt-filename', default=CKPT_FILENAME,
                        help='filename for saved checkpoint')

    # training
    parser.add_argument('--num-epochs', default=10, type=int,
                        help='number of epochs (iterations through entire dataset) to train model')

    # model parameters
    parser.add_argument('--anchor-sizes', nargs='+', default=[15, 30, 45],
                        help='candidate anchor sizes for faster-rcnn')
    parser.add_argument('--aspect-ratios', nargs='+',
                        default=[1.0], help='image aspect ratios for training faster-rcnn')
    parser.add_argument('--rpn-score-thresh', default=0.0, type=float,
                        help='rpn score threshold for faster-rcnn')

    # evaluation
    parser.add_argument('--overlap-thresh', type=float, default=0.5,
                        help='overlap threshold to use when computing mAP for evaluation')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
