"""
extract activations from neural networks
in response to images from datasets of visual search stimuli


"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import pyprojroot
import searchnets
import torch
import torch.nn
from tqdm import tqdm


ACTIVATIONS = {}


def get_activation(op_name):
    def hook(model, input, output):
        ACTIVATIONS[op_name] = output.detach()
    return hook


SOFTMAXER = torch.nn.Softmax(dim=1)


LAYER_NAME_IND_MAP = {
    # ind to access layer from model
    # varies between models because dropout layers are in different locations
    'alexnet': {
        'fc6': -2,
        'fc7': -5,
    },
    'VGG16': {
        'fc6': -3,
        'fc7': -6,
    },
}


FC_LAYER_NAMES = ('fc6', 'fc7')


def extract_activations_from_model(net_name, model, test_loader, device, dst):
    for layer_name in FC_LAYER_NAMES:
        layer_ind = LAYER_NAME_IND_MAP[net_name][layer_name]
        model.module.classifier[layer_ind].register_forward_hook(get_activation(op_name=layer_name))

    records = []

    pbar = tqdm(test_loader)
    for batch_idx, batch in enumerate(pbar):
        # TODO: are we also getting image path out of dataset?
        x, y_true, set_size = batch['img'].to(device), batch['target'].to(device), batch['set_size']

        with torch.no_grad():
            logits = model(x)
        softmax = SOFTMAXER(logits)
        y_preds = torch.argmax(softmax, dim=1)
        correct = y_preds == y_true

        y_probs_max = softmax.max(dim=1)[0]

        fc6_activations = ACTIVATIONS['fc6']
        fc7_activations = ACTIVATIONS['fc7']

        zipped = zip(
            y_true.cpu().numpy(),
            set_size.cpu().numpy(),
            y_preds.cpu().numpy(),
            correct.cpu().numpy(),
            y_probs_max.cpu().numpy(),
            # activations / vector layer outputs
            softmax.cpu().numpy(),
            logits.cpu().numpy(),
            fc6_activations.cpu().numpy(),
            fc7_activations.cpu().numpy(),
        )

        # shadow some variable names above; it's ok since they're already zipped
        for vector_idx, (y_true,
                         set_size,
                         y_pred,
                         correct,
                         y_prob,
                         # activations / vector layer outputs
                         softmax,
                         logits,
                         fc6_activation,
                         fc7_activation) in enumerate(zipped):
            idx = batch_idx + vector_idx

            row_dict = {
                'idx': idx,
                'y_true': y_true,
                'set_size': set_size,
                'y_pred': y_pred,
                'correct': correct,
                'y_prob': y_prob,
            }

            npz_dict = {
                'logits': logits,
                'softmax': softmax,
                'fc6': fc6_activation,
                'fc7': fc7_activation,
            }
            npz_filename = f'{net_name}-sample-{idx:06d}.npz'
            npz_path = dst / npz_filename
            np.savez(npz_path, **npz_dict)
            row_dict['npz_path'] = str(npz_path)

            records.append(row_dict)

    df = pd.DataFrame.from_records(records)
    return df


DATASET_TYPE = 'searchstims'
LOSS_FUNC = 'CE'


def main(stims,
         dataset_csv_file,
         batch_size,
         num_workers,
         net_name,
         model_train_str,
         output_dir,
         num_nets=8,
         num_classes=2):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        raise NotADirectoryError(
            f"didn't find output dir: {output_dir}"
        )

    # ---- get the dataloader, that will be the same for all nets
    transform, target_transform = searchnets.transforms.util.get_transforms(
        dataset_type=DATASET_TYPE,
        loss_func=LOSS_FUNC,
        pad_size=500  # no effect since we're using searchstims not VOC
    )

    testset = searchnets.datasets.Searchstims(
        csv_file=dataset_csv_file,
        split='test',
        transform=transform,
        target_transform=target_transform
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ---- other set-up stuff
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # paths
    ckpts_root = pyprojroot.here() / 'results' / 'searchstims' / 'checkpoints'
    expt_root = ckpts_root / stims / model_train_str
    output_expt_root = output_dir / stims / model_train_str
    output_expt_root.mkdir(exist_ok=True, parents=True)
    # ---- main loop ----
    for net_number in range(1, num_nets + 1):
        if net_name == 'alexnet':
            model = searchnets.nets.alexnet.build(pretrained=False, weights_path=False, num_classes=num_classes)
        elif net_name == 'VGG16':
            model = searchnets.nets.vgg16.build(pretrained=False, weights_path=False, num_classes=num_classes)

        net_root = expt_root / f'trained_200_epochs/net_number_{net_number}'
        ckpt_path = net_root / f'{net_name}_trained_200_epochs_number_{net_number}-best-val-acc-ckpt.pt'
        checkpoint = torch.load(ckpt_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])

        model.to(device)

        dst = output_expt_root / f'net_number_{net_number}_activations'
        dst.mkdir(exist_ok=True)
        df = extract_activations_from_model(net_name, model, test_loader, device, dst)
        df_path = output_expt_root / f'net_number_{net_number}-activations.csv'
        df.to_csv(df_path)


DATASET_CSV_FILE = '../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv'
DATASET_CSV_PATH = pyprojroot.here() / DATASET_CSV_FILE


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stims',
        type=str,
        default=None,
        help="for 'searchstims' expt, what was searchstims dataset, e.g. '3stims', '10stims'"
    )
    parser.add_argument(
        '--dataset-csv-file',
        type=str,
        default=DATASET_CSV_PATH,
        help="path to .csv file representing dataset, generated by 'searchnets split'"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="batch size, number of samples per batch. Default is 64."
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help="number of workers to parallelize data loading, ~number of CPUs. Default is 4."
    )

    # ---- experiment / config-related args ----
    parser.add_argument(
        '--net-name',
        type=str,
        choices={'alexnet', 'VGG16'},
        help="name of neural network architecture, one of {'alexnet', 'VGG16'}"
    )
    parser.add_argument(
        '--model_train_str',
        type=str,
        help=(
            "string indicating model and training, e.g."
            "'alexnet_initialize_lr_1e-03_three_stims_38400samples_balanced'."
            "This will be the name of the directory SAVE_PATH in the config file used with `searchnets`."
        )
    )
    parser.add_argument(
        '--num-nets',
        type=int,
        default=8,
        help="number of training replicates, default is 8"
    )
    parser.add_argument(
        '--num-classes',
        type=str,
        default=2,
        help="number of classes, default is 2"
    )

    parser.add_argument(
        '--output-dir',
        default='./results/searchstims/activations',
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(stims=args.stims,
         dataset_csv_file=args.dataset_csv_file,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         net_name=args.net_name,
         model_train_str=args.model_train_str,
         num_nets=args.num_nets,
         num_classes=args.num_classes,
         output_dir=args.output_dir
         )
