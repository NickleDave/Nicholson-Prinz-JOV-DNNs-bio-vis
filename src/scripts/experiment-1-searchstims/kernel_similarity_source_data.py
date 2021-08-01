#!/usr/bin/env python
# coding: utf-8
"""script that produces source data for figure analyzing
similarity of kernels/filters in convolutional layers
"""
import pandas as pd
import pyprojroot
import searchnets
import torch


def pairwise_similarity(x, eps=1e-8, self_sim=False):
    x_n = x.norm(dim=1)[:, None]
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    sims = torch.mm(x_norm, x_norm.transpose(0, 1))

    # if self_sim is False, throw out diagonal (self similarity that will be 1)
    if not self_sim:
        n = sims.shape[0]
        sims = sims.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
    
    return sims


ALEXNET_LAYERS = {
    0: 'Conv_1_1',
    3: 'Conv_2_1',
    6: 'Conv_3_1',
    8: 'Conv_4_1',
    10: 'Conv_5_1',
}

VGG16_LAYERS = {
    0: 'Conv_1_1',
    2: 'Conv_1_2',
    5: 'Conv_2_1',
    7: 'Conv_2_2',
    10: 'Conv_3_1',
    12: 'Conv_3_2',
    14: 'Conv_3_3',
    17: 'Conv_4_1',
    19: 'Conv_4_2',
    21: 'Conv_4_3',
    24: 'Conv_5_1',
    26: 'Conv_5_2',
    28: 'Conv_5_3',
}


def main():
    ckpts_root = pyprojroot.here() / 'results/searchstims/checkpoints/3stims'

    for net_name in 'alexnet', 'VGG16':
        results_dirs = {
            'search stimuli (classify)': f'{net_name}_transfer_lr_1e-03_searchstims_classify_38400samples_balanced',
            'ImageNet': f'{net_name}_transfer_lr_1e-03_no_finetune_three_stims_38400samples_balanced',
            'Stylized ImageNet': f'{net_name}_transfer_lr_1e-03_SIN_three_stims_38400samples_balanced',
            'random weights': f'{net_name}_transfer_lr_1e-03_random_three_stims_38400samples_balanced',
            'DomainNet, Clipart domain': f'{net_name}_transfer_lr_1e-03_Clipart_three_stims_38400samples_balanced',
        }

        dfs = []
        for stim_type, results_dir in results_dirs.items():
            results_dir = ckpts_root / results_dir
            print(f'getting similarities for: {results_dir}')
            replicates = sorted(results_dir.joinpath('trained_200_epochs').glob('net_number*/*best-val*pt'))

            for net_number, ckpt_path in enumerate(replicates):
                sd = torch.load(ckpt_path)
                sd = {k.replace('module.', ''): v for k, v in sd['model'].items()}

                if net_name == 'alexnet':
                    model = searchnets.nets.alexnet.build(num_classes=2)
                    layers = ALEXNET_LAYERS
                elif net_name == 'VGG16':
                    model = searchnets.nets.vgg16.build(num_classes=2)
                    layers = VGG16_LAYERS
                model.load_state_dict(sd);

                for layer_ind, layer_name in layers.items():
                    block_num, layer_num  = layer_name.split('_')[1], layer_name.split('_')[2]
                    filters = list(model.features[layer_ind].parameters())[0]
                    filters = torch.flatten(filters, start_dim=1)

                    with torch.no_grad():
                        sims = pairwise_similarity(filters)
                        sims = sims.cpu().numpy().ravel()

                    s = pd.Series(sims, name='similarity')
                    df = pd.DataFrame(s)
                    df['net_name'] = net_name
                    df['layer_name'] = layer_name
                    df['block_num'] = block_num
                    df['layer_num'] = layer_num
                    df['net_number'] = net_number
                    df['stim_type'] = stim_type
                    dfs.append(df)

        df_all = pd.concat(dfs)
        source_data_dst = pyprojroot.here() / 'results' / 'searchstims' / 'source_data' / 'filter-similarity'
        csv_filename = f'filter-similarities-{net_name}.csv'
        df_all.to_csv(source_data_dst / csv_filename, index=False)


if __name__ == '__main__':
    main()
