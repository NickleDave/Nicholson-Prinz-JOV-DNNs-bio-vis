from pathlib import Path

import searchstims.make
from searchstims.stim_makers import (
    RVvGVStimMaker,
    RVvRHGVStimMaker,
    Two_v_Five_StimMaker,
    TLStimMaker,
    xoStimMaker,
    TStimMaker
)

HERE = Path(__file__).parent

WINDOW_SIZE = (227, 227)
BORDER_SIZE = (30, 30)
GRID_SIZE = (5, 5)
ITEM_BBOX_SIZE = (30, 30)
JITTER = 12

keys = ['RVvGV',
        'RVvRHGV',
        '2_v_5',
        'YT_v_BTYL',
        'YT_v_BTBL',
        'Bx_v_RxBo',
        'Bx_v_RxRo',
        'TvT']
vals = [
    RVvGVStimMaker(target_color='red',
                   distractor_color='green',
                   window_size=WINDOW_SIZE,
                   border_size=BORDER_SIZE,
                   grid_size=GRID_SIZE,
                   item_bbox_size=ITEM_BBOX_SIZE,
                   jitter=JITTER),
    RVvRHGVStimMaker(target_color='red',
                     distractor_color='green',
                     window_size=WINDOW_SIZE,
                     border_size=BORDER_SIZE,
                     grid_size=GRID_SIZE,
                     item_bbox_size=ITEM_BBOX_SIZE,
                     jitter=JITTER),
    Two_v_Five_StimMaker(target_color='white',
                         distractor_color='white',
                         window_size=WINDOW_SIZE,
                         border_size=BORDER_SIZE,
                         grid_size=GRID_SIZE,
                         item_bbox_size=ITEM_BBOX_SIZE,
                         jitter=JITTER,
                         target_number=2,
                         distractor_number=5),
    TLStimMaker(window_size=WINDOW_SIZE,
                border_size=BORDER_SIZE,
                grid_size=GRID_SIZE,
                item_bbox_size=ITEM_BBOX_SIZE,
                jitter=JITTER),
    TLStimMaker(distractor_L_color=(100, 149, 237),
                window_size=WINDOW_SIZE,
                border_size=BORDER_SIZE,
                grid_size=GRID_SIZE,
                item_bbox_size=ITEM_BBOX_SIZE,
                jitter=JITTER),
    xoStimMaker(target_x_color='blue',
                distractor_x_color='red',
                distractor_o_color='blue',
                window_size=WINDOW_SIZE,
                border_size=BORDER_SIZE,
                grid_size=GRID_SIZE,
                item_bbox_size=ITEM_BBOX_SIZE,
                jitter=JITTER),
    xoStimMaker(target_x_color='blue',
                distractor_x_color='red',
                distractor_o_color='red',
                window_size=WINDOW_SIZE,
                border_size=BORDER_SIZE,
                grid_size=GRID_SIZE,
                item_bbox_size=ITEM_BBOX_SIZE,
                jitter=JITTER),
    TStimMaker(window_size=WINDOW_SIZE,
               border_size=BORDER_SIZE,
               grid_size=GRID_SIZE,
               item_bbox_size=ITEM_BBOX_SIZE,
               jitter=JITTER),
]

alexnet_dict = dict(zip(keys, vals))

# in a separate folder in same parent dir as source code root
OUTPUT_DIR = HERE.joinpath('../../../visual_search_stimuli')
TARGET_PRESENT = [3600, 7200, 14400, 28800]
TARGET_ABSENT = [3600, 7200, 14400, 28800]
SET_SIZES = [1, 2, 4, 8]


def main():
    csv_filename = f'alexnet_multiple_stims.csv'
    output_dir = OUTPUT_DIR.joinpath(f'alexnet_multiple_stims_v2')
    searchstims.make.make(root_output_dir=output_dir,
                          stim_dict=alexnet_dict,
                          csv_filename=csv_filename,
                          num_target_present=TARGET_PRESENT,
                          num_target_absent=TARGET_ABSENT,
                          set_sizes=SET_SIZES)


if __name__ == '__main__':
    main()
