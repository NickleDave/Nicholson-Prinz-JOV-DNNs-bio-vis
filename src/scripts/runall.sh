# generate all source data for figures from experimental results
PROJECT_ROOT=~/Documents/repos/L2M/untangling-visual-search/

cd $(PROJECT_ROOT)

# experiment 1, searchstims
## source data for results figures
python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims \
  --source_data_root results/searchstims/source_data/3stims \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_three_stims_38400samples_balanced_split.csv

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/10stims \
  --source_data_root results/searchstims/source_data/10stims

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims_white_background/ \
  --source_data_root results/searchstims/source_data/3stims_white_background \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_three_stims_38400samples_balanced_split.csv \
  --net_names alexnet

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/10stims_white_background/ \
  --source_data_root results/searchstims/source_data/10stims_white_background --net_names alexnet

## source data for training history figures
python src/scripts/experiment-1-searchstims/generate_source_data_training_histories_csv.py \
  --ckpt_root results/searchstims/checkpoints/3stims \
  --source_data_root results/searchstims/source_data/3stims

python src/scripts/experiment-1-searchstims/generate_source_data_training_histories_csv.py \
  --ckpt_root results/searchstims/checkpoints/10stims \
  --source_data_root results/searchstims/source_data/10stims

python src/scripts/experiment-1-searchstims/generate_source_data_training_histories_csv.py \
  --ckpt_root results/searchstims/checkpoints/10stims_white_background/ \
  --source_data_root results/searchstims/source_data/10stims_white_background/ \
  --net_names alexnet

python src/scripts/experiment-1-searchstims/generate_source_data_training_histories_csv.py \
  --ckpt_root results/searchstims/checkpoints/10stims_white_background/ \
  --source_data_root results/searchstims/source_data/10stims_white_background/ \
  --net_names alexnet
