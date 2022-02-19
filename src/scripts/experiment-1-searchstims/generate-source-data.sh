python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims \
  --source_data_root results/searchstims/source_data/3stims \
  --net_names alexnet,VGG16,CORnet_Z,CORnet_S --methods transfer,initialize --modes classify \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_three_stims_38400samples_balanced_split.csv

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/10stims \
  --source_data_root results/searchstims/source_data/10stims \
  --net_names alexnet,VGG16,CORnet_Z,CORnet_S --methods transfer,initialize --modes classify \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_multiple_stims_128000samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_multiple_stims_128000samples_balanced_split.csv

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims_nih_chest_xray/ \
  --source_data_root results/searchstims/source_data/3stims_nih_chest_xray \
  --net_names alexnet,VGG16,CORnet_Z,CORnet_S --methods transfer --modes classify \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_three_stims_38400samples_balanced_split.csv

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims_CelebASpoof/ \
  --source_data_root results/searchstims/source_data/3stims_CelebASpoof \
  --net_names alexnet,VGG16,CORnet_Z,CORnet_S --methods transfer --modes classify \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_multiple_stims/alexnet_three_stims_38400samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/VGG16_multiple_stims/VGG16_three_stims_38400samples_balanced_split.csv

python src/scripts/experiment-1-searchstims/generate_source_data_csv.py \
  --results_gz_root results/searchstims/results_gz/3stims_big_set_size/ \
  --source_data_root results/searchstims/source_data/3stims_big_set_size/ \
  --net_names alexnet,VGG16 --methods initialize --modes classify \
  --alexnet_split_csv_path ../visual_search_stimuli/alexnet_big_set_and_sample_size/alexnet_three_stims_big_set_size_67200samples_balanced_split.csv \
  --VGG16_split_csv_path ../visual_search_stimuli/alexnet_big_set_and_sample_size/alexnet_three_stims_big_set_size_67200samples_balanced_split.csv
