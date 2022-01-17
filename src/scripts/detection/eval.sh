python src/scripts/detection/eval_script.py results_210710_201826/

python src/scripts/detection/eval_script.py results_210710_201826/ \
  --rpn_pre_nms_top_n_test 100 \
  --rpn_post_nms_top_n_test 100

python src/scripts/detection/eval_script.py results_210710_201826/ \
  --rpn-score-thresh 0.95 \
  --rpn_pre_nms_top_n_test 100 \
  --rpn_post_nms_top_n_test 100 \

python src/scripts/detection/eval_script.py results_210710_201826/ \
  --rpn-score-thresh 0.95 \
  --rpn_pre_nms_top_n_test 100 \
  --rpn_post_nms_top_n_test 100 \
  --overlap-thresh 0.95
