CKPT_DIR="checkpoints/reproduce"
export CUDA_VISIBLE_DEVICES=1
model="maple"
max_samples=10000
dataset="yelp"
fold_num=1 # change to 5
# !! PREPARE all inference files for 1 dataset before running this script!!
filenames=("generated_supervised_k=1.jsonl")
# "generated_supervised_k=2.jsonl"
# "generated_supervised_k=3.jsonl" "generated_supervised_k=4.jsonl"
# "generated_gt.jsonl" "generated_heuristic_k=3.jsonl"

for filename in "${filenames[@]}"; do
    echo "Run diversity and mauve..."
    # (iFMR, GT-FMR) run feat-metrics
    # they run 5 folds by default
    echo "Running feat-metrics for $filename"
    python3 metrics/run_feat_metrics.py \
                --maple_ckpt_dir=$CKPT_DIR \
                --auto_arg_by_dataset=$dataset \
                --max_samples=${max_samples} \
                --model=$model \
                --fold_num=$fold_num \
                --input_filename=$filename
    # (FCR for all quartiles and overall-FCR) run fcr
    echo "Running fcr for $filename"
    python3 metrics/run_fcr.py \
                --maple_ckpt_dir=$CKPT_DIR \
                --auto_arg_by_dataset=$dataset \
                --max_samples=${max_samples} \
                --model=$model \
                --q_splits=4 \
                --fold_num=$fold_num \
                --input_filename=$filename
    for index in $(seq 1 $fold_num); do
        filepath=${CKPT_DIR}/${dataset}/${index}/${filename}
        echo "filepath: $filepath"
        # Diversity
        python3 metrics/run_diversity.py --input_file=$filepath -a=$dataset -i=$index --model=$model\
        --max_samples=${max_samples} --do_diversity --do_feature
    done
done

# MAUVE takes time to run; run it separately
for filename in "${filenames[@]}"; do
    for index in $(seq 1 $fold_num); do
        filepath=${CKPT_DIR}/${dataset}/${index}/${filename}
        echo "filepath: $filepath"
        # (Text-gen quality) mauve
        python3 metrics/run_mauve.py --input_file=$filepath -a=$dataset -i=$index\
        --max_samples=${max_samples}
    done
done