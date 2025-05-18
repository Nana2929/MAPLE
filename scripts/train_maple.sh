export CUDA_VISIBLE_DEVICES=1 # change to your device id
export WANDB_MODE=dryrun
aspect_select_strategy="supervised"
feat_select_strategy="longest"
ckptdir="reproduce"      # ckptdir="dbloss_nomerged_noff" dir name under `./checkpoints` # PAPER VERSION
max_test_aspect_tokens=3 # K in the paper
# generate a `checkpoints` dir if not already present
mkdir -p checkpoints
# run training and scoring
# DEFAULT: text_epochs=30, epochs=5
for dataset in "yelp"; do # "yelp23"
    for index in 1 2 3 4 5; do
        echo " Training MAPLE $ckptdir, $dataset; index $index $strategy"
        python3 maple/train_maple.py --feat_select_strategy=${feat_select_strategy} --ckptdir=${ckptdir} --aspect_rating_reg=0.01 \
        --text_epochs=30 \
        --epochs=5 --endure_times=1 \
        --auto_arg_by_dataset=$dataset --index=$index \
        --aspect_select_strategy=${aspect_select_strategy} \
        --max_test_aspect_tokens=$max_test_aspect_tokens
        done
done