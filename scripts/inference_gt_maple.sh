# DATASET=yelp
# set -e
export CUDA_VISIBLE_DEVICES=0
CKPTDIR="checkpoints/reproduce"
STRATEGY="gt"
K=1
for fold in 1 2 3 4 5; do
    for DATASET in "yelp"; do
        # if [ -f "$CKPTDIR/$DATASET/$fold/generated_${STRATEGY}_k=${max_test_aspect_tokens}.json" ]; then
        #     echo "File exists: $CKPTDIR/$DATASET/$fold/generated_${STRATEGY}_k=${max_test_aspect_tokens}.json"
        # else
        python3 generate.py -a=$DATASET -i=$fold -s=$STRATEGY --max_samples=10000 --ckptdir=$CKPTDIR  \
        --max_test_aspect_tokens=${K}
        # --do_retrieval
    done
done

