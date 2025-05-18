export CUDA_VISIBLE_DEVICES=0
CKPTDIR="checkpoints/reproduce"

for dataset in "yelp"; do
    for fold in 1 2 3 4 5; do
        for strategy in "supervised"; do
            for k in 1 2 3 4; do
            python3 generate.py -a=$dataset -i=$fold -s=$strategy --max_samples=10000 --ckptdir=$CKPTDIR  \
            --max_test_aspect_tokens=${k}
            done
        done
    done
done
