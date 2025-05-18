MODEL_NAME="facebook/bart-large-mnli"
SAVE_DIR="resources"

export CUDA_VISIBLE_DEVICES=0

# ======== GEST ========

# echo "GEST"
# mkdir -p ${SAVE_DIR}/gest
# for split in "test" "train" "val"; do
#     python3 aspect_classify.py \
#         --model_name $MODEL_NAME \
#         --auto_arg_by_dataset gest \
#         --save_period 1000
# done

# =========YELP'23 ========
# echo "YELP"
# mkdir -p ${SAVE_DIR}/yelp
# python3 aspect_classify.py \
#     --model_name $MODEL_NAME \
#     --auto_arg_by_dataset yelp23 \
#     --save_period 1000 \

# ========= Li's datasets =======
# if error exit
set -e # "yelp" "tripadvisor"
for name in  "gest"; do
echo $name
python3 /home/P76114511/projects/enrich_rec_dataset/aspect_classify.py \
    --model_name $MODEL_NAME \
    --auto_arg_by_dataset $name \
    --save_period 1000
done