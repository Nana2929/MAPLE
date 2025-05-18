set -e # "yelp" "tripadvisor"
for name in "yelp23"; do
SAVE_DIR="resources"
MODEL_NAME="facebook/bart-large-mnli"
echo $name
mkdir -p ${SAVE_DIR}/${name}
python3 /home/P76114511/projects/enrich_rec_dataset/aspect_classify.py \
    --model_name $MODEL_NAME \
    --auto_arg_by_dataset $name \
    --save_period 10000
done