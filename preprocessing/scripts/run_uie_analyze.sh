
MODEL_DIR="../checkpoints/uie_finetune_checkpoints/run1"
SCHEMA_FOLDER="UIE/data/text2spotasoc/absa/14-15-16res"
GEST_TEXT_COL_NAME="review_text"     #"caption" for Yelp'23
BATCH_SIZE=32
CONFIG_MAP="UIE/config/offset_map/closest_offset_en.yaml"
# ============ GEST  ============

for SPLIT in "test" "train" "val"; do
    DATA_DIR="../Gest/Google-data/filter_all_t/${SPLIT}"
    OUTPUT_DIR="../Gest/uie_annotated_data/${SPLIT}"
    LOG_FILE_PATH=${OUTPUT_DIR}/${SPLIT}_uie_.log

    echo "DATA_DIR: $DATA_DIR"
    echo "MODEL_DIR: $MODEL_DIR"
    echo "SCHEMA_FOLDER: $SCHEMA_FOLDER"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES=1 python3 uie_analyze.py \
        -d $DATA_DIR \
        -m $MODEL_DIR \
        -c $CONFIG_MAP \
        -t $GEST_TEXT_COL_NAME \
        -o $OUTPUT_DIR \
        -l $LOG_FILE_PATH \
        -b $BATCH_SIZE
done
# run postprocess
# python3 postprocess_uie.py --auto_arg_by_dataset=gest


# ============ YELP'23 ============

# reviews
YELP_DIR="../yelp_2023/yelp_2023_pruned/yelp_academic_dataset_review_pruned"
YELP_OUTPUT_DIR="../yelp_2023/yelp_2023_pruned/uie_annotated_data"
YELP_LOG_FILE_PATH=${YELP_OUTPUT_DIR}/yelp_2023_uie_.log
YELP_TEXT_COL_NAME="text"

echo "YELP_DIR: $YELP_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "SCHEMA_FOLDER: $SCHEMA_FOLDER"
echo "YELP_OUTPUT_DIR: $YELP_OUTPUT_DIR"



CUDA_VISIBLE_DEVICES=1 python3 uie_analyze.py \
    -d $YELP_DIR \
    -m $MODEL_DIR \
    -c $CONFIG_MAP \
    -t $YELP_TEXT_COL_NAME \
    -o $YELP_OUTPUT_DIR \
    -l $YELP_LOG_FILE_PATH \
    -b $BATCH_SIZE

# photos "caption"
# YELP_PHOTOS_TEXT_COL_NAME="caption"
# python3 postprocess_uie.py --auto_arg_by_dataset=yelp23
python3 uie_analyze_and_postprocess.py \
    --data_path="/home/P76114511/projects/enrich_rec_dataset/nlg4rec_data_symlink/Yelp/reviews.pickle" \
    --output_dir="/home/P76114511/projects/enrich_rec_dataset/nlg4rec_data_symlink/Yelp/uie_annotated_data"
