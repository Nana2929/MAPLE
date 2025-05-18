for dataset in "yelp" "tripadvisor" "clothing" "movies_and_tv"; do
    echo "Processing $dataset"
    python3 convert_Li.py --auto_arg_by_dataset $dataset
done

python3 convert_yelp23_nete.py
