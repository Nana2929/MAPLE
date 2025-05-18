for dataset in "yelp23"; do
    echo "Creating frequency file for $dataset"
    python3 maple/create_freqfile.py --auto_arg_by_dataset $dataset
done