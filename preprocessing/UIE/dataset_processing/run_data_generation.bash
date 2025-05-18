#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# at level of UIE/dataset_processing
for data_format in absa
do
    python uie_convert.py -format spotasoc -config data_config/${data_format} -output ${data_format}
done

python scripts/data_statistics.py -data converted_data/text2spotasoc/
# After running the script,
# enrich_absa/UIE/dataset_processing/converted_data/text2spotasoc/absa/{dataset_name} eg. 14lap is created