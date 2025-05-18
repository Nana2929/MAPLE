# -*- coding:utf-8 -*-
"""
@File    :   uie_analyze.py
@Time    :   2023/10/30 11:03:11
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Revision based on UIE/evaluation.py (originally UIE/inference.py)
            Input: pre-tokenized text and tokens (use SpaCy)
            Output: uie records
"""

import argparse
import glob
import json
import logging
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Final

import transformers as huggingface_transformers
from tqdm import tqdm

sys.path.append("UIE")

from spacy.lang.en import English
from uie.extraction.record_schema import RecordSchema
from uie.extraction.scorer import *
from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record

from common.io import read_jsonl_file, write_jsonl_file

UIE_PATH = "../checkpoints/uie_finetune_checkpoints/run1"
CONFIG_MAP = "UIE/config/offset_map/closest_offset_en.yaml"
SCHEMA_FOLDER = "UIE/data/text2spotasoc/absa/14-15-16res"
DATA_FOLDER = "UIE/data/text2spotasoc/absa/14-15-16res/test"
split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {"<pad>", "</s>"}


def create_dual_logger(log_file_path: str, verbose: bool = False):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s %(module)s - %(message)s"
    )
    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file_path, mode="w+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Create a stream handler to write to stdout and set the formatter
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger


TASK_DICT: Final[dict] = {
    "entity": EntityScorer,
    "relation": RelationScorer,
    "event": EventScorer,
}


def tokenize(data: dict, text_col_name: str = "text", max_len: int = 256):
    for d in data:
        tok_list = [str(t) for t in tokenizer(d[text_col_name])]
        truncated_tok_list = tok_list[:max_len]
        d["tokens"] = truncated_tok_list
        assert (
            len(truncated_tok_list) <= max_len
        ), f"len(truncated_tok_list)={len(truncated_tok_list)} > max_len={max_len}"
    return data


def timer(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        minutes = (end - start) // 60
        secs = (end - start) % 60
        logger.info(f"time: {minutes}m {secs:.2f}s")
        return result

    return wrapper


def schema_to_ssi(schema: RecordSchema):
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, "")
    return x.strip()


class HuggingfacePredictor:
    def __init__(
        self, model_path, schema_file, max_source_length=256, max_target_length=192
    ) -> None:
        logger.info(f"Loading model from {model_path}")
        self._tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(
            model_path, local_files_only=True
        )
        self._model = (
            huggingface_transformers.T5ForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True
            )
        )
        self._model.cuda()
        self._schema = RecordSchema.read_from_file(schema_file)
        self._ssi = schema_to_ssi(self._schema)
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def predict(self, text):
        text = [self._ssi + x for x in text]
        inputs = self._tokenizer(text, padding=True, return_tensors="pt").to(
            self._model.device
        )

        inputs["input_ids"] = inputs["input_ids"][:, : self._max_source_length]
        inputs["attention_mask"] = inputs["attention_mask"][
            :, : self._max_source_length
        ]

        result = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self._max_target_length,
        )
        return self._tokenizer.batch_decode(
            result, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


@timer
def inference(
    in_filename: str,
    predictor: HuggingfacePredictor,
    sel2record: SEL2Record,
    eval: bool,
    options: argparse.Namespace,
) -> dict:
    data = read_jsonl_file(in_filename)
    text_list = [x[options.text_col_name] for x in data]
    if "tokens" not in data[0]:
        logger.info(
            f"The file is NOT pre-tokenized. Tokenizing {options.text_col_name}..."
        )
        data = tokenize(
            data, text_col_name=options.text_col_name, max_len=options.max_source_length
        )
        # add "tokens" key to each data entry

        token_list = [x["tokens"] for x in data]
        assert "tokens" in data[0], f"tokens should be in data[0] but not!"
    else:
        token_list = [x["tokens"] for x in data]

    # ================= EDA =================

    logger.info(f"Predicting on: {in_filename}")
    logger.info(f"data size: {len(text_list)}")
    avg_token_length = sum([len(x) for x in token_list]) / len(token_list)
    logger.info(f"avg token length (after truncation): {avg_token_length}")
    logger.info(f"[sample] text: {text_list[0]}")
    logger.info(f"[sample] tokens: {token_list[0]}")
    batch_num = math.ceil(len(text_list) / options.batch_size)

    predict = list()
    records = list()
    for index in tqdm(range(batch_num)):
        start = index * options.batch_size
        end = index * options.batch_size + options.batch_size

        pred_seq2seq = predictor.predict(text_list[start:end])
        pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

        predict += pred_seq2seq

    for p, text, tokens in zip(predict, text_list, token_list):
        r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        records += [r]
    # ================= SCORING =================
    results = dict()
    if eval:
        for task, scorer in TASK_DICT.items():
            gold_list = [x[task] for x in read_jsonl_file(in_filename)]
            pred_list = [x[task] for x in records]

            gold_instance_list = scorer.load_gold_list(gold_list)
            pred_instance_list = scorer.load_pred_list(pred_list)

            sub_results = scorer.eval_instance_list(
                gold_instance_list=gold_instance_list,
                pred_instance_list=pred_instance_list,
                verbose=options.verbose,
                match_mode=options.match_mode,
            )
            results.update(sub_results)
    return {
        "records": records,
        "predict": predict,
        "results": results,
        "tokens": token_list,
    }


def main(options: argparse.Namespace):

    logger.info(f"config: {options}")

    data_folder = options.data
    model_path = options.model
    schema_folder = options.schema

    predictor = HuggingfacePredictor(
        model_path=model_path,
        schema_file=f"{schema_folder}/record.schema",
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length,
    )

    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict(schema_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )

    gold_filenames = glob.glob(os.path.join(data_folder, "*.jsonl"))
    # check if data folder exists
    if not os.path.exists(data_folder):
        from os.path import abspath

        data_folder = abspath(data_folder)
        raise ValueError(f"data_folder ({data_folder}) does not exist.")
    logger.info(
        f"Found {len(gold_filenames)} files in {data_folder}: \n{gold_filenames}"
    )

    save_dir = options.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # check if save_dir == data_folder
    if os.path.abspath(save_dir) == os.path.abspath(data_folder):
        raise ValueError(
            f"save_dir ({save_dir}) should not be the same as data_folder ({data_folder}),\
            otherwise the original data will be overwritten!"
        )
    logger.info(f"Inferencing...")
    for gold_filename in gold_filenames:
        logger.info(f"* {gold_filename}")
        out = inference(
            in_filename=gold_filename,
            predictor=predictor,
            sel2record=sel2record,
            eval=options.eval,
            options=options,
        )
        # align the out with the input file
        # add one key called "uie_absa" to each record
        entries = read_jsonl_file(gold_filename)

        for entry in entries:
            entry["uie_absa"] = out["records"].pop(0)
            entry["tokens"] = out["tokens"].pop(0)
        logger.info(f"Peek the first 3 entries: {entries[:3]}")
        out_stem = Path(gold_filename).stem

        retain_keys = ["uie_absa", "tokens", options.text_col_name]

        for key in retain_keys:
            assert key in entries[0], f"{key} should be but not in entries[0]!"

        if options.to_pickle:
            out_path = os.path.join(save_dir, out_stem + ".pkl")
            pickle.dump(entries, open(out_path, "wb"))
        else:
            out_path = os.path.join(save_dir, out_stem + ".json")
            write_jsonl_file(entries, out_path)
        logger.info(f"Saving results to {out_path}")

        if options.eval:
            results = out["results"]
            gold_filename_stem = Path(gold_filename).stem
            final_path = os.path.join(save_dir, gold_filename_stem + "_results.json")
            logger.info(f"Saving stats to {final_path}")
            with open(final_path, "w") as output:
                for key, value in results.items():
                    output.write(f"{key}={value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "-d",
        default=DATA_FOLDER,
        help="a folder containing json/jsonl files; each entry in each file contains 'text' and 'tokens' fields",
    )
    parser.add_argument(
        "--schema",
        "-s",
        default=SCHEMA_FOLDER,
        help="a folder containing 4 *.schema files; required by UIE architecture. We use `absa` task so simply copy \
                        the schema files from `text2spotasoc/absa/14lap` to this folder is enough",
    )
    parser.add_argument("--model", "-m", default=UIE_PATH)
    parser.add_argument("--log_file_path", "-l", default="uie.log")
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Whether the data contains gold labels or not; if yes, we will \
                        evaluate the model performance on the data and generate scores",
    )
    parser.add_argument("--text_col_name", "-t", default="texts")
    parser.add_argument("--to_pickle", action="store_true")
    parser.add_argument("--max_source_length", default=256, type=int)
    parser.add_argument("--max_target_length", default=192, type=int)
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    parser.add_argument(
        "-c",
        "--config",
        dest="map_config",
        help="Offset Re-mapping Config",
        default=CONFIG_MAP,
    )
    parser.add_argument("--decoding", default="spotasoc")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--match_mode", default="normal", choices=["set", "normal", "multimatch"]
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=".",
        help="a folder to save the output files, note that it should not be the same as the data folder",
    )
    options = parser.parse_args()
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer

    save_dir = options.output_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = create_dual_logger(options.log_file_path)
    main(options)
