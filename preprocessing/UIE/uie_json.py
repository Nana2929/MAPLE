# coding=utf-8

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import datasets
import pyarrow as pa
import pyarrow.json as paj


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    field: Optional[str] = None
    use_threads: bool = True
    block_size: Optional[int] = None
    newlines_in_values: Optional[bool] = None

    @property
    def pa_read_options(self):
        return paj.ReadOptions(use_threads=self.use_threads, block_size=self.block_size)

    @property
    def pa_parse_options(self):
        import pickle
        table_schema = pickle.load(open('etc/record.dataload.schema', 'rb'))
        print(table_schema)
        return paj.ParseOptions(explicit_schema=table_schema, newlines_in_values=self.newlines_in_values)

    @property
    def schema(self):
        return pa.schema(self.features.type) if self.features is not None else None


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_tables(self, files):
        for i, file in enumerate(files):
            if self.config.field is not None:
                with open(file, encoding="utf-8") as f:
                    dataset = json.load(f)

                # We keep only the field we are interested in
                dataset = dataset[self.config.field]

                # We accept two format: a list of dicts or a dict of lists
                if isinstance(dataset, (list, tuple)):
                    pa_table = paj.read_json(
                        BytesIO("\n".join(json.dumps(row) for row in dataset).encode("utf-8")),
                        read_options=self.config.pa_read_options,
                        parse_options=self.config.pa_parse_options,
                    )
                else:
                    pa_table = pa.Table.from_pydict(mapping=dataset)
            else:
                try:
                    pa_table = paj.read_json(
                        file,
                        read_options=self.config.pa_read_options,
                        parse_options=self.config.pa_parse_options,
                    )
                except pa.ArrowInvalid:
                    with open(file, encoding="utf-8") as f:
                        dataset = json.load(f)
                    raise ValueError(
                        f"Not able to read records in the JSON file at {file}. "
                        f"You should probably indicate the field of the JSON file containing your records. "
                        f"This JSON file contain the following fields: {str(list(dataset.keys()))}. "
                        f"Select the correct one and provide it as `field='XXX'` to the dataset loading method. "
                    )
            if self.config.features:
                # Encode column if ClassLabel
                for i, col in enumerate(self.config.features.keys()):
                    if isinstance(self.config.features[col], datasets.ClassLabel):
                        pa_table = pa_table.set_column(
                            i, self.config.schema.field(col), [self.config.features[col].str2int(pa_table[col])]
                        )
                # Cast allows str <-> int/float, while parse_option explicit_schema does NOT
                # Before casting, rearrange JSON field names to match passed features schema field names order
                pa_table = pa.Table.from_arrays(
                    [pa_table[name] for name in self.config.features], schema=self.config.schema
                )
            yield i, pa_table
