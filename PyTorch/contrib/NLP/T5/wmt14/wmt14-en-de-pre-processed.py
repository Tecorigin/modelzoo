# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" WMT16 English-Romanian Translation Data with further preprocessing """

from __future__ import absolute_import, division, print_function

import csv
import json
import os

import datasets

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {WMT14 English-German Translation Data with further preprocessing},
authors={},
year={2016}
}
"""

_DESCRIPTION = "WMT14 English-German Translation Data with further preprocessing"
_HOMEPAGE = "http://www.statmt.org/wmt16/"
_LICENSE = ""


# _DATA_URL = "https://cdn-datasets.huggingface.co/translation/wmt_en_de.tgz"
_DATA_URL = "wmt_en_de.tgz"


class Wmt14EnDePreProcessedConfig(datasets.BuilderConfig):
    """BuilderConfig for wmt16."""

    def __init__(self, language_pair=(None, None), **kwargs):
        """BuilderConfig for wmt16

        Args:
            for the `datasets.features.text.TextEncoder` used for the features feature.
          language_pair: pair of languages that will be used for translation. Should
            contain 2-letter coded strings. First will be used at source and second
            as target in supervised mode. For example: ("se", "en").
          **kwargs: keyword arguments forwarded to super.
        """
        name = "%s%s" % (language_pair[0], language_pair[1])

        description = ("Translation dataset from %s to %s") % (language_pair[0], language_pair[1])
        super(Wmt14EnDePreProcessedConfig, self).__init__(
            name=name,
            description=description,
            version=datasets.Version("1.1.0", ""),
            **kwargs,
        )

        # Validate language pair.
        assert "en" in language_pair, ("Config language pair must contain `en`, got: %s", language_pair)
        source, target = language_pair
        non_en = source if target == "en" else target
        assert non_en in ["de"], ("Invalid non-en language in pair: %s", non_en)

        self.language_pair = language_pair


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class Wmt14EnDePreProcessed(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        Wmt14EnDePreProcessedConfig(
            language_pair=("en", "de"),
        ),
    ]

    def _info(self):
        source, target = self.config.language_pair
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"translation": datasets.features.Translation(languages=self.config.language_pair)}
            ),
            supervised_keys=(source, target),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        source, target = self.config.language_pair
        non_en = source if target == "en" else target
        path_tmpl = "{dl_dir}/wmt_en_de/{split}.{type}"

        files = {}
        for split in ("train", "val", "test"):
            files[split] = {
                "source_file": path_tmpl.format(dl_dir=dl_dir, split=split, type="source"),
                "target_file": path_tmpl.format(dl_dir=dl_dir, split=split, type="target"),
            }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=files["train"]),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=files["val"]),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=files["test"]),
        ]

    def _generate_examples(self, source_file, target_file):
        """This function returns the examples in the raw (text) form."""
        with open(source_file, mode="rb") as f:
            source_sentences = f.read().decode("utf8").split("\n")
        with open(target_file, mode="rb") as f:
            target_sentences = f.read().decode("utf8").split("\n")

        assert len(target_sentences) == len(source_sentences), "Sizes do not match: %d vs %d for %s vs %s." % (
            len(source_sentences),
            len(target_sentences),
            source_file,
            target_file,
        )

        source, target = self.config.language_pair
        for idx, (l1, l2) in enumerate(zip(source_sentences, target_sentences)):
            result = {"translation": {source: l1, target: l2}}
            # Make sure that both translations are non-empty.
            if all(result.values()):
                yield idx, result
