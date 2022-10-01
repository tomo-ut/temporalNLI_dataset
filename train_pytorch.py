# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on HANS."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from train_pytorch_util import HansDataset, InputFeatures, hans_processors, hans_tasks_num_labels
from sklearn.metrics import classification_report


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(hans_processors.keys())}
    )
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    train_data_name: str = field(
        metadata={"help": "The name of the file used as the train dataset"}
    )
    eval_data_name: str = field(
        metadata={"help": "The name of the file used as the eval dataset"}
    )
    result_file_name: str = field(
        metadata={"help": "The name of the output file"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def hans_data_collator(features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
    """
    Data collator that removes the "pairID" key if present.
    """
    batch = default_data_collator(features)
    _ = batch.pop("pairID", None)
    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use"
            " --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = hans_tasks_num_labels[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        HansDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            train_data_name=data_args.train_data_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        HansDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            eval_data_name=data_args.eval_data_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            evaluate=True,
        )
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=hans_data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output = trainer.predict(eval_dataset)
        preds = output.predictions
        preds = np.argmax(preds, axis=1)

        pair_ids = [ex.pairID for ex in eval_dataset]
        output_eval_file = os.path.join(training_args.output_dir, data_args.result_file_name)
        label_list = eval_dataset.get_labels()
        y_true = []
        y_pred = []
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                writer.write("pairID,pred_label,gold_label\n")
                for i, (pid, pred) in enumerate(zip(pair_ids, preds)):
                    gold_label = label_list[eval_dataset.features[i].label]
                    pred_label = label_list[int(pred)]
                    y_true.append(gold_label)
                    y_pred.append(pred_label)
                    writer.write("\t".join(["ex" + str(pid), pred_label,
                                 gold_label, str(pred_label == gold_label)]) + "\n")

                for k, v in output.metrics.items():
                    writer.write(str(k) + "\t" + str(v) + "\n")
        classify_result = classification_report(y_true, y_pred, digits=4, output_dict=True)

        with open('results/result.txt', 'a') as outfile:
            ind = ['precision', 'recall', 'f1-score', 'support']
            result1 = [
                '',
                'roberta',
                'pytorch_model.bin',
                '',
                data_args.train_data_name,
                data_args.eval_data_name,
                42,
                training_args.per_device_train_batch_size,
                training_args.learning_rate]
            result2 = [classify_result['accuracy'], classify_result['macro avg']['support']]
            result3 = [classify_result['macro avg'][name]for name in ind[:3]]
            result4 = [classify_result['weighted avg'][name]for name in ind[:3]]
            result5 = [classify_result['entailment'][name]for name in ind]
            result6 = [classify_result['contradiction'][name]for name in ind]
            result7 = [classify_result['neutral'][name]for name in ind]
            result = result1 + result2 + result3 + result4 + result5 + result6 + result7
            result = [str(r) for r in result]
            r_str = "\t".join(result)
            outfile.write(f'{r_str}\n')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
