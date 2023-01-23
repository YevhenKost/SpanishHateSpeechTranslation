import logging
import json
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from data_loading import FoldDataLoader
from typing import Dict, Any
import argparse
import os
parser.add_argument('--folds_path_dir', type=str, default="DataSplits/",
                        help='path to dir, which contains directories with folds data (e.g. DataSplits -> Fold0, Fold1, ...), Fold0 -> train.json, test.json' )
parser.add_argument('--save_dir', type=str, default="t5-baseline",
                        help="dir to save results")

args = parser.parse_args()
train_folds(args)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
def train_folds(args: argparse.Namespace) -> Dict[str, float]:

    for fold_dirname in os.listdir(args.folds_path_dir):

        fold_save_dir = os.path.join(args.save_dir, fold_dirname)

        data_loaded_dict = FoldDataLoader.load_train_eval_dfs(
            fold_dir_path=os.path.join(args.folds_path_dir, fold_dirname),
            prefix=PREFIX
        )


train_df=data_loaded_dict["train_df"]
eval_df=data_loaded_dict["eval_df"]

# train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
# eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

train_df["prefix"] = ""
eval_df["prefix"] = ""

model_args = T5Args()
model_args.max_seq_length = 96
model_args.train_batch_size = 20
model_args.eval_batch_size = 20
model_args.num_train_epochs = 1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
model_args.num_return_sequences = 1
model_args.wandb_project = "MT5 Sinhala-English Translation"

model = T5Model("mt5", "google/mt5-base", args=model_args)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Optional: Evaluate the model. We'll test it properly anyway.
results = model.eval_model(eval_df, verbose=True)