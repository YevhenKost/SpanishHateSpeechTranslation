import json

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from data_loading import FoldDataLoader
import os
from metrics_utils import calculate_metrics

from config.mbart_config import PREFIX, load_model_args, MODEL_TYPE, MODEL_NAME

from typing import Dict, Any
import argparse


def train_folds(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    for fold_dirname in os.listdir(args.folds_path_dir):
        print(fold_dirname)
        if fold_dirname in os.listdir(args.save_dir):
            print("Already processed")
        else:
            print("Start processing")
            fold_save_dir = os.path.join(args.save_dir, fold_dirname)

            data_loaded_dict = FoldDataLoader.load_train_eval_dfs(
                fold_dir_path=os.path.join(args.folds_path_dir, fold_dirname),
                prefix=PREFIX
            )

            model_args = load_model_args()
            model_args.output_dir = fold_save_dir

            model = Seq2SeqModel(MODEL_TYPE, MODEL_NAME, args=model_args,
                                 encoder_decoder_type='bart', encoder_decoder_name='facebook/bart-large', use_cuda=args.use_cuda)

            model.train_model(data_loaded_dict["train_df"], eval_data=data_loaded_dict["eval_df"])

            train_loss = model.eval_model(data_loaded_dict["eval_df"])["eval_loss"]
            eval_loss = model.eval_model(data_loaded_dict["train_df"])["eval_loss"]

            preds = model.predict(data_loaded_dict["eval_df_refs"]["input_text"].values.tolist())

            metrics = calculate_metrics(
                predicted_texts=preds, references=data_loaded_dict["eval_df_refs"]["target_text"].values.tolist()
            )
            metrics["train_loss"] = train_loss
            metrics["eval_loss"] = eval_loss

            # saving metrics to a save_dir
            os.makedirs(fold_save_dir, exist_ok=True)
            with open(
                    os.path.join(fold_save_dir, "metrics.json"), "w"
            ) as f:
                json.dump(metrics, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Baseline Training')

    parser.add_argument('--folds_path_dir', type=str, default="DataSplits/",
                        help='path to dir, which contains directories with folds data (e.g. DataSplits -> Fold0, Fold1, ...), Fold0 -> train.json, test.json' )
    parser.add_argument('--save_dir', type=str, default="t5-baseline",
                        help="dir to save results")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help="whether to train on GPU")


    args = parser.parse_args()
    train_folds(args)