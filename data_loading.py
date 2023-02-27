import pandas as pd
import json, os
from typing import List, Dict, Any


class FoldDataLoader:

    @classmethod
    def format_fold_data(cls, json_df: List[Dict[str, Any]], prefix: str, explode_label: bool = True) -> pd.DataFrame:

        df = pd.DataFrame(json_df)
        df = df.rename(columns={
            "data": "input_text",
            "label": "target_text"
        })
        df["prefix"] = [prefix for i in range(len(df))]

        if explode_label:
            df = df.explode(column="target_text")

        return df

    @classmethod
    def load_train_eval_dfs(cls, fold_dir_path: str, prefix: str) -> Dict[str, pd.DataFrame]:

        train_json_df = json.load(open(
            os.path.join(fold_dir_path, "train.json")
        ))
        test_json_df = json.load(open(
            os.path.join(fold_dir_path, "test.json")
        ))

        train_df = cls.format_fold_data(json_df=train_json_df, prefix=prefix, explode_label=True)
        test_df = cls.format_fold_data(json_df=test_json_df, prefix=prefix, explode_label=False)
        test_df_exploded = cls.format_fold_data(json_df=test_json_df, prefix=prefix, explode_label=True)


        return {
            "train_df": train_df,
            "eval_df": test_df_exploded,
            "eval_df_refs": test_df
        }