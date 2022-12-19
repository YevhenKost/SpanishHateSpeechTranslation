from simpletransformers.t5 import T5Args


PREFIX = "paraphrase"
MODEL_TYPE = "t5"
MODEL_NAME = "t5-small"

SEED = 2

NUM_EPOCHS = 4
MAX_SEQ_LEN = 128

OPTIMIZER = "AdamW"
LR = 1e-5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64


def load_model_args() -> T5Args:
    model_args = T5Args(
        max_seq_length=MAX_SEQ_LEN,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        train_batch_size=TRAIN_BATCH_SIZE,
        # output_dir="/",
        overwrite_output_dir=True,
        optimizer=OPTIMIZER,
        eval_batch_size=EVAL_BATCH_SIZE,
        manual_seed=SEED,
        save_model_every_epoch=False,
        evaluate_during_training=False,
        save_steps=-1,
        no_cache=True,
        no_save=True
    )

    return model_args

