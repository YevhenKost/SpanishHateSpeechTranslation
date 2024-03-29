from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


PREFIX = "paraphrase"
MODEL_TYPE = "mbart"
# encoder_decoder_type="mbart"
encoder_decoder_name="facebook/bart-large"
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

SEED = 2888

NUM_EPOCHS = 50
MAX_SEQ_LEN = 128

OPTIMIZER = "AdamW"
LR = 1e-5

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


def load_model_args() -> Seq2SeqArgs:
    model_args = Seq2SeqArgs(
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

