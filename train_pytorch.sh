export PYTORCH_ENABLE_MPS_FALLBACK=1
export HANS_DIR=dataset
export MODEL_TYPE=roberta
export MODEL_PATH=nlp-waseda/roberta-large-japanese
# export MODEL_PATH=./models_roberta/checkpoint-1090
# export TOKENIZER_PATH=./models_roberta_mini
export OUTPUT_DIR=./models_roberta
export TRAIN_DATA=train_wakati.tsv
export EVAL_DATA=test_wakati.tsv
export RESULT_FILE=predictions/predictions_lr2e5.txt

python train_pytorch.py \
        --task_name hans \
        --data_dir $HANS_DIR \
        --model_name_or_path $MODEL_PATH \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --use_mps_device \
        --per_device_train_batch_size 16 \
        --train_data_name $TRAIN_DATA \
        --eval_data_name $EVAL_DATA \
        --result_file_name $RESULT_FILE \
        --do_eval \
        --save_strategy epoch \
        --overwrite_cache \
        --per_device_eval_batch_size 8 \
        --do_train \
        --learning_rate 2e-5 \
        # --tokenizer_name $TOKENIZER_PATH \
        # --auto_find_batch_size \
        # --model_type $MODEL_TYPE \