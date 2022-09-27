export PYTORCH_ENABLE_MPS_FALLBACK=1
export HANS_DIR=dataset
export MODEL_TYPE=roberta
export MODEL_PATH=nlp-waseda/roberta-large-japanese
# export MODEL_PATH=./models_roberta/checkpoint-1090
# export TOKENIZER_PATH=./models_roberta2
export OUTPUT_DIR=./models_roberta2
export TRAIN_DATA=train_wakati.tsv
export EVAL_DATA=test_wakati.tsv
export RESULT_FILE=predictions/predictions.txt

python train_roberta.py \
        --task_name hans \
        --data_dir $HANS_DIR \
        --model_name_or_path $MODEL_PATH \
        --tokenizer_name $TOKENIZER_PATH \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --use_mps_device \
        --per_device_train_batch_size 1 \
        --train_data_name $TRAIN_DATA \
        --eval_data_name $EVAL_DATA \
        --result_file_name $RESULT_FILE \
        --do_eval \
        --save_strategy epoch \
        --overwrite_cache \
        --per_device_eval_batch_size 1 \
        --do_train \
        # --auto_find_batch_size \
        # --model_type $MODEL_TYPE \