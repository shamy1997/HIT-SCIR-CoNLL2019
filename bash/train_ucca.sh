CUDA_VISIBLE_DEVICES=0
TRAIN_PATH=data/ucca/temp/ucca_100_train.aug.mrp \
DEV_PATH=data/ucca/temp/ucca_20_dev.aug.mrp \
BERT_PATH=data/pretrained/bert-large-cased-wwm \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/ucca_bert_wmm_test \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_ucca.jsonnet