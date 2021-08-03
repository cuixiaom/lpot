ROOT=~/
MODEL_DIR=$ROOT/Data/transformer/mlperf_transformer
PARAM_SET=big

export OMP_NUM_THREADS=28

numactl --cpunodebind=0 --membind=0 \
python main.py --input_graph=$MODEL_DIR/graph/fp32/mlperf_transformer_fp32.pb --inputs_file=$MODEL_DIR/data/newstest2014_short.en --reference_file=$MODEL_DIR/data/newstest2014_short.de --vocab_file=$MODEL_DIR/data/vocab.ende.32768 --output_model=$MODEL_DIR/graph/fp32/mlperf_transformer_int8_mlp07.pb --config=./transformer_lt.yaml --tune

exit 0

#transformer_lt_official
numactl --cpunodebind=0 --membind=0 \
python main.py --input_graph=$ROOT/model/fp32_graphdef.pb --inputs_file=$ROOT/data/newstest2014.en --reference_file=$ROOT/data/newstest2014.de --vocab_file=$ROOT/data/vocab.txt --output_model=$ROOT/model/int8_graphdef.pb --config=./transformer_lt.yaml --tune
