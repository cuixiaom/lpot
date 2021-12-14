ROOT=~
ROOT=/localdisk/cuixiaom

# only valid for intel-tensorflow 2.5.0
export TF_ENABLE_MKL_NATIVE_FORMAT=0

# for TF2.6
#export TF_ENABLE_ONEDNN_OPTS=1

export OMP_NUM_THREADS=28

function quantize_transformer_lt() {
  MODEL_DIR=$ROOT/Data/transformer_lt
  VOCAB_FILE=$MODEL_DIR/data/vocab.txt
  INPUT_FILE=$MODEL_DIR/data/newstest2014_256sent
  PARAM_SET=big
  FP32_PBFILE=$MODEL_DIR/graph/fp32_graphdef.pb
  INT8_PBFILE=$MODEL_DIR/graph/int8_graphdef.pb
  python3 main.py --input_graph=$FP32_PBFILE --inputs_file=$INPUT_FILE.en --reference_file=$INPUT_FILE.de --vocab_file=$VOCAB_FILE --output_model=$INT8_PBFILE --config=./transformer_lt.yaml --tune
}

function quantize_transformer_mlperf() {
  MODEL_DIR=$ROOT/Data/transformer/mlperf_transformer
  INPUT_FILE=$MODEL_DIR/data/newstest2014_short
  VOCAB_FILE=$MODEL_DIR/data/vocab.ende.32768
  PARAM_SET=big
  FP32_PBFILE=$MODEL_DIR/graph/fp32/mlperf_transformer_fp32.pb
  INT8_PBFILE=$MODEL_DIR/graph/fp32/mlperf_transformer_int8_nativeformat.pb
  INT8_PBFILE=$MODEL_DIR/graph/fp32/mlperf_transformer_int8_test.pb
  python3 main_mlperf.py --input_graph=$FP32_PBFILE --inputs_file=$INPUT_FILE.en --reference_file=$INPUT_FILE.de --vocab_file=$VOCAB_FILE --output_model=$INT8_PBFILE --config=./transformer_lt.yaml --tune
}

cd ./examples/tensorflow/nlp/transformer_lt

#quantize_transformer_lt

quantize_transformer_mlperf

cd -

