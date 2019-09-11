# Encoder-Agnostic Adaptation for Conditional Language Generation

This repo contains the code used in [Encoder-Agnostic Adaptation for Conditional Language Generation](https://arxiv.org/abs/1908.06938), Zachary M. Ziegler, Luke Melas-Kyriazi, Sebastian Gehrmann and Alexander M. Rush. It extends [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

This code was tested with `pytorch 1.0.1`. See requirements.txt for a complete list of dependencies.

## Download GPT2 weights

`cd gpt2 && python download_model.py 124M`

## General notes

All experiments use gradient accumulation to mimic the large batch sizes these hyperparameter settings were optimized for by e.g. Facebook. If you run into GPU memory issues simply reduce the batch size and increase the `accum_count` to keep the effective batch size the same.

## Data

The BPEized data used in the experiments in the paper can be found [here](https://drive.google.com/file/d/1Z6AdOr2MtWlN7sYRTMibzAcghBjSBzZK/view?usp=sharing). To run any of these models with your own data you should first BPEize it with `python gpt2/encode_text.py <filename>`. Before training the raw data is preprocessed into binary data shards with the commands below.

## Class-conditional generation

### Preprocess

`python preprocess.py -train_src data/imdb/train.src.bpe -train_tgt data/imdb/train.tgt.bpe -valid_src data/imdb/valid.src.bpe -valid_tgt data/imdb/valid.tgt.bpe -save_data data/imdb/IMDB_BPETGT -tgt_seq_length_trunc 400 -tgt_vocab gpt2/vocab.txt -fixed_vocab -free_src`

### Train
**Baseline**: `python train.py -config config/imdb/transformer_imdb_cond.yml -run_name baseline`

**Simple fusion**: `python train.py -config config/imdb/transformer_imdb_cond.yml -run_name simple_fusion -gpt2_params_path gpt2/models/124M/ -simple_fusion -dropout 0.1 -accum_count 30 -batch_size 1000 -valid_batch_size 16`

**Repr-transformer**: `python train.py -config config/imdb/transformer_imdb_cond.yml -run_name repr_trans -GPT_representation_loc tgt -GPT_representation_mode elmo -gpt2_params_path gpt2/models/124M/ -position_encoding_learned_dec -word_vec_size 768 -rnn_size 768`

**Context attention**: `python train.py -config config/imdb/transformer_imdb_ctxattn.yml -run_name ctxattn -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

**Pseudo self attention**: `python train.py -config config/imdb/transformer_imdb_psa.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Generation

Generation is performed via random sampling.

`python translate.py -beam_size 1 -random_sampling_topk -1 -random_sampling_temp 0.7 -model <path/to/model.pt> -src data/imdb/test.src.bpe -min_length 1 -max_length 400 -verbose`

## Summarization

### Preprocess

`python preprocess.py -train_src data/cnndm/train.txt.src.bpe -train_tgt data/cnndm/train.txt.tgt.bpe -valid_src data/cnndm/val.txt.src.bpe -valid_tgt data/cnndm/val.txt.tgt.bpe -save_data data/cnndm/CNNDM_BPE_COPY -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -src_vocab gpt2/vocab.txt -tgt_vocab gpt2/vocab.txt -dynamic_dict -fixed_vocab`

### Train
The default settings use 4 GPUs (see config files). If using more GPUs or fewer GPUs, modify the `world_size` and `gpu_ranks` values in the config file and adjust `accum_count` so the effective batch size remains the same.

**Baseline**: `python train.py -config config/cnndm/transformer_cnndm_baseline.yml -run_name baseline`

**Repr-transformer**: `python train.py -config config/cnndm/transformer_cnndm_baseline.yml -run_name repr_trans  -GPT_representation_loc tgt -GPT_representation_mode elmo -gpt2_params_path gpt2/models/124M/ -position_encoding_learned_dec -word_vec_size 768 -rnn_size 768 -train_steps 50000`

**Context attention**: `python train.py -config config/cnndm/transformer_cnndm_ctxattn.yml -run_name ctxattn -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

**Pseudo self attention**: `python train.py -config config/cnndm/transformer_cnndm_psa.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Generation

Generation is performed via beam search.

`python translate.py -beam_size 5 -model <path/to/model.pt> -src data/cnndm/test.txt.src.bpe -min_length 60 -verbose -block_ngram_repeat 3`

## Story generation
The default settings use 4 GPUs (see config files). If using more GPUs or fewer GPUs, modify the `world_size` and `gpu_ranks` values in the config file and adjust `accum_count` so the effective batch size remains the same.

### Preprocess

`python preprocess.py -train_src data/stories/train.wp_source.bpe -train_tgt data/stories/train.wp_target.bpe -valid_src data/stories/valid.wp_source.bpe -valid_tgt data/stories/valid.wp_target.bpe -save_data data/stories/STORIES_BPE -src_vocab gpt2/vocab.txt -tgt_vocab gpt2/vocab.txt -fixed_vocab`

### Train
**Baseline**: `python train.py -config config/story_gen/transformer_story_baseline.yml -run_name baseline`

**Repr-transformer**: `python train.py -config config/story_gen/transformer_story_baseline.yml -run_name repr_trans -GPT_representation_loc tgt -GPT_representation_mode elmo -gpt2_params_path gpt2/models/124M/ -position_encoding_learned_dec -word_vec_size 768 -rnn_size 768`

**Context attention**: `python train.py -config config/story_gen/transformer_story_ctxattn.yml -run_name ctxattn -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

**Pseudo self attention**: `python train.py -config config/story_gen/transformer_story_psa.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Generation

Generation is performed via top-k/random sampling.

`python translate.py -beam_size 1 -random_sampling_topk 100 -random_sampling_temp 0.9 -model <path/to/model.pt> -src data/stories/test.wp_source.bpe -max_length 1000 -verbose`

## Image captioning

Coming soon...
