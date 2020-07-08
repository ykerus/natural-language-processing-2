
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Systematicity/Test1/syst1_train_exc012" -save_model "models/transformer/run_2/Systematicity/Test1/syst1_train_exc012" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Systematicity/Test1/syst1_train_exc012"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/All/train30k_all" -save_model "models/transformer/run_0/All/train30k_all" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/All/train30k_all"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Tasks/Test2/prod2_train_13tasks" -save_model "models/transformer/run_1/Productivity/Tasks/Test2/prod2_train_13tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Productivity/Tasks/Test2/prod2_train_13tasks"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Tasks/Test1/prod1_train_12tasks" -save_model "models/transformer/run_2/Productivity/Tasks/Test1/prod1_train_12tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Productivity/Tasks/Test1/prod1_train_12tasks"
  
#python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Productivity/Whens/Test3/prod3_train_012whens" -save_model "models/transformer/run_0/Productivity/Whens/Test3/prod3_train_012whens" \
#                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
#                            -encoder_type transformer -decoder_type transformer -position_encoding \
#                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
#                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
#                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Productivity/Whens/Test3/prod3_train_012whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Productivity/Tasks/Test1/prod1_train_12tasks" -save_model "models/transformer/run_0/Productivity/Tasks/Test1/prod1_train_12tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Productivity/Tasks/Test1/prod1_train_12tasks"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test4/prod4_train_0123whens" -save_model "models/transformer/run_1/Productivity/Whens/Test4/prod4_train_0123whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Productivity/Whens/Test4/prod4_train_0123whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Systematicity/Test1/syst1_train_exc012" -save_model "models/transformer/run_0/Systematicity/Test1/syst1_train_exc012" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Systematicity/Test1/syst1_train_exc012"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Tasks/Test2/prod2_train_13tasks" -save_model "models/transformer/run_2/Productivity/Tasks/Test2/prod2_train_13tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Productivity/Tasks/Test2/prod2_train_13tasks"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test4/prod4_train_0123whens" -save_model "models/transformer/run_2/Productivity/Whens/Test4/prod4_train_0123whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Productivity/Whens/Test4/prod4_train_0123whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test5/prod5_train_024whens" -save_model "models/transformer/run_1/Productivity/Whens/Test5/prod5_train_024whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Productivity/Whens/Test5/prod5_train_024whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test5/prod5_train_024whens" -save_model "models/transformer/run_2/Productivity/Whens/Test5/prod5_train_024whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Productivity/Whens/Test5/prod5_train_024whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Productivity/Whens/Test5/prod5_train_024whens" -save_model "models/transformer/run_0/Productivity/Whens/Test5/prod5_train_024whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Productivity/Whens/Test5/prod5_train_024whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test3/prod3_train_012whens" -save_model "models/transformer/run_1/Productivity/Whens/Test3/prod3_train_012whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Productivity/Whens/Test3/prod3_train_012whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Systematicity/Test1/syst1_train_exc012" -save_model "models/transformer/run_1/Systematicity/Test1/syst1_train_exc012" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Systematicity/Test1/syst1_train_exc012"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Tasks/Test1/prod1_train_12tasks" -save_model "models/transformer/run_1/Productivity/Tasks/Test1/prod1_train_12tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/Productivity/Tasks/Test1/prod1_train_12tasks"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/All/train30k_all" -save_model "models/transformer/run_1/All/train30k_all" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 1 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_1/All/train30k_all"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Productivity/Tasks/Test2/prod2_train_13tasks" -save_model "models/transformer/run_0/Productivity/Tasks/Test2/prod2_train_13tasks" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Productivity/Tasks/Test2/prod2_train_13tasks"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_0/Productivity/Whens/Test4/prod4_train_0123whens" -save_model "models/transformer/run_0/Productivity/Whens/Test4/prod4_train_0123whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 0 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_0/Productivity/Whens/Test4/prod4_train_0123whens"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/All/train30k_all" -save_model "models/transformer/run_2/All/train30k_all" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/All/train30k_all"
  
python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test3/prod3_train_012whens" -save_model "models/transformer/run_2/Productivity/Whens/Test3/prod3_train_012whens" \
                            -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                            -encoder_type transformer -decoder_type transformer -position_encoding \
                            -optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -max_grad_norm 0 \
                            -param_init 0 -param_init_glorot -batch_size 64 -max_generator_batches 2 \
                            -valid_steps 2500 -train_steps 25000 -save_checkpoint_steps 2500 -warmup_steps 5000                             -learning_rate 1 -dropout 0.1 -gpu_rank 0 -seed 2 \
                            -tensorboard -tensorboard_log_dir "logs/transformer/run_2/Productivity/Whens/Test3/prod3_train_012whens"
  