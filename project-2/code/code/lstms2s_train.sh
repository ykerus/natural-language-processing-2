
python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/All/train30k_all" -save_model "models/lstms2s/run_1/All/train30k_all" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/All/train30k_all"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Tasks/Test1/prod1_train_12tasks" -save_model "models/lstms2s/run_1/Productivity/Tasks/Test1/prod1_train_12tasks" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Productivity/Tasks/Test1/prod1_train_12tasks"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test4/prod4_train_0123whens" -save_model "models/lstms2s/run_1/Productivity/Whens/Test4/prod4_train_0123whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Productivity/Whens/Test4/prod4_train_0123whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test5/prod5_train_024whens" -save_model "models/lstms2s/run_1/Productivity/Whens/Test5/prod5_train_024whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Productivity/Whens/Test5/prod5_train_024whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Tasks/Test2/prod2_train_13tasks" -save_model "models/lstms2s/run_1/Productivity/Tasks/Test2/prod2_train_13tasks" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Productivity/Tasks/Test2/prod2_train_13tasks"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Systematicity/Test1/syst1_train_exc012" -save_model "models/lstms2s/run_2/Systematicity/Test1/syst1_train_exc012" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Systematicity/Test1/syst1_train_exc012"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Systematicity/Test1/syst1_train_exc012" -save_model "models/lstms2s/run_1/Systematicity/Test1/syst1_train_exc012" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Systematicity/Test1/syst1_train_exc012"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Tasks/Test1/prod1_train_12tasks" -save_model "models/lstms2s/run_2/Productivity/Tasks/Test1/prod1_train_12tasks" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Productivity/Tasks/Test1/prod1_train_12tasks"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test3/prod3_train_012whens" -save_model "models/lstms2s/run_2/Productivity/Whens/Test3/prod3_train_012whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Productivity/Whens/Test3/prod3_train_012whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test4/prod4_train_0123whens" -save_model "models/lstms2s/run_2/Productivity/Whens/Test4/prod4_train_0123whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Productivity/Whens/Test4/prod4_train_0123whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_1/Productivity/Whens/Test3/prod3_train_012whens" -save_model "models/lstms2s/run_1/Productivity/Whens/Test3/prod3_train_012whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 1 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_1/Productivity/Whens/Test3/prod3_train_012whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Tasks/Test2/prod2_train_13tasks" -save_model "models/lstms2s/run_2/Productivity/Tasks/Test2/prod2_train_13tasks" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Productivity/Tasks/Test2/prod2_train_13tasks"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/Productivity/Whens/Test5/prod5_train_024whens" -save_model "models/lstms2s/run_2/Productivity/Whens/Test5/prod5_train_024whens" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/Productivity/Whens/Test5/prod5_train_024whens"

python OpenNMT-py/train.py  -data "data/OpenNMT/run_2/All/train30k_all" -save_model "models/lstms2s/run_2/All/train30k_all" \
                            -train_steps 25000 -layers 2 -rnn_size 512 -word_vec_size 512 \
                            -rnn_type LSTM --encoder_type brnn \
                            -batch_size 64 -gpu_rank 0 -seed 2 \
                            -learning_rate 0.1 -optim sgd -global_attention general \
                            -tensorboard -tensorboard_log_dir "logs/lstms2s/run_2/All/train30k_all"
