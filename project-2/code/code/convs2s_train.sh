
python fairseq/train.py "data/Fairseq/Productivity/Tasks/Test1/prod1_train_12tasks" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Productivity/Tasks/Test1/prod1_train_12tasks" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Productivity/Tasks/Test1/prod1_train_12tasks" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/Productivity/Whens/Test5/prod5_train_024whens" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Productivity/Whens/Test5/prod5_train_024whens" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Productivity/Whens/Test5/prod5_train_024whens" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/Productivity/Whens/Test3/prod3_train_012whens" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Productivity/Whens/Test3/prod3_train_012whens" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Productivity/Whens/Test3/prod3_train_012whens" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/All/train30k_all" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/All/train30k_all" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/All/train30k_all" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/Productivity/Whens/Test4/prod4_train_0123whens" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Productivity/Whens/Test4/prod4_train_0123whens" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Productivity/Whens/Test4/prod4_train_0123whens" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/Systematicity/Test1/syst1_train_exc012" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Systematicity/Test1/syst1_train_exc012" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Systematicity/Test1/syst1_train_exc012" \
                                      --batch-size 64 --seed 0

python fairseq/train.py "data/Fairseq/Productivity/Tasks/Test2/prod2_train_13tasks" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "models/convs2s/run_0/Productivity/Tasks/Test2/prod2_train_13tasks" \
                                      --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                      --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                      --tensorboard-logdir "logs/convs2s/run_0/Productivity/Tasks/Test2/prod2_train_13tasks" \
                                      --batch-size 64 --seed 0
