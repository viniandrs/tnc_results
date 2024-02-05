#/bin/bash

cd ..

./tnc.py fit \
    --data ./../../../data/view_concatenated/MotionSense \
    --epochs 100 \
    --batch_size 16 \
    --accelerator gpu \
    --devices 1 \
    --training_mode pretrain \
    --checkpoint_metric train_loss \
    --repeat 5 \
    --mc_sample_size 20 \
    --window_size 60 \
    --encoding_size 10 \
    --w 0.05 \