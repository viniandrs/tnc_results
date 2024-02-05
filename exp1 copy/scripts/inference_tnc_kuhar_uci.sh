#/bin/bash

cd ..

#for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI"; 
for dset in  "KuHar" "MotionSense" "UCI"; 
do 
    ./tnc.py test \
        --data ./../../../data/standartized_balanced/${dset} \
        --load logs/finetune/TNC/kuhar_uci/checkpoints/last.ckpt \
        --batch_size 128 \
        --accelerator gpu \
        --devices 1 \
        --mc_sample_size 20 \
        --window_size 60 \
        --encoding_size 10 \
        --w 0.05
done