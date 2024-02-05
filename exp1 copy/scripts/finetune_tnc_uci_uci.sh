#/bin/bash

cd ..

./tnc.py fit \
    --data ./../../../data/standartized_balanced/UCI \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone logs/pretrain/TNC/UCI/checkpoints/last.ckpt \
    --training_mode finetune \
    --repeat 5 \
    --mc_sample_size 20 \
    --window_size 60 \
    --encoding_size 10 \
    --w 0.05 \
    --update_backbone False

# renaming results folder
new_result_folder_name=uci_uci
mv $(find logs/finetune/TNC -name "2024*" -type d) logs/finetune/TNC/$new_result_folder_name

# updating symbolic link
cd logs/finetune/TNC/$new_result_folder_name/checkpoints
results_file_name=$(find -name "epoch*" -type "f" -printf "%f\n")
ln -vsf $results_file_name last.ckpt