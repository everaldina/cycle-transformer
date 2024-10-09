#!/bin/bash
cd /workspace/repositorio
echo "-----------------" > /workspace/data/output.log
python train.py --dataroot /workspace/data/data_list/train_list.pkl --epoch 48 --epoch_count 49 --continue_train --name teste &> /workspace/data/output.log