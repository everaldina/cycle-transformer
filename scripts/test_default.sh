cd /workspace/repositorio
echo "----------------------------------" >> /workspace/data/output.log
python test.py --dataroot /workspace/data --result_dir /workspace/repositorio/test --epoch 90 --name teste &>> /workspace/data/output.log