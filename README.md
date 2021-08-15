# DeepLabv3Plus-Pytorch for Xview dataset

Just testing for now

### 2. Train your model on Xview
Put the rdata512 folder in the root of the repo after cloning then run this command:
```bash
python main.py --model deeplabv3plus_resnet50 --gpu_id 0 --crop_val --lr 0.01 --crop_size 513 --batch_size 2 --output_stride 16 --dataset xview --save_val_results
```

## Reference
Most of the code taken from https://github.com/VainF/DeepLabV3Plus-Pytorch
