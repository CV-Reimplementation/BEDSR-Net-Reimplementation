# BEDSR-Net

This repository is unofficial implementation of [BEDSR-Net: A Deep Shadow Removal Network From a Single Document Image](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_BEDSR-Net_A_Deep_Shadow_Removal_Network_From_a_Single_Document_CVPR_2020_paper.html) [Lin+, CVPR 2020] with PyTorch.

A refined version of [IsHYuhi's implementation](https://github.com/IsHYuhi/BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image).

## Fix several problems
1. nn.ConvTranspose2d compatible with higher version of Pytorch
2. gradcam uses too much vram
3. provide default correct training config 
4. provide easy inference code

## create environment
```bash
conda create --name bedsrnet --file requirements.txt
```

## Training

Training BE-Net
```python
python3 train_benet.py ./configs/model\=benet/config.yaml
```

Training BEDSR-Net
```python
python3 train_bedsrnet.py ./configs/model\=bedsrnet/config.yaml
```

You can use W&B by ```--use_wandb```.

## Infer

mask sure put all your model state_dict into pretrained directory

```python
python infer.py
```

result images will be produced in results folder
