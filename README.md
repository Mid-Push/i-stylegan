# Official pytorch implementation of i-stylegan

## [Multi-domain image generation and translation with identifiability guarantees](https://openreview.net/pdf?id=U2g8OGONA_V)

We build our code on the StyleGAN2-ADA, please go to https://github.com/NVlabs/stylegan2-ada-pytorch for environment setup.
(You can still run the code if you don't build NVIDIA conv ops, though it might be slower)

## Usage

### Step 0. clone the project.
```
git clone https://github.com/Mid-Push/i-stylegan.git
cd i-stylegan
```

### Step 1. prepare the dataset.

We have modified the dataset_tools.py so it accepts the multi-domain data with different directories. 


Download the AFHQ or CELEBA-HQ dataset from https://github.com/clovaai/stargan-v2
```
python dataset_tool.py --source=datasets/afhq/train --dest=datasets/afhq.zip --height=256 --width=256
```

### Step2. Training for Image Generation
```
python train.py --outdir=training-runs --data=datasets/afhq.zip --gpus=4 --cond_mode=flow --flow_norm=1 --i_dim=256 --lambda_sparse=0.1 
```

### Step3. Image Translation (Optional)
For image-to-image translation, after training the AFHQ stylegan model, 
put your i-stylegan checkpoint into stylegan_dir.
```
cd stargan
python main.py --mode train --num_domains 3 --w_hpf 0 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --batch_size=16\
               --lambda_pair=0.1\
               --tag=Run\
               --train_img_dir ../../datasets/afhq/train \
               --val_img_dir ../../datasets/afhq/val \
              --stylegan_dir expr/stylegan/afhq
```









