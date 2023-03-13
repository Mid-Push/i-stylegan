import pickle
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import PIL
import os

batch = 2
cur = 0
with open(sys.argv[1], 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    for i in range(20):
        images = []
        for cur in range(batch):
            z = torch.randn([1, G.z_dim]).cuda()    # latent codes
            num_domains = G.c_dim
            img = []
            for j in range(num_domains):
                c = F.one_hot((torch.ones([1])*j).long(),num_domains).cuda()                                # class labels (not used in this example)
                tmp_img = G(z, c=c, truncation_psi=0.7)                           # NCHW, float32, dynamic range [-1, +1]
                img.append(tmp_img)
            img = torch.cat(img, 0)
            img = (img*127.5+128).clamp(0,255)/255.
            images.append(img)
        images = torch.cat(images, 0)
        dire = 'results/transgan_faces'
        os.makedirs(dire, exist_ok=True)
        save_image(images, '%s/faces%d.jpg'%(dire, i), nrow=num_domains)
