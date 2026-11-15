#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import torch
from torch.nn import functional as F
r=Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)
x=np.random.default_rng(4).normal(size=(1,8,2,3)).astype(np.float32)
# file is ggml C,W,H physical order
x.transpose(2,3,1,0).copy().tofile(r/'input.f32')
y=F.pixel_shuffle(torch.from_numpy(x),2);z=F.interpolate(y,size=(3,5),mode='bilinear',align_corners=False);p=F.adaptive_avg_pool2d(y,(3,5))
np.save(r/'shuffle.npy',y.numpy());np.save(r/'resize.npy',z.numpy());np.save(r/'pool.npy',p.numpy())
json.dump({'format_version':1,'tensors':{'pixel.shuffle':{'file':'shuffle.npy','dtype':'float32','shape':list(y.shape)},'pixel.resize':{'file':'resize.npy','dtype':'float32','shape':list(z.shape)},'pixel.pool':{'file':'pool.npy','dtype':'float32','shape':list(p.shape)}}},open(r/'manifest.json','w'))
