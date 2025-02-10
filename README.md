# ComfyUI_FramePainter
Official pytorch implementation of "[FramePainter](https://github.com/YBYBZhang/FramePainter): Endowing Interactive Image Editing with Video Diffusion Priors",you can use it in comfyUI

# Update
* use single checkpoint now 改成SVD单体模型加载方式
* now 8G VRAM can run 512*512  峰值显存7G多，按理8G也能跑512了  


# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FramePainter.git
```
---

# 2. Requirements  
* no need, because it's normal for comfyUI ,Perhaps someone may be missing the library.没什么特殊的库,懒得删了
```
pip install -r requirements.txt
```

# 3.Model
* 3.1 download  checkpoints  from [here](https://huggingface.co/Yabo/FramePainter/tree/main) 从抱脸下载必须的模型,文件结构如下图
```
--  ComfyUI/models/FramePainter/
    |-- unet_diffusion_pytorch_model.safetensors
    |-- encoder_diffusion_pytorch_model.safetensors
```
*  3.2 SVD checkpoints  [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)    

```
--   ComfyUI/models/checkpoints
    ├── svd_xt.safetensors  or  svd_xt_1_1.safetensors
```


# 4.Example
![](https://github.com/smthemex/ComfyUI_FramePainter/blob/main/example.png)

# 5.Citation
[FramePainter](https://github.com/YBYBZhang/FramePainter)

* diffusers
```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
* controlnext
```
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
``
