# ComfyUI_FramePainter
Official pytorch implementation of "[FramePainter](https://github.com/YBYBZhang/FramePainter): Endowing Interactive Image Editing with Video Diffusion Priors",you can use it in comfyUI


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
*  3.2 SVD repo [stabilityai/stable-video-diffusion-img2vid-xt
](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) online or offline   
* if offline
```
--   anypath/stable-video-diffusion-img2vid-xt/  # or stable-video-diffusion-img2vid-xt-1-1 
    ├── model_index.json
    ├── vae...
    ├── unet...
    ├── feature_extractor...
    ├── scheduler...
```
* 3.3 clip_vison
```
--  ComfyUI/models/clip_vision/
    ├── clip_vision_H.safetensors   # or 'stabilityai/stable-video-diffusion-img2vid-xt' image encoder safetensors or ipadapter image encoder
```


# 4.Example
![](https://github.com/smthemex/ComfyUI_FramePainter/blob/main/example1.png)

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
