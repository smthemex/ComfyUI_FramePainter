# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from safetensors.torch import load_file
from .modules.pipelines.pipeline_framepainter import FramePainterPipeline
from .modules.sparse_control_encoder import SparseControlEncoder
from .modules.unet_spatio_temporal_condition_edit import UNetSpatioTemporalConditionEdit
from .modules.attention_processors import MatchingAttnProcessor2_0
from .modules.utils.attention_utils import set_matching_attention, set_matching_attention_processor
from .node_utils import  process_image_with_mask,timer,pil2narry,tensor_upscale,convert_cf2diffuser
import folder_paths


MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
FramePainter_weigths_path = os.path.join(folder_paths.models_dir, "FramePainter")
if not os.path.exists(FramePainter_weigths_path):
    os.makedirs(FramePainter_weigths_path)
folder_paths.add_model_folder_path("FramePainter", FramePainter_weigths_path)



class FramePainter_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        FramePainter_unet_list = [i for i in folder_paths.get_filename_list("FramePainter") if
                                 "unet" in i]
        sparse_control_ckpt_list = [i for i in folder_paths.get_filename_list("FramePainter") if
                                 "encoder" in i]
        return {
            "required": {
                "model": ("MODEL",),
                "FramePainter_unet": (["none"] + FramePainter_unet_list,),
                "sparse_control_ckpt": (["none"] + sparse_control_ckpt_list,),
            },
        }

    RETURN_TYPES = ("MODEL_FramePainter",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "FramePainter"

    def loader_main(self,model,FramePainter_unet,sparse_control_ckpt):

        if FramePainter_unet!="none":  
            FramePainter_unet=folder_paths.get_full_path("FramePainter", FramePainter_unet)
        else:
            raise ValueError("Please select a valid FramePainter_unet.")
        if sparse_control_ckpt!="none":  
            sparse_control_ckpt=folder_paths.get_full_path("FramePainter", sparse_control_ckpt)
        else:
            raise ValueError("Please select a valid sparse_control_ckpt.")
        
        svd_repo = os.path.join(current_node_path, "svd_repo")
        # load model
        print("***********Load model ***********")
        unet_config_file=os.path.join(svd_repo, "unet")
        unet=convert_cf2diffuser(model.model,unet_config_file)

        # unet = UNetSpatioTemporalConditionEdit.from_pretrained(
        #     svd_repo,
        #     subfolder="unet",
        #     low_cpu_mem_usage=True,
        #     variant="fp16"
        # )

        sparse_control_encoder = SparseControlEncoder()

        # vae = AutoencoderKLTemporalDecoder.from_pretrained(
        #     svd_repo, subfolder="vae",variant="fp16")
        vae_config=os.path.join(svd_repo, "vae/config.json")
        vae_config=OmegaConf.load(vae_config)
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            svd_repo, subfolder="scheduler")
        pipeline = FramePainterPipeline.from_pretrained(
            svd_repo,
            sparse_control_encoder=sparse_control_encoder, 
            unet=unet,
            vae_config=vae_config,
            revision=None,
            noise_scheduler=noise_scheduler,
            #vae=vae,
            )

        set_matching_attention(pipeline.unet)
        set_matching_attention_processor(pipeline.unet, MatchingAttnProcessor2_0(batch_size=2))

        pipeline.set_progress_bar_config(disable=False)

        sparse_control_dict=load_file(sparse_control_ckpt)
        pipeline.sparse_control_encoder.load_state_dict(sparse_control_dict, strict=True)
        FramePainter_unet_dict=load_file(FramePainter_unet)
        pipeline.unet.load_state_dict(FramePainter_unet_dict, strict=True)

        print("***********Load model done ***********")
        del sparse_control_dict,FramePainter_unet_dict
        gc.collect()
        torch.cuda.empty_cache()
        return (pipeline,)



class FramePainter_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_FramePainter",),
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "image": ("IMAGE",), # B H W C C=3
                "mask": ("MASK",), # B H W 
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 15, "max": 100, "step": 1, "display": "number"}),
                "guidance_scale":("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0,"step": 0.5}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1920, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1920, "step": 64, "display": "number"}),
                "control_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0,"step": 0.05}),
                },
                }
                         
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "FramePainter"
    
    
    def sampler_main(self, model,clip_vision,vae,image,mask,seed,steps,guidance_scale,width,height,control_scale):

        model.to("cuda")
        #vae=kwargs.get("vae", None)
        image_embeds = clip_vision.encode_image(image)["image_embeds"] #torch.Size([1, 1024])
        image_embeds=image_embeds.clone().detach().to(device, dtype=torch.float16) # dtype需要改成可选

        print("***********Start infer  ***********")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), timer("inference"):
            
            input_image_resized,merged_image=process_image_with_mask(image,mask, width, height)
            
            validation_control_images = [
                Image.new("RGB", (width, height), color=(0, 0, 0)), 
                merged_image
            ]
            if vae is not None:
                validation_control_ = [np.array(pil2narry(image)) for image in validation_control_images]
                validation_control_ = [torch.from_numpy(img).to(device, dtype=torch.float16) for img in validation_control_]
                validation_control_ = torch.cat(validation_control_, dim=0)
                input_latents=vae.encode(tensor_upscale(image,width,height)).to(device, dtype=torch.float16)
                validation_control_images=validation_control_.unsqueeze(0).permute(0,1,4,2,3).to(device, dtype=torch.float16)
                negative_image_latents = torch.zeros_like(input_latents)
                input_latents=torch.cat([negative_image_latents, input_latents])
            else:
                raise "vae is None"
            result = model(
                input_image_resized, 
                validation_control_images,
                height=height,
                width=width,
                edit_cond_scale=control_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator().manual_seed(seed),
                output_type="latent",#out lantents
                image_embs=image_embeds,
                input_latents=input_latents,
               
            ).frames[0],
        
        b,_,_,_=result[0].shape
        last= torch.chunk(result[0], chunks=b)
        image=vae.decode(last[-1])
        gc.collect()
        torch.cuda.empty_cache()
        return (image,)
        #image = result[0][1]
        #return (pil2narry(image),out,)



NODE_CLASS_MAPPINGS = {
    "FramePainter_Loader":FramePainter_Loader,
    "FramePainter_Sampler":FramePainter_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePainter_Loader":"FramePainter_Loader",
    "FramePainter_Sampler":"FramePainter_Sampler",
}
