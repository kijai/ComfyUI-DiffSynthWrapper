import torch
import os
import sys

from torchvision import transforms

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from diffsynth import ModelManager, SVDVideoPipeline

class DownloadAndLoadDiffSynthExVideoSVD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "diffsynth_model": (
                    [ 
                    'ECNU-CILab/ExVideo-SVD-128f-v1',
                    ],
                    {
                    "default": 'ECNU-CILab/ExVideo-SVD-128f-v1'
                    }),
            "svd_model": (folder_paths.get_filename_list("checkpoints"),),
            },
        }

    RETURN_TYPES = ("DIFFSYNTHMODEL",)
    RETURN_NAMES = ("diffsynth_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DiffSynthWrapper"

    def loadmodel(self, diffsynth_model, svd_model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = torch.float16

        svd_model_path = folder_paths.get_full_path("checkpoints", svd_model)

        model_name = diffsynth_model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "diffsynth", model_name)
        model_full_path = os.path.join(model_path, "model.fp16.safetensors")
        
        if not os.path.exists(model_full_path):
            print(f"Downloading DiffSynth model to: {model_full_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="ECNU-CILab/ExVideo-SVD-128f-v1",
                            allow_patterns=['*fp16*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
    
        print(f"Loading DiffSynth model from: {model_full_path}")
        print(f"Loading SVD model from: {svd_model_path}")
        model_manager = ModelManager(torch_dtype=dtype, device=device)
        model_manager.load_models([svd_model_path, model_full_path])
        pipe = SVDVideoPipeline.from_model_manager(model_manager)

        diffsynth_model = {
            'pipe': pipe, 
            'dtype': dtype
            }

        return (diffsynth_model,)

class DiffSynthSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffsynth_model": ("DIFFSYNTHMODEL", ),
                "image": ("IMAGE", ),
                "frames": ("INT", {"default": 128, "min": 1, "max": 128, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 512, "step": 1}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 512, "step": 1}),
                "min_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "contrast_enhance_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 10.0, "step": 0.01}),
                "denoising_strength": ("FLOAT", {"default": 1., "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_video": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "DiffSynthWrapper"

    def process(self, diffsynth_model, height, width, steps, motion_bucket_id, fps, frames, image, 
                seed, min_cfg_scale, max_cfg_scale, denoising_strength, contrast_enhance_scale, noise_aug_strength, 
                keep_model_loaded, input_video=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        pipe = diffsynth_model['pipe']
        pipe.to(device)
        torch.manual_seed(seed)

        input_image = image.clone().permute(0, 3, 1, 2) * 2 - 1
        
        if input_video is not None:
            input_video = input_video.permute(0, 3, 1, 2) * 2 - 1

        video = pipe(
        input_image=input_image,
        num_frames=frames, 
        fps=fps, 
        height=height, 
        width=width,
        motion_bucket_id=motion_bucket_id,
        num_inference_steps=steps,
        min_cfg_scale=min_cfg_scale, 
        max_cfg_scale=max_cfg_scale, 
        contrast_enhance_scale=contrast_enhance_scale,
        noise_aug_strength=noise_aug_strength,
        denoising_strength=denoising_strength,
    )
        if not keep_model_loaded:
            pipe.to(offload_device)
            mm.soft_empty_cache()
        
        transform = transforms.ToTensor()
        tensors_list = [transform(image) for image in video]
        batch_tensor = torch.stack(tensors_list, dim=0)
        batch_tensor = batch_tensor.permute(0, 2, 3, 1).cpu().float()
        
        return (batch_tensor,)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDiffSynthExVideoSVD": DownloadAndLoadDiffSynthExVideoSVD,
    "DiffSynthSampler": DiffSynthSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDiffSynthExVideoSVD": "DownloadAndLoadDiffSynthExVideoSVD",
    "DiffSynthSampler": "DiffSynth Sampler",
}