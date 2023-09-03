import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipeline = StableDiffusionPipeline.from_pretrained(
    "andite/anything-v4.0", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)

pipeline.load_lora_weights(".", weight_name="genshin.safetensors")
prompt = ("masterpiece, best quality, absurdres, 1girl, school uniform, kangel, smile, standing, contrapposto, "
          "bedroom, leaning forward, <lora:kangelNeedyGirl_v10:1>, <lora:genshin-char-model:1>, genshin,")
negative_prompt = ("(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), "
                   "bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2)")

images = pipeline(prompt=prompt, 
    negative_prompt=negative_prompt, 
    width=512, 
    height=512, 
    num_inference_steps=20, 
    num_images_per_prompt=1,
    guidance_scale=7.0,
    generator=torch.manual_seed(9625644)
).images

images[0].save("test.jpg")
