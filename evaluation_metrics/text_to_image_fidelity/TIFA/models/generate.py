import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        device="cuda"
    ):
        self.device = device

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None  #safety-check off
        )
        self.pipe = self.pipe.to(device)

    @torch.no_grad()
    def generate(
        self,
        prompt,
        seed=42,
        num_steps=50,
        guidance_scale=7.5
    ):
        generator = torch.Generator(self.device).manual_seed(seed)

        image = self.pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        return image
