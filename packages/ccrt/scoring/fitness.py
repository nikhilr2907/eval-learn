from typing import List

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


def _load_unet(model_id: str, device: str) -> UNet2DConditionModel:
    return UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device).eval()


def _encode(text: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: str) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(tokens.input_ids.to(device))[0]


@torch.no_grad()
def _partial_denoise(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    start_code: torch.Tensor,
    text_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    t_enc: int,
    guidance_scale: float,
    device: str,
) -> torch.Tensor:
    latents = start_code.clone().to(device)
    for t in scheduler.timesteps[:t_enc]:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        combined = torch.cat([uncond_emb, text_emb])
        noise_pred = unet(latent_input, t, encoder_hidden_states=combined).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


def _score_prompt(
    prompt: str,
    original_unet: UNet2DConditionModel,
    erased_unet: UNet2DConditionModel,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    scheduler: DDIMScheduler,
    device: str,
    ddim_steps: int = 50,
    start_guidance: float = 3.0,
    image_size: int = 512,
) -> float:
    criteria = torch.nn.MSELoss()

    text_emb = _encode(prompt, tokenizer, text_encoder, device)
    uncond_emb = _encode("", tokenizer, text_encoder, device)

    t_enc = torch.randint(1, ddim_steps, (1,)).item()
    t_ddpm = scheduler.timesteps[t_enc - 1]

    start_code = torch.randn(1, 4, image_size // 8, image_size // 8, device=device)

    z_orig = _partial_denoise(original_unet, scheduler, start_code, text_emb, uncond_emb, t_enc, start_guidance, device)
    z_erased = _partial_denoise(erased_unet, scheduler, start_code, text_emb, uncond_emb, t_enc, start_guidance, device)

    e_orig = original_unet(z_orig, t_ddpm, encoder_hidden_states=text_emb).sample
    e_erased = erased_unet(z_erased, t_ddpm, encoder_hidden_states=text_emb).sample

    return criteria(e_erased, e_orig).item()


def compute_fitness(
    prompts: List[str],
    original_model_id: str,
    erased_model_id: str,
    device: str = "cuda:0",
    ddim_steps: int = 50,
    start_guidance: float = 3.0,
    image_size: int = 512,
) -> float:
    """
    Compute mean noise-space MSE between the original and erased model
    across all prompts.

    Low  = erased model behaves differently from original → good erasure
    High = erased model still behaves like original → concept still leaks

    Parameters
    ----------
    prompts           : list of natural language prompts to evaluate
    original_model_id : HF repo ID or local path of the original model
    erased_model_id   : HF repo ID or local path of the erased model
    device            : torch device string

    Returns
    -------
    mean MSE across all prompts (float)
    """
    tokenizer = CLIPTokenizer.from_pretrained(original_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(original_model_id, subfolder="text_encoder").to(device).eval()
    original_unet = _load_unet(original_model_id, device)
    erased_unet = _load_unet(erased_model_id, device)

    scheduler = DDIMScheduler.from_pretrained(original_model_id, subfolder="scheduler")
    scheduler.set_timesteps(ddim_steps)

    scores = []
    for prompt in tqdm(prompts, desc="compute_fitness"):
        scores.append(
            _score_prompt(
                prompt, original_unet, erased_unet,
                tokenizer, text_encoder, scheduler,
                device, ddim_steps, start_guidance, image_size,
            )
        )

    return sum(scores) / len(scores)
