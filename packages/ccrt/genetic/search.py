import random
import pickle
import os
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from .individual import Individual
from ..vocab.loader import load_vocab, load_wnids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models(model_id: str, device: str):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return tokenizer, text_encoder, unet, scheduler


# ---------------------------------------------------------------------------
# Diffusers partial denoising  (replaces LDM quick_sample_till_t)
# ---------------------------------------------------------------------------

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
    """Partially denoise start_code for t_enc DDIM steps."""
    latents = start_code.clone().to(device)
    for t in scheduler.timesteps[:t_enc]:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        combined_emb = torch.cat([uncond_emb, text_emb])
        noise_pred = unet(latent_input, t, encoder_hidden_states=combined_emb).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


# ---------------------------------------------------------------------------
# Per-individual fitness score
# ---------------------------------------------------------------------------

def _score_individual(
    word: str,
    original_unet: UNet2DConditionModel,
    erased_unet: UNet2DConditionModel,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    scheduler: DDIMScheduler,
    device: str,
    ddim_steps: int,
    start_guidance: float,
    image_size: int,
) -> float:
    criteria = torch.nn.MSELoss()

    def encode(text: str) -> torch.Tensor:
        tokens = tokenizer(
            [text],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return text_encoder(tokens.input_ids.to(device))[0]

    text_emb = encode(word)
    uncond_emb = encode("")

    t_enc = torch.randint(1, ddim_steps, (1,)).item()
    t_ddpm = scheduler.timesteps[t_enc - 1]

    start_code = torch.randn(1, 4, image_size // 8, image_size // 8, device=device)

    z_orig = _partial_denoise(
        original_unet, scheduler, start_code, text_emb, uncond_emb, t_enc, start_guidance, device
    )
    z_erased = _partial_denoise(
        erased_unet, scheduler, start_code, text_emb, uncond_emb, t_enc, start_guidance, device
    )

    with torch.no_grad():
        e_orig = original_unet(z_orig, t_ddpm, encoder_hidden_states=text_emb).sample
        e_erased = erased_unet(z_erased, t_ddpm, encoder_hidden_states=text_emb).sample

    return criteria(e_erased, e_orig).item()


# ---------------------------------------------------------------------------
# Genetic helpers
# ---------------------------------------------------------------------------

def _exist_ancestor(ind1: Individual, ind2: Individual, ancestors_map, descendants_map):
    common = set()
    for i in ind1.id:
        for j in ind2.id:
            common |= ancestors_map.get(i, set()) & ancestors_map.get(j, set())
            common |= descendants_map.get(i, set()) & descendants_map.get(j, set())
    return (True, list(common)) if common else (False, [])


def _remove_oversized(individual_list: List[Individual], max_concepts: int = 3) -> List[Individual]:
    return [ind for ind in individual_list if len(ind.concepts) < max_concepts]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_genetic_search(
    original_model_id: str,
    erased_model_id: str,
    concept_name: str,
    vocab_dir: Optional[str],
    output_dir: str,
    max_iterations: int,
    top_k: int,
    device: str = "cuda:0",
    ddim_steps: int = 50,
    image_size: int = 512,
    start_guidance: float = 3.0,
    crossover_num: float = 0.8,
) -> List[Individual]:
    """
    Run the CCRT genetic search to find concept combinations that most expose
    residual leakage in the erased model.

    Returns the final surviving population as a list of Individual objects
    (with glosses appended to concepts), and saves them to output_dir/entities.pkl.
    """
    class_map, gloss_map, ancestors_map, descendants_map = load_vocab(vocab_dir)
    wnids = load_wnids(vocab_dir)

    # build initial population — one individual per vocab entry
    individual_list: List[Individual] = []
    for wnid in wnids:
        if wnid in class_map:
            individual_list.append(Individual(id=[wnid], concepts=[class_map[wnid]]))

    # load both models (shared tokenizer + text encoder since only UNet is modified)
    tokenizer, text_encoder, original_unet, scheduler = _load_models(original_model_id, device)
    _, _, erased_unet, _ = _load_models(erased_model_id, device)
    text_encoder.eval()
    original_unet.eval()
    erased_unet.eval()
    scheduler.set_timesteps(ddim_steps)

    iteration = 0
    while len(individual_list) > top_k and iteration < max_iterations:
        iteration += 1
        pbar = tqdm(individual_list, desc=f"Scoring iteration {iteration}")
        for ind in pbar:
            word = ",".join(ind.concepts)
            ind.score = _score_individual(
                word, original_unet, erased_unet,
                tokenizer, text_encoder, scheduler,
                device, ddim_steps, start_guidance, image_size,
            )

        # select top-k survivors
        individual_list = sorted(individual_list, key=lambda x: x.score, reverse=True)[:top_k]

        # crossover
        n_crossover = int(max(2, top_k * crossover_num) // 2 * 2)
        pool = random.sample(individual_list, min(n_crossover, len(individual_list)))
        random.shuffle(pool)
        while len(pool) >= 2:
            ind1 = pool.pop()
            ind2 = pool.pop()
            flag, shared = _exist_ancestor(ind1, ind2, ancestors_map, descendants_map)
            if flag:
                new_concepts = [class_map[k] for k in shared if k in class_map]
                new_ids = shared
            else:
                new_concepts = ind1.concepts + ind2.concepts
                new_ids = ind1.id + ind2.id
            individual_list.append(Individual(id=new_ids, concepts=new_concepts))

        # mutation — 5% chance per concept slot
        n_mutate = max(1, len(individual_list) // 4)
        to_mutate = random.sample(individual_list, n_mutate)
        individual_list = [ind for ind in individual_list if ind not in to_mutate]
        all_ids = list(class_map.keys())
        for ind in to_mutate:
            for i in range(len(ind.concepts)):
                if random.random() < 0.05:
                    new_id = random.choice(all_ids)
                    ind.id[i] = new_id
                    ind.concepts[i] = class_map[new_id]
            individual_list.append(ind)

        individual_list = _remove_oversized(individual_list)
        print(f"iteration {iteration}: population={len(individual_list)}")

    # append glosses for downstream LLM prompt generation
    for ind in individual_list:
        id_to_idx = {id_: i for i, id_ in enumerate(ind.id)}
        for id_ in ind.id:
            if id_ in gloss_map:
                ind.concepts[id_to_idx[id_]] += f" : {gloss_map[id_]}"

    # persist
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = Path(output_dir) / "entities.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(individual_list, f)

    return individual_list
