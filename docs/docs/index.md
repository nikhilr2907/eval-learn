# Eval-Learn

A benchmarking framework for evaluating concept unlearning techniques 
in text-to-image diffusion models.

Unlearning techniques modify or constrain Stable Diffusion to prevent 
it from generating specific concepts — nudity, artistic styles, named 
individuals. Eval-Learn provides a common interface to run, compare, 
and evaluate these techniques under consistent conditions.

## What it includes

- **9 techniques** — ESD, MACE, UCE, AdvUnlearn, SAeUron, SAFREE, 
  SLD, Concept Steerers, Free Run
- **7 metrics** — ASR, FID, CLIP Score, ERR, TIFA, UA-IRA, MMA-Diffusion
- **2 Evaluation Modes** - Run an unlearning technique with a single metric pair together or run multiple evaluation metrics for a single technique.
