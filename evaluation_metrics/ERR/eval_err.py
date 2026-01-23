import .err
from .generate_err import ERREvaluator
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


error_scorer = ERREvaluator(oracle_classifier=model, processor=processor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


