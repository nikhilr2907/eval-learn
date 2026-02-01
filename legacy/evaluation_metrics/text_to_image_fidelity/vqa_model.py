import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class VQAModel:
    """
    VQA Model wrapper utilizing BLIP-2 for answering visual questions.
    """
    def __init__(self, model_id="Salesforce/blip2-flan-t5-xl", device="cuda"):
        # Auto-detect device (GPU preference)
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load the BLIP-2 model components
        print(f"Initializing VQA: {model_id} on {self.device}")
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.model.eval()

    @torch.no_grad()
    def answer_pil(self, pil_image, question: str, max_new_tokens=10) -> str:
        """
        Directly process a PIL Image and a question string.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Encode image and text question
        inputs = self.processor(
            images=pil_image, 
            text=question, 
            return_tensors="pt"
        ).to(self.device, torch.float16)

        # Generate the response
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode results
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
