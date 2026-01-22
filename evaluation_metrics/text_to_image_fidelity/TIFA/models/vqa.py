import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class VQAModel:
    def __init__(
        self,
        model_id="Salesforce/blip2-flan-t5-xl",
        device="cuda"
    ):
        self.device = device

        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def answer(self, image_path, question, max_new_tokens=10):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device, torch.float16)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        answer = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return answer
