import base64
import io
from typing import List

import openai
from PIL import Image


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_message(
    reference_images: List[Image.Image],
    generated_image: Image.Image,
    concept_name: str,
    concept_desc: str,
    all_concepts: List[str],
) -> dict:
    content = [
        {
            "type": "text",
            "text": (
                f"The style of {concept_name} work is: {concept_desc}. "
                f"As shown in the first {len(reference_images)} image(s). "
                f"Please determine if the last picture is in one of the following styles: "
                f"{all_concepts}. "
                "Just answer yes or no. "
                "If the image has no recognisable style, answer null."
            ),
        }
    ]

    for ref in reference_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(ref)}"},
        })

    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(generated_image)}"},
    })

    return {"role": "user", "content": content}


def evaluate_style(
    generated_images: List[Image.Image],
    reference_images: List[Image.Image],
    concept_name: str,
    concept_desc: str,
    api_key: str,
    all_concepts: List[str] = None,
) -> float:
    """
    Use GPT-4V to estimate what fraction of generated_images still visually
    exhibit concept_name's style, using reference_images as few-shot exemplars.

    Parameters
    ----------
    generated_images : images produced by the erased model
    reference_images : 3+ real examples of the concept style
    concept_name     : e.g. "Van Gogh"
    concept_desc     : short description e.g. "emotional colours, swirling brushwork"
    api_key          : OpenAI API key
    all_concepts     : full list of concept names for discriminative context
                       (defaults to [concept_name] if not provided)

    Returns
    -------
    style_precision : fraction of images still showing the concept (float 0–1)
                      lower = better erasure
    """
    if all_concepts is None:
        all_concepts = [concept_name]

    client = openai.OpenAI(api_key=api_key)

    yes_count = 0
    evaluated = 0

    for image in generated_images:
        message = _build_message(reference_images, image, concept_name, concept_desc, all_concepts)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[message],
            max_tokens=10,
        )
        answer = response.choices[0].message.content.strip().lower()
        if answer.startswith("yes"):
            yes_count += 1
        evaluated += 1

    return yes_count / evaluated if evaluated > 0 else 0.0
