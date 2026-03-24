import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional


class UCEWeightCreator:
    """
    Create custom UCE weights for erasing specific concepts.

    Uses the bundled UCE training script included in this package.
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cuda",
    ):
        """
        Initialize weight creator.

        Args:
            model_id: Base Stable Diffusion model to modify.
            device: Device for training ('cuda' recommended).
        """
        self.model_id = model_id
        self.device = device

        # Get bundled training script
        package_dir = Path(__file__).parent
        self.train_script = package_dir / "training" / "uce_sd_erase.py"

        if not self.train_script.exists():
            raise FileNotFoundError(
                f"UCE training script not found at {self.train_script}\n"
                "The training script should be bundled with the package."
            )

    def create_weights(
        self,
        concept: str,
        output_path: str,
        concept_type: str = "object",
    ):
        """
        Create UCE weights to erase a specific concept.

        Args:
            concept: Concept to erase (e.g., "car", "nudity", "violence").
            output_path: Where to save the weights (.safetensors file).
            concept_type: Type of concept ("object", "style", "attribute").

        This will run UCE training which may take 5-30 minutes on GPU.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Creating UCE weights to erase concept: '{concept}'")
        print(f"This may take 5-30 minutes depending on your GPU...")
        print(f"Model: {self.model_id}")
        print(f"Device: {self.device}")

        # Prepare save directory
        temp_save_dir = output_path.parent / f"_uce_temp_{concept}"
        temp_save_dir.mkdir(parents=True, exist_ok=True)

        # Run UCE training script
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(self.train_script),
                    "--edit_concepts", concept,
                    "--concept_type", concept_type,
                    "--save_dir", str(temp_save_dir),
                    "--exp_name", f"uce_{concept}",
                    "--model_id", self.model_id,
                    "--device", self.device,
                ],
                check=True,
            )

            # Move created weights to final location
            # UCE creates: {save_dir}/{exp_name}.safetensors
            created_weight = temp_save_dir / f"uce_{concept}.safetensors"
            if created_weight.exists():
                shutil.move(str(created_weight), str(output_path))
                print(f"✓ Weights created successfully: {output_path}")
            else:
                raise FileNotFoundError(f"Expected weight file not found: {created_weight}")

        finally:
            # Clean up temp directory
            if temp_save_dir.exists():
                shutil.rmtree(temp_save_dir)

        return str(output_path)

    def create_bundled_weights(self, concepts: list, output_dir: str):
        """
        Create multiple UCE weights (helper for creating bundled presets).

        Args:
            concepts: List of concepts to erase.
            output_dir: Directory to save all weights.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_weights = {}
        for concept in concepts:
            output_path = output_dir / f"uce_{concept}.safetensors"
            try:
                self.create_weights(concept, str(output_path))
                created_weights[concept] = str(output_path)
            except Exception as e:
                print(f"Failed to create weights for '{concept}': {e}")

        return created_weights
