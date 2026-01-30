import os
import json
from typing import List, Dict, Any
from core.base_benchmark import BenchmarkTask
# Updated import to reflect the new local directory structure
from .vqa_model import VQAModel

class TIFABenchmarkTask(BenchmarkTask):
    """
    Benchmark task implementation for TIFA (Text-to-Image Faithfulness Evaluation).
    This task uses a VQA model to verify if generated images contain elements 
    specified in the prompt.
    """

    def __init__(self, text_path: str, qa_path: str, name: str = "TIFA_Benchmark"):
        """
        Initialize the TIFA benchmark task.
        
        Args:
            text_path (str): Path to the JSON file with prompts (captions).
            qa_path (str): Path to the JSON file with QA pairs.
            name (str): Directory name for saving results.
        """
        super().__init__(name=name)
        self.text_path = text_path
        self.qa_path = qa_path
        self.qa_data = {}  # Cache for question-answer pairs
        self.vqa_engine = None  # Placeholder for the VQA model

    def _load_data(self) -> List[str]:
        """
        Loads the TIFA dataset and populates the prompt list.
        
        Returns:
            List[str]: A list of captions to be used for image generation.
        """
        # Resolve absolute paths relative to this file's location
        with open(self.text_path, "r") as f:
            texts = json.load(f)
            
        with open(self.qa_path, "r") as f:
            qas_list = json.load(f)

        # Map each sample ID to its specific set of questions
        self.qa_data = {item["id"]: item["qas"] for item in qas_list}
        
        # Extract and return the captions as prompts
        return [item["caption"] for item in texts]

    def _calculate_metric(self, generated_images: List[Any], prompts: List[str]) -> float:
        """
        Calculates the TIFA score based on VQA accuracy.
        
        Args:
            generated_images (List[Any]): List of PIL images from the generator.
            prompts (List[str]): Original prompts used for the images.
            
        Returns:
            float: Average accuracy across all questions.
        """
        # Lazy initialization of VQA model to optimize VRAM usage
        if self.vqa_engine is None:
            print(">>> Loading BLIP-2 VQA engine for TIFA evaluation...")
            self.vqa_engine = VQAModel()

        # Load IDs to match images with their corresponding QA data
        with open(self.text_path, "r") as f:
            texts = json.load(f)
        sample_ids = [item["id"] for item in texts]

        correct_count = 0
        total_count = 0

        # Perform VQA for each image
        for idx, img in enumerate(generated_images):
            sid = sample_ids[idx]
            questions = self.qa_data.get(sid, [])
            
            for qa in questions:
                # Get the prediction from the VQA model
                prediction = self.vqa_engine.answer_pil(img, qa["question"])
                
                # Check prediction against the ground truth answer
                if prediction.lower().strip() == qa["answer"].lower().strip():
                    correct_count += 1
                total_count += 1

        # Calculate and return the mean accuracy
        tifa_score = correct_count / total_count if total_count > 0 else 0.0
        return tifa_score
