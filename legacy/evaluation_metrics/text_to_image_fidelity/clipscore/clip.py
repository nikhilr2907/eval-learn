import pandas as pd
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from pycocotools.coco import COCO
import random
import json
import csv

"""
For clip score, there needs to be two data sets:
- data_forget: contains prompts that asking the model to generate the forgetten object
- data_normal: contains prompts the model to generate other objects that do not relate to the unlearnt object

The model exists in two states, before unlearning and after unlearning
- data_forget goes into the model after unlearning, but implementation of SLD, directly uses the prompt to unlearn and generate the image
- data_normal goes into the model before unlearning

example:
untrain on dog, 100 of the 10000 in forget contains dog prompts
100 of the 10000 in normal contains other objects
iterate 100 times, with 100 images generated from 100 forget dog prompts
and 100 image generated from 100 normal prompts
find the clip score for the 100 and 100, find average, and that is for dogs
do for cat, repeat
"""


# CLIPSCORE
def CLIP_score_calculation(json_filepath, model_name = "openai/clip-vit-base-patch32"):
    """
    Given a json file containing the image paths and prompts, calculate the CLIP score
    Return the average CLIP score and score dictionary for that json file
    """
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_score = CLIPScore(model_name_or_path =model_name).to(device)

    with open(json_filepath, "r") as f:
        data = json.load(f)
    
    total_clip_score = 0.0
    score_dict = {}
    for entry in data:
        # load image
        image = Image.open(entry['image_path']).convert("RGB")
        prompt = entry['prompt']
        # calculate clip score
        # input to clip score is an image in tensor form and a prompt in string
        image_tensor = transforms.ToTensor()(image)
        image_tensor = (image_tensor * 255).to(torch.uint8).to(device)
        score = clip_score(image_tensor, prompt)
        # score is in the form of a tensor, and we need extract numerical part
        # used for calculating average over all entries
        total_clip_score += score.item()
        # we store clip scores for each entry in a dictionary for further reference and data analysis if needed
        score_dict[entry['image_id']] = score.item()
    
    average_clip_score = total_clip_score / len(data)
    return average_clip_score, score_dict

def write_results(average_forget_clipscore, forget_scores_dict, average_normal_clipscore, normal_scores_dict, output_filepath = "results/clipscore/clipscore_results.json", category_unlearnt="dog"):
    """
    Write the results to a json file
    """
    results = {"forget_data":{"category_unlearnt": category_unlearnt, "average_clipscore": average_forget_clipscore, "scores_dict": forget_scores_dict},
    "normal_data":{"category_unlearnt": category_unlearnt, "average_clipscore": average_normal_clipscore, "scores_dict": normal_scores_dict}}
    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results from unlearning {category_unlearnt} written and saved to: ", output_filepath)
if __name__ == "__main__":
    average_forget_clipscore, forget_scores_dict = CLIP_score_calculation("results/clipscore/forget/generated_unlearnt_images.json")
    average_normal_clipscore, normal_scores_dict = CLIP_score_calculation("results/clipscore/normal/generated_normal_images.json")
    print(f"CLIP Score for forget data: Average: {average_forget_clipscore}, Scores: {forget_scores_dict}")
    print(f"CLIP Score for normal data: Average: {average_normal_clipscore}, Scores: {normal_scores_dict}")
    write_results(average_forget_clipscore, forget_scores_dict, average_normal_clipscore, normal_scores_dict, "results/clipscore/clipscore_results.json", category_unlearnt="knife")
