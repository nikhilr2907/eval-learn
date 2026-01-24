import json
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from evaluation_metrics.text_to_image_fidelity.clipscore.image_generation import diffusion_model_pipeline
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from pycocotools.coco import COCO
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from core.base_technique import UnlearningTechnique


class ClipScore(UnlearningTechnique):
    """
    Clipscore Evaluation
    Data used: COCO validation data set

    Call obtain data sets first, to extract the needed information from COCO to form two separate data sets

    Using these two data sets, apply it to the unlearning technique and generate images

    Finally, call score method to calculate the clip score

    If needed, save and write results into a json file
    """

    def __init__(self, model_name = "openai/clip-vit-base-patch32", number_of_samples = 50):
        self.model_name = model_name
        self.number_of_samples = number_of_samples

    def score(self, json_filepath):
        """
        Given a json file containing the image paths and prompts,
        converts the image into tensor form and calculates the CLIP score

        Inputs:
        json_filepath (str): file path to the json file containing the image paths and prompts

        Return:
        average CLIP score (float) for that data set
        score dictionary(dict) for that data set, key: image_id, value clipscore for that image
        """
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_score = CLIPScore(model_name_or_path =self.model_name).to(device)
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
        

    def obtain_datasets(self, instances_json_path = "/tmp/ko25/datasets/cocodataset/annotations/instances_val2017.json",
                        captions_json_path = "/tmp/ko25/datasets/cocodataset/annotations/captions_val2017.json",
                        output_directory = "data/clipscore", forget_category = 'knife', number_of_samples = 50):
        """
        dataset_extraction takes in the COCO data set, and turns it into a suitable format for CLIPscore
        This function was used to create the data_forget.json and data_normal.json data sets
        Where each entry for the json files are the image id and the prompt

        Inputs:
            instances_json_path (str): file path to the COCO instances json file (has default value)
            captions_json_path (str): file path to the COCO captions json file (has default value)
            output_directory (str): file path to save the created data files (has default value)
            forget_category (str): the object category to unlearn from the data set (has default value)
            number_of_samples (int): number of samples to extract from each data set (has default value)
        Returns two ouputs:
            data_forget (list): a list of dictionaries containing image ids as the key and the prompt as the value
            data_normal (list): a list of dictionaries containing image ids as the key and the prompt as the value
        """
        # Load COCO datasets
        data_instances = COCO(instances_json_path)
        data_captions = COCO(captions_json_path)

        # Based on the category to forget, we get all images ids associated with that category id we want to forget
        # getCatIds returns an id number for a 'word' input
        # index 0 used as output of getCatIds is a list, we only want the first element
        forget_category_IDs = data_instances.getCatIds(catNms=[forget_category])[0]
        forget_image_IDs = data_instances.getImgIds(catIds=[forget_category_IDs])

        All_Image_IDS = data_instances.getImgIds()
        # Remove images that have the forgetten category inside, remaining will form the normal data set
        normal_image_IDs = list(set(All_Image_IDS) - set(forget_image_IDs))

        # Randomly sample data from both list of IDs
        sampled_forget_image_IDs = random.sample(forget_image_IDs, number_of_samples)
        sampled_normal_image_IDs = random.sample(normal_image_IDs, number_of_samples)

        # Create data_forget and data_normal
        data_forget = []
        data_normal = []
        for index, image_ID in enumerate(sampled_forget_image_IDs):
            # finding the caption ids based on that image having thatinput image id
            captions_IDs = data_captions.getAnnIds(imgIds=[image_ID])
            # produces the captions associated with the caption ids
            captions = data_captions.loadAnns(captions_IDs)
            # captions is a list of dictionaries, we only want just one prompt per entry
            # so we take the prompt with the first index
            data_forget.append({"image_id": image_ID,"prompt": captions[0]['caption']})
    
        for index, image_ID in enumerate(sampled_normal_image_IDs):
            # finding the caption ids based on that image having thatinput image id
            captions_IDs = data_captions.getAnnIds(imgIds=[image_ID])
            # produces the captions associated with the caption ids
            captions = data_captions.loadAnns(captions_IDs)
            # captions is a list of dictionaries, we only want just one prompt per entry
            data_normal.append({"image_id": image_ID,"prompt": captions[0]['caption']})

        # Create output directory if it does not exist, files created will be nested inside the previous folder
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        # Writing of the data files to the output directory
        with open(Path(output_directory) / "data_forget.json", "w") as f_forget: 
            json.dump(data_forget, f_forget, indent = 4)
        with open(Path(output_directory) / "data_normal.json", "w") as f_normal:
            json.dump(data_normal, f_normal, indent = 4)

        return data_forget, data_normal
    
    def SLD_generate_images(self, forget_data_filepath, output_file_directory, sld_config = SafetyConfig.STRONG, pipe = None):
        """
        Using the forget_data prompts, SLD identifies the category to unlearn
        The diffusion model is unlearnt from that category
        And based on the prompts given in data_forget.json, the unlearnt model generates images
        these generated images will be used for CLIPscore calculation
        """
        if pipe is None: # if diffusion pipeline is not initialised we load it
            pipe = diffusion_model_pipeline()
    
        # navigate to the file path containing the data_forget.json and open it
        with open(forget_data_filepath, "r") as f:
            forget_data = json.load(f)

        # Creat the output directory if it does not exist
        Path(output_file_directory).mkdir(parents=True, exist_ok=True)

        generated_images = []
        # Iterate through the forget prompts and identify the category to unlearn
        # afterwhich apply the unlearning SLD technique
        # after unlearning, apply the prompts to generate images
        for entry in forget_data:
            unlearnt_image = pipe.generate([entry['prompt']], config=sld_config)[0]
            # create a file path to save the unlearnt image and save the image
            image_path = f"{output_file_directory}/unlearnt_image_{entry['image_id']}.png"
            unlearnt_image.save(image_path)
            # the last key pair shows the directory of where the unlearnt images are
            generated_images.append({"image_id": entry['image_id'], "prompt": entry['prompt'], "image_path": image_path})
        # Save the generated images list containing the images filepath
        output_path = f"{output_file_directory}/generated_unlearnt_images.json"
        # write into a json file and create this file
        with open(output_path, "w") as f:
            json.dump(generated_images,f, indent=4)
    
        return output_path

    def generate_normal_images(self,normal_data_filepath, output_file_directory, pipe = None):
        """
        Generate images from normal data set
        """
        if pipe is None: # if diffusion pipeline is not initialised we load it
            pipe = diffusion_model_pipeline()
    
        # navigate to the file path containing the data_forget.json and open it
        with open(normal_data_filepath, "r") as f:
            normal_data = json.load(f)

        # Creat the output directory if it does not exist
        Path(output_file_directory).mkdir(parents=True, exist_ok=True)

        generated_images = []
        # Iterate through the normal prompts and generate images
        for entry in normal_data:
            # for normal image generation, we disable SLD completely by using guidance scale = 0
            normal_image = pipe.generate([entry['prompt']], config={"sld_guidance_scale": 0})[0]
            # create a file path to save the unlearnt image and save the image
            image_path = f"{output_file_directory}/normal_image_{entry['image_id']}.png"
            normal_image.save(image_path)
            # the last key pair shows the directory of where the images are
            generated_images.append({"image_id": entry['image_id'], "prompt": entry['prompt'], "image_path": image_path})
        # Save the generated images list containing the images filepath
        output_path = f"{output_file_directory}/generated_normal_images.json"
        # write into a json file and create this file
        with open(output_path, "w") as f:
            json.dump(generated_images,f, indent=4)
    
        return output_path
    def benchmark(self, instances_json_path = "/tmp/ko25/datasets/cocodataset/annotations/instances_val2017.json",
                        captions_json_path = "/tmp/ko25/datasets/cocodataset/annotations/captions_val2017.json",
                        output_directory = "data/clipscore", forget_category = 'knife'):
        self.obtain_datasets(output_directory=output_directory, forget_category= forget_category, number_of_samples=number_of_samples,
                             instances_json_path=instances_json_path, captions_json_path=captions_json_path)
        
        forget_data_filepath = f"{output_directory}/data_forget.json"
        normal_data_filepath = f"{output_directory}/data_normal.json"

        unlearnt_images_json_filepath = self.SLD_generate_images(forget_data_filepath = forget_data_filepath, output_directory = output_directory)
        normal_images_json_filepath = self.generate_normal_images(normal_data_filepath = normal_data_filepath, output_file_directory= output_directory)
        unlearnt_average_clipscore, unlearnt_score_dict = self.score(json_filepath = unlearnt_images_json_filepath)
        normal_average_clipscore, normal_score_dict = self.score(json_filepath = normal_images_json_filepath)

        return unlearnt_average_clipscore, unlearnt_score_dict, normal_average_clipscore, normal_score_dict

if __name__ == "__main__":
    clip_score = ClipScore(number_of_samples = 50)
    clip_score_results = clip_score.benchmark(instances_json_path = "/tmp/ko25/datasets/cocodataset/annotations/instances_val2017.json",
                        captions_json_path = "/tmp/ko25/datasets/cocodataset/annotations/captions_val2017.json",
                        output_directory = "data/clipscore", forget_category = 'knife', number_of_samples = 50)
    print('Average clipscore after unlearning: ', clip_score_results[0], 'Clipscore Dictionary after unlearning:' clip_score_results[1], 'Average clipscore for normal data: ',  clip_score_results[2], 'Clipscore Dictionary for normal data:', clip_score_results[3])