import json
from pathlib import Path
from pycocotools.coco import COCO
import random


def dataset_extraction(instances_json_path = "/tmp/ko25/datasets/cocodataset/annotations/instances_val2017.json",
                        captions_json_path = "/tmp/ko25/datasets/cocodataset/annotations/captions_val2017.json",
                        output_directory = "data/clipscore", forget_category = 'knife', seed = 66900, number_of_samples = 50):
    """
    dataset_extraction takes in the COCO data set, and turns it into a suitable format for CLIPscore
    This function was used to create the data_forget.json and data_normal.json data sets
    Where each entry for the json files are the image id and the prompt
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

if __name__ == "__main__":
    dataset_extraction(forget_category="knife", number_of_samples=5)