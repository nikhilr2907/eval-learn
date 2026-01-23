# Msc AI Group Project

for .env file make sure to add huggingface token

Evaluation Benchmark for Unlearning in T2I diffusion models
Project Description
Text-guided diffusion models are used by millions of users, but can be easily exploited to produce harmful content. Concept unlearning methods aim at reducing the models’ likelihood of generating harmful content. To address this issue, there are many efforts made towards developing unlearning techniques for harmful concepts for these models. To guarantee effective unlearning, these techmmniques are evaluated on extensive benchmarks and across varied metrics.
However, given that there are more than 25 different unlearning techniques, it becomes essential to have an exhaustive and fair comparison across all of them. Unfortunately, there is no common standard benchmark for evaluating the effectiveness of unlearning in these models, leading to inconclusive projections about the effectiveness of one technique over the other.
As a part of this project, we aim to close this gap by developing a comprehensive evaluation benchmark for evaluating the effectiveness of unlearning in Text-to-Image Diffusion models. The project will also involve evaluating multiple unlearning techniques on the designed benchmark to give a comprehensive view of the development and effectiveness of the safeguard, which would motivate better directions for research and would ease the task of unlearning evaluation for upcoming research works in the area.
____________________________________________________________________________
Guidelines for the project:
For the sake of simplicity and modularity, we have divided the project into 4 stages.
Stage 1: Setting up the evaluation techniques 
In this stage, we aim to target the following goals:
Reading about the evaluation techniques. Refer to the Evaluation Papers tab for the list of targeted techniques.
Standardizing the datasets for each of these techniques.
Setting up the implementation of the evaluation techniques.
During this stage, please keep a note of  the decisions taken while standardizing the dataset and setting up the implementation techniques. The following, for instance would be necessary to include in the final report later towards the end of the project:
Dataset:
Reasoning for the choice of dataset for each evaluation technique.
Link to the dataset and proper citation of the dataset.
Evaluation technique:
Hyperparameters set for the evaluation and the reasoning behind the choice.
Proper citation for the evaluation technique.
Stage 2: Packaging the techniques into a python library
In this stage you will be required to package the project into a python library. This is a very straightforward stage and involves preparing a module first for the code, and then submitting it as a python package.
Here are some resources which would be helpful:
Some(really quick and very brief) theory on Python Packages if you are new: Python Packages - GeeksforGeeks
Official Guide on creating and structuring your package: Packaging Python Projects
Things to complete at the end of Stage 2: 
You must prepare a Readme for the Github repository you are working in, where you clearly describe the instructions to install the package and how to use it.
You should create a page on https://pypi.org/ and share the link to the page.
Stage 3: Running the evaluations
In this stage, you are required to run the evaluation of different unlearning techniques using the created python library. To start with, we aim at evaluating 25 prominent unlearning techniques. Please find the link to the respective papers under the Unlearning Techniques tab.
Stage 4: Setting up a leaderboard on Huggingface and report finalization
In the final stage of the project, we would create a leaderboard page on huggingface to show the comparison of different unlearning techniques on the metrics in the library. Following is a tutorial on how to setup a Huggingface leaderboard: Building a leaderboard using a template




Sprint 1

Goal: we want to implement a pipeline thats able to benchmark the SLD-max unleanring technique on 4 different benchmarks, which are: ASR, TIFA, CLIPScore, FID Score

Datasets per metrix:
ASR - I2P
TIFA - TIFA v1.0 Benchmark (subset of COCO/DrawBench with pre-generated questions)
CLIPScore - MS-COCO Validation Set
FID Score - MS-COCO Validation Set (at least 30k) https://gist.github.com/sayakpaul/0c4435a1df6eb6193f824f9198cabaa5



3. How to Test Each Benchmark (The "How-To")
You need two different datasets: one for Safety (ASR) and one for Utility (TIFA, CLIP, FID).
A. ASR (Attack Success Rate)
Goal: Measure how often SLD-max fails to suppress unsafe content.
Dataset: I2P (Inappropriate Image Prompts). This contains ~4,700 prompts designed to trigger nudity/violence.
Method: 
1. Generate images for I2P prompts using SLD-max.
2. Run an external classifier on the output images. Standard practice uses NudeNet (for nudity) and Q16 (for general inappropriateness).
3. Calculation: $ASR = \frac{\text{Count of Detected Unsafe Images}}{\text{Total Images Generated}}$
Expected Result: SLD-max should have a very low ASR (e.g., < 1%).
B. TIFA (Text-to-Image Faithfulness Assessment)
Goal: Measure if the safety filter "broke" the image semantics (e.g., did it remove the "person" entirely when asked for a "naked person"?).
Dataset: TIFA v1.0 Benchmark (subset of COCO/DrawBench with pre-generated questions).
Method:
Generate images for TIFA prompts.
For each image, feed the image + the pre-generated questions (from the benchmark JSON) into a VQA model (like mPLUG-large or BLIP-2).
Calculation: Accuracy of the VQA model answering the questions correctly (Yes/No or multiple choice).
C. CLIPScore
Goal: Measure general semantic alignment (does the image match the text?).
Dataset: MS-COCO Validation Set (standard for this).
Method:
Generate images for COCO prompts.
Use the clip-score library (or PyTorch torchmetrics).
Compute the Cosine Similarity between the Prompt Embedding and the Generated Image Embedding.
Calculation: Average cosine similarity across the dataset (often scaled by 2.5 or 100 depending on implementation).
D. FID (Fréchet Inception Distance)
Goal: Measure image realism and quality.
Dataset: MS-COCO Validation Set (30k images recommended for stable FID).
Method:
Generate a large batch of images (e.g., 10k or 30k) using SLD-max.
Use the clean-fid or torch-fidelity library.
Calculation: Compute the distance between the distribution of your Generated Images and the Ground Truth Images (from the COCO dataset source files).
Note: Lower FID is better.
