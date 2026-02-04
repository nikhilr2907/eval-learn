import subprocess
import sys
from pathlib import Path
import argparse

# finds the projects root directory and uses full absolute path
# root is at eval-learn/
root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

# added root directory to sys.path for importing from eval_learn
sys.path.append(str(root_dir))

def git_clone_repo():
    repo_directory = root_dir / "models" / "unified_concept_editing"
    if repo_directory.exists():
        print('UCE Repository already exists. No cloning required.')
    else:
        print('Cloning Required. Cloning UCE Repository...')
        subprocess.run(["git", "clone", "https://github.com/rohitgandikota/unified-concept-editing.git", str(repo_directory)])

def weight_creation(concept: str, output_path: str = "src/eval_learn/external/UCE/weights/"):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # filepath to the unlearning script
    UCE_library_filepath = root_dir / "models" / "unified_concept_editing" / "trainscripts" / "uce_sd_erase.py"
    # runs the python code to create weights
    # flags are edit_concepts, concept_type, exp_name and save_dir based on what uce_sd_erase.py requires
    print("Creating Weights...")
    subprocess.run([sys.executable, str(UCE_library_filepath), "--edit_concepts", concept, "--concept_type", "object", "--save_dir", str(output_path), "--exp_name", f"uce_{concept}"], check = True)
    print("Weights Created Successfully and saved to" + str(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept', type=str, required=True) # concept to unlearn
    parser.add_argument('--output_path', type=str, default = "src/eval_learn/external/UCE/weights/") # optional argument, by default we save all weights to weight folder
    arg = parser.parse_args()
    git_clone_repo()
    weight_creation(arg.concept, arg.output_path)