from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.runners import BenchmarkRunner

dataset = get_dataset("i2p.csv")
technique = get_technique("uce")
metric = get_metric("asr")

data_set_config = {'path': 'data/i2p/i2p_benchmark.csv', 'limit': 10}

technique_config = {
    'weights_path': 'src/eval_learn/external/UCE/weights/uce_nudity.safetensors',
    'model_id': 'CompVis/stable-diffusion-v1-4', 'device': 'cuda'}

metric_config = {'use_nudenet': True, 'device': 'cuda'}

runner = BenchmarkRunner(dataset_loader = dataset, technique_factory = technique, metric_factory = metric,
                         dataset_config = data_set_config, technique_config = technique_config, metric_config = metric_config, 
                         output_dir = 'results/i2p_uce_asr/', run_name = 'UCE_I2P_Nudity_ASR')
results = runner.run()

print( results ['metric_result']['value'])