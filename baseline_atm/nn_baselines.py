from nn_utils import *
import os
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 0
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
model_types = ['MLP', 'LSTM', 'LATTICE', 'TCN', 'cTransformer', 'iTransformer']
for model_type in model_types:
    reproducibility(seed)
    all_results = {}
    for month in months:
        start = time.time()
        scenarios = load_scenarios(f"../data/scenario_generation_2022-{month}.pkl")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = main(scenarios, model_type, device, verbose=False)
        end = time.time()
        runtime = end - start
        results['runtime_min'] = runtime / 60.0  # in minutes
        print(f"[Month {month}] Model={model_type} → Test Results: {results}")
        all_results[month] = results
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    average_performance = results_df.mean()
    print(f"[Month Average] Model={model_type} → Test Results: {average_performance.to_dict()}")
    results_df.loc['Average'] = average_performance
    results_df.to_csv(f'nn_results/{model_type}.csv')
