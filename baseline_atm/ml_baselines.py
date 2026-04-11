from ml_utils import *
import pandas as pd

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
models = ['linear', 'rf', 'svm', 'xgb']
for model in models:
    all_results = {}
    for month in months:
        scenarios = load_scenarios(f"../data/scenario_generation_2022-{month}.pkl")
        res = train_and_evaluate_from_scenarios(scenarios, model_type=model)
        print(f"[Month {month}] Model={model} → Test Results: {res['test']}")
        all_results[month] = res['test']
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    average_performance = results_df.mean()
    print(f"[Month Average] Model={model} → Test Results: {average_performance.to_dict()}")
    results_df.loc['Average'] = average_performance
    results_df.to_csv(f'ml_results/{model}.csv')
