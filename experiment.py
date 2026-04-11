from train import *
import wandb

# CUDA Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Tokens for Hugging Face and Weights & Biases
hf_token = "YOUR HUGGING FACE TOKEN HERE"
wanb_token = 'YOUR W&B TOKEN'
wanb_project = 'YOUR W&B PROJECT NAME'
wandb.login(key=wanb_token)

# Experiment Sweep Configuration
sweep_config = {'method': 'grid'}
metric = {'name': 'test/MAE', 'goal': 'minimize'}
sweep_config['metric'] = metric
parameters_dict = {
    'seed': {
        'values': [42] # Fixed
        },
    'model_name': {
        'values': ['meta-llama/Llama-3.2-1B',
                   'meta-llama/Llama-3.2-1B-Instruct',
                   'Qwen/Qwen3-0.6B',
                   'Qwen/Qwen3-0.6B-Base',
                   'EleutherAI/pythia-1b',
                   ]
        },
    'ablation_tag': {
        'values': [None, # Full Model
                   'exclude_flt_plan', # Context Removal Configurations
                   'exclude_text',
                   'exclude_notam',
                   'exclude_metar',
                   'exclude_taf',
                   'exclude_trajectory',
                   'exclude_focusing',
                   'exclude_active',
                   'exclude_affecting',
                   'LLMTIME', # Data Fusion Baseline
                   'AutoTimes',
                   'TimeLLM',
                   ]
        },
    'encoder':{
        'values': [None, # None if pre-encoded trajectory representations are available
                   'atscc',
                   'tcn_autoencoder',
                   'ts2vec',
                   'infots',
                   ]
        },
    'month':{ # Jan to Dec
        'values': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        }
    }
sweep_config['parameters'] = parameters_dict

# Fixed parameters
batch_size = 4
train_data_size = 0.8
learning_rate = 1e-5
weight_decay = 1e-5
num_epochs = 15
verbose = True
test_every = 1

# Run Experiment Sweep
def run_sweep(sweep_config):
    sweep_id = wandb.sweep(sweep_config, project=wanb_project)

    def sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config

            # Parameters from sweep
            model_name = config.model_name
            seed = config.seed
            ablation_tag = config.ablation_tag
            traj_encoder = config.encoder
            data = load_scenarios_from_hf(int(config.month), hf_token)

            # Run model training + eval
            model, loss_log, test_matrices_log, train_mean, train_std, model_state_log = model_runner(
                model_name, hf_token, data, batch_size, train_data_size,
                learning_rate, weight_decay, num_epochs, verbose,
                test_every, seed, ablation_tag, traj_encoder
            )

            # Find best validation loss
            best_epoch = 0
            best_loss = float('inf')
            for i, (train_loss, val_loss) in enumerate(loss_log):
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = i

            # Round best loss and test metrics to 4 decimal places for logging except R2
            best_loss = round(best_loss, 4)
            best_test_matrices = {k: round(v, 4) if k != 'R2' else v for k, v in test_matrices_log[best_epoch].items()}

            # Log best results to W&B
            wandb.log({
                'learning_rate': learning_rate,
                'max_epochs': num_epochs,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                **{f"test/{k}": v for k, v in best_test_matrices.items()}
            })

            # Additional logging: loss and test metrics per epoch
            for epoch, (train_loss, val_loss) in enumerate(loss_log):
                wandb.log({
                    f'epoch/{epoch + 1}_train_loss': train_loss,
                    f'epoch/{epoch + 1}_val_loss': val_loss,
                    **{f'epoch/{epoch + 1}_test_{k}': v for k, v in test_matrices_log[epoch].items()}
                })

            # Save model only trainable parts z_projection and linear_proj at path with model_name, ablation_tag, seed and month
            best_model_state = model_state_log[best_epoch]
            model.load_state_dict(best_model_state)
            ablation_tag = 'None' if ablation_tag is None else ablation_tag
            traj_encoder = config.encoder if config.encoder is not None else 'None'
            z_projection = model.z_projection.state_dict()
            linear_proj = model.linear_proj.state_dict()
            model_save_path = f"checkpoints/{model_name.replace('/', '_')}_{ablation_tag}_seed{seed}_month{config.month}_encoder{traj_encoder}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'z_projection': z_projection,
                'linear_proj': linear_proj,
                'mean': train_mean,
                'std': train_std,
            }, model_save_path)

            torch.cuda.empty_cache() # Clear all memory

            print(f"Run completed for model: {model_name}, ablation: {ablation_tag}, seed: {seed}, encoder: {traj_encoder}")

    wandb.agent(sweep_id, function=sweep, count=100)

if __name__ == "__main__":
    run_sweep(sweep_config)
