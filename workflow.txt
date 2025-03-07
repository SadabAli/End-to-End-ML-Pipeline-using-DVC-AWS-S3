Building Pipeline:
1> Create a GitHub repo and clone it in local (Add experiments).
2> Add src folder along with all components(run them individually).
3> Add data, models, reports directories to .gitignore file
4> Now git add, commit, push

Setting up dcv pipeline (without params)
5> Create dvc.yaml file and add stages to it.
6> dvc init then do "dvc repro" to test the pipeline automation. (check dvc dag)
7> Now git add, commit, push

Setting up dcv pipeline (with params)
8> add params.yaml file
9> Add the params setup (mentioned below)
10> Do "dvc repro" again to test the pipeline along with the params
11> Now git add, commit, push

Expermients with DVC:
12> pip install dvclive
13> Add the dvclive code block (mentioned below)
14> Do "dvc exp run", it will create a new dvc.yaml(if already not there) and dvclive directory (each run will be considered as an experiment by DVC)
15> Do "dvc exp show" on terminal to see the experiments or use extension on VSCode (install dvc extension)
16> Do "dvc exp remove {exp-name}" to remove exp (optional) | "dvc exp apply {exp-name}" to reproduce prev exp
17> Change params, re-run code (produce new experiments)
18> Now git add, commit, push










-------------------------------------------------------------------------------

params.yaml setup:
1> import yaml
2> add func:
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
3> Add to main():

# data_ingestion
params = load_params(params_path='params.yaml')
test_size = params['data_ingestion']['test_size']

# feature_engineering
params = load_params(params_path='params.yaml')
max_features = params['feature_engineering']['max_features']

# model_building
params = load_params('params.yaml')['model_building']

-------------------------------------------------------------------------------

-------------------------------------------------------------------------------

dvclive code block:
1> import dvclive and yaml:
from dvclive import Live
import yaml
2> Add the load_params function and initiate "params" var in main
3> Add below code block to main:
with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy_score(y_test, y_test))
    live.log_metric('precision', precision_score(y_test, y_test))
    live.log_metric('recall', recall_score(y_test, y_test))

    live.log_params(params)