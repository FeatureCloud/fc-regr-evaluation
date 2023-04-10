import os
import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from FeatureCloud.app.engine.app import AppState, app_state, Role
from algo import check, create_score_df, aggregate_prediction_errors, compute_local_prediction_error, \
    create_cv_accumulation, plot_boxplots

# Local computations
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'
   
   
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('preprocess', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Read input and config")
        self.read_config()
        splits = self.load('splits')
        
        for split in splits.keys():
            y_test_path = split + "/" + self.load('y_test_filename')
            if self.load('y_test_filename').endswith(".csv"):
                y_test = pd.read_csv(y_test_path, sep=",")
            elif self.load('y_test_filename').endswith(".tsv"):
                y_test = pd.read_csv(y_test_path, sep="\t")
            else:
                y_test = pickle.load(y_test_path)

            y_pred_path = split + "/" + self.load('y_proba_filename')
            if self.load('y_proba_filename').endswith(".csv"):
                y_proba = pd.read_csv(y_pred_path, sep=",")
            elif self.load('y_proba_filename').endswith(".tsv"):
                y_proba = pd.read_csv(y_pred_path, sep="\t")
            else:
                y_proba = pickle.load(y_pred_path)
            y_test, y_pred = check(y_test, y_proba)
            splits[split] = [y_test, y_pred]
        
        return 'preprocess'
            
    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        self.store('sep', ",")
        self.store('split_dir', ".")
        self.store('pred_errors', {})
        self.store('global_errors', {})
        self.store('score_dfs', {})
        splits = {}
        
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_regr_evaluation']
            self.store('y_test_filename', config['input']['y_true'])
            self.store('y_proba_filename', config['input']['y_pred'])
            self.store('split_mode', config['split']['mode'])
            self.store('split_dir', config['split']['dir'])

        if self.load('split_mode') == "directory":
            splits = dict.fromkeys([f.path for f in os.scandir(f"{self.load('INPUT_DIR')}/{self.load('split_dir')}") if f.is_dir()])
        else:
            splits[self.load('INPUT_DIR')] = None

        for split in splits.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)
        shutil.copyfile(self.load('INPUT_DIR') + '/config.yml', self.load('OUTPUT_DIR') + '/config.yml')
        self.log(f'Read config file.')
        
        self.store('splits', splits)


@app_state('preprocess', Role.BOTH)
class PreprocessState(AppState):
    """
    Send computation data to coordinator.
    """

    def register(self):
        self.register_transition('aggregate prediction errors', Role.COORDINATOR)
        self.register_transition('wait for prediction errors', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        pred_errors = self.load('pred_errors')
        splits = self.load('splits')
        for split in splits.keys():
            y_test = splits[split][0]
            y_pred = splits[split][1]
            pred_errors[split] = compute_local_prediction_error(y_test, y_pred)

        data_to_send = jsonpickle.encode(self.load('pred_errors'))
        self.send_data_to_coordinator(data_to_send)
        self.log(f'[CLIENT] Sending computation data to coordinator')
        
        if self.is_coordinator:
            return 'aggregate prediction errors'
        else:
            return 'wait for prediction errors'
 
 
@app_state('wait for prediction errors', Role.PARTICIPANT)
class WaitForPredictionErrorsState(AppState):
    """
    The participant waits until it receives the aggregated prediction errors from the coordinator.
    """

    def register(self):
        self.register_transition('compute scores', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Wait for prediction errors")
        data = self.await_data()
        self.log("[CLIENT] Received aggregated prediction_errors from coordinator.")
        global_errors = jsonpickle.decode(data)
        self.store('global_errors', global_errors)
        return 'compute scores'
        

@app_state('compute scores', Role.BOTH)
class ComputeScoresState(AppState):
    """
    Compute scores.
    """

    def register(self):
        self.register_transition('writing results', Role.BOTH)
        
    def run(self) -> str or None:
        maes = []
        maxs = []
        rmses = []
        mses = []
        medaes = []
        score_dfs = self.load('score_dfs')
        
        for split in self.load('splits').keys():
            score_dfs[split], data = create_score_df(self.load('global_errors')[split])
            maes.append(data[0])
            maxs.append(data[1])
            rmses.append(data[2])
            mses.append(data[3])
            medaes.append(data[4])
        
        if len(self.load('splits').keys()) > 1:
            cv_averages = create_cv_accumulation(maes, maxs, rmses, mses, medaes)
            self.store('cv_averages', cv_averages)
        
        return 'writing results'


@app_state('writing results', Role.BOTH)
class WritingResultsState(AppState):
    """
    Write the results of the aggregated errors.
    """

    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log('[CLIENT] Save results')
        score_dfs = self.load('score_dfs')
        for split in self.load('splits').keys():
            score_dfs[split].to_csv(split.replace("/input", "/output") + "/scores.csv", index=False)

        if len(self.load('splits').keys()) > 1:
            self.load('cv_averages').to_csv(self.load('OUTPUT_DIR') + "/cv_evaluation.csv", index=False)

            self.log("[CLIENT] Plot images")
            plt = plot_boxplots(self.load('cv_averages'), title=f"{len(self.load('splits'))}-fold Cross Validation")

            for format in ["png", "svg", "pdf"]:
                try:
                    plt.write_image(self.load('OUTPUT_DIR') + "/boxplot." + format, format=format, engine="kaleido")
                except Exception as e:
                    print("Could not save plot as " + format + ".")
                    print(e)
        
        self.send_data_to_coordinator('DONE')

        if self.is_coordinator:
            return 'finishing'
        else:
            return 'terminal'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.gather_data()
        self.log("Finishing")
        return 'terminal'


# GLOBAL AGGREGATIONS
@app_state('aggregate prediction errors', Role.COORDINATOR)
class AggregatePredictionErrorsState(AppState):
    """
    The coordinator receives the local computation data from each client and aggregates the prediction errors.
    The coordinator broadcasts the aggregated prediction errors to the clients.
    """
    
    def register(self):
        self.register_transition('compute scores', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Aggregate prediction errors")
        data_incoming = self.gather_data()
        data = [jsonpickle.decode(client_data) for client_data in data_incoming]
        global_errors = self.load('global_errors')
        
        for split in self.load('splits').keys():
            split_data = []
            for client in data:
                split_data.append(client[split])
            global_errors[split] = aggregate_prediction_errors(split_data)
        data_to_broadcast = jsonpickle.encode(self.load('global_errors'))
        self.broadcast_data(data_to_broadcast, send_to_self=False)
        self.log(f'[CLIENT] Broadcasting aggregated prediction errors to clients')
        
        return 'compute scores'
