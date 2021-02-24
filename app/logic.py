import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from app.algo import roc_plot, check, compute_min_max_score, compute_threshold_conf_matrices, compute_roc_parameters, \
    agg_compute_thresholds, aggregate_confusion_matrices, create_score_df, find_nearest, compute_roc_auc


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.y_test_filename = None
        self.y_proba_filename = None
        self.output_format = None
        self.y_test = None
        self.y_proba = None
        self.thresholds = None
        self.confusion_matrix = None
        self.confusion_matrices = None
        self.roc_params = None
        self.roc_auc = None
        self.plt = None
        self.df = None
        self.score_df = None

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_roc']
            self.y_test_filename = config['files']['y_test']
            self.y_proba_filename = config['files']['y_proba']
            self.output_format = config['files']['output_format']

        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocess = 3
        state_aggregate_thresholds = 4
        state_wait_for_thresholds = 5
        state_compute_confusion_matrix = 6
        state_aggregate_confusion_matrices = 7
        state_wait_for_confusion_matrices = 8
        state_compute_roc = 9
        state_writing_results = 10
        state_finishing = 11

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            # Local computations

            if state == state_initializing:
                print("[CLIENT] Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", self.coordinator)

            if state == state_read_input:
                print('[CLIENT] Read input and config')
                self.read_config()
                if self.y_test_filename.endswith(".csv"):
                    self.y_test = pd.read_csv(self.INPUT_DIR + "/" + self.y_test_filename, sep=",")
                elif self.y_test_filename.endswith(".tsv"):
                    self.y_test = pd.read_csv(self.INPUT_DIR + "/" + self.y_test_filename, sep="\t")
                else:
                    self.y_test = pickle.load(self.INPUT_DIR + "/" + self.y_test_filename)

                if self.y_proba_filename.endswith(".csv"):
                    self.y_proba = pd.read_csv(self.INPUT_DIR + "/" + self.y_proba_filename, sep=",")
                elif self.y_proba_filename.endswith(".tsv"):
                    self.y_proba = pd.read_csv(self.INPUT_DIR + "/" + self.y_proba_filename, sep="\t")
                else:
                    self.y_proba = pickle.load(self.INPUT_DIR + "/" + self.y_proba_filename)
                state = state_preprocess

            if state == state_preprocess:
                self.y_test, self.y_proba = check(self.y_test, self.y_proba)
                min_score, max_score = compute_min_max_score(self.y_proba)

                data_to_send = jsonpickle.encode([min_score, max_score])

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_thresholds
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_thresholds
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_thresholds:
                print("[CLIENT] Wait for thresholds")
                self.progress = 'wait for thresholds'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregated thresholds from coordinator.")
                    self.thresholds = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_confusion_matrix

            if state == state_compute_confusion_matrix:
                confusion_matrix = compute_threshold_conf_matrices(self.y_test, self.y_proba, self.thresholds)

                data_to_send = jsonpickle.encode(confusion_matrix)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_confusion_matrices
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_confusion_matrices
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_confusion_matrices:
                print("[CLIENT] Wait for confusion matrix")
                self.progress = 'wait for confusion matrix'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregated confusion matrix from coordinator.")
                    self.confusion_matrices = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_roc

            if state == state_compute_roc:
                print('[CLIENT] Compute roc parameters')
                self.roc_params = compute_roc_parameters(self.confusion_matrices, self.thresholds)
                self.roc_auc = compute_roc_auc(self.roc_params["FPR"], self.roc_params["TPR"])
                idx = find_nearest(self.thresholds, 0.5)
                self.confusion_matrix = self.confusion_matrices[idx]
                self.score_df = create_score_df(self.confusion_matrix, self.roc_auc)
                state = state_writing_results

            if state == state_writing_results:
                print('[CLIENT] Save results')
                plt, df = roc_plot(self.roc_params["FPR"], self.roc_params["TPR"], self.roc_params["THR"])
                plt.savefig(self.OUTPUT_DIR + "/roc." + self.output_format, format=self.output_format)
                df.to_csv(self.OUTPUT_DIR + "/roc.csv", index=False)
                self.score_df.to_csv(self.OUTPUT_DIR + "/scores.csv", index=False)
                state = state_finishing

            if state == state_finishing:
                print("[CLIENT] Finishing")
                self.progress = 'finishing...'
                if self.coordinator:
                    time.sleep(3)
                self.status_finished = True
                break

            # GLOBAL AGGREGATIONS

            if state == state_aggregate_thresholds:
                print("[CLIENT] Aggregate thresholds")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    self.thresholds = agg_compute_thresholds(data)
                    data_to_broadcast = jsonpickle.encode(self.thresholds)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_confusion_matrix
                    print(f'[CLIENT] Broadcasting aggregated thresholds to clients', flush=True)

            if state == state_aggregate_confusion_matrices:
                print("[CLIENT] Aggregate confusion matrices")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    self.confusion_matrices = aggregate_confusion_matrices(data)
                    data_to_broadcast = jsonpickle.encode(self.confusion_matrices)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_roc
                    print(f'[CLIENT] Broadcasting aggregated thresholds to clients', flush=True)

            time.sleep(1)


logic = AppLogic()
