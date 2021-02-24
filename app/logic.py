import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from app.algo import check, create_score_df, aggregate_prediction_errors, compute_local_prediction_error


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

        self.y_test = None
        self.y_proba = None
        self.global_prediction_errors = None
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
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_regr_evaluation']
            self.y_test_filename = config['files']['y_test']
            self.y_proba_filename = config['files']['y_proba']

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
        state_aggregate_prediction_errors = 4
        state_wait_for_prediction_errors = 5
        state_compute_scores = 6
        state_writing_results = 7
        state_finishing = 8

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
                pred_errors = compute_local_prediction_error(self.y_test, self.y_proba)

                data_to_send = jsonpickle.encode(pred_errors)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_prediction_errors
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_prediction_errors
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_prediction_errors:
                print("[CLIENT] Wait for prediction errors")
                self.progress = 'wait for prediction_errors'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregated prediction_errors from coordinator.")
                    self.global_prediction_errors = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_scores

            if state == state_compute_scores:
                self.score_df = create_score_df(self.global_prediction_errors)
                state = state_writing_results

            if state == state_writing_results:
                print('[CLIENT] Save results')

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

            if state == state_aggregate_prediction_errors:
                print("[CLIENT] Aggregate prediction errors")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    print(data)
                    self.data_incoming = []
                    self.global_prediction_errors = aggregate_prediction_errors(data)
                    data_to_broadcast = jsonpickle.encode(self.global_prediction_errors)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_scores
                    print(f'[CLIENT] Broadcasting aggregated prediction errors to clients', flush=True)

            time.sleep(1)


logic = AppLogic()
