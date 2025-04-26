import time

class ClientLogic:
    def __init__(self, client_metrics_ref, log_dir):
        self.client_metrics_ref = client_metrics_ref
        self.log_dir = log_dir

    def fit(self):
        start_time = time.time()
        # Pass the reference to the shared metrics dictionary
        # save_round_metrics(current_round, self.client_metrics_ref, self.log_dir)

        fit_time = time.time() - start_time
        # ... existing code ... 