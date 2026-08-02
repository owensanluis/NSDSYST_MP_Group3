import csv
import queue
import threading
import time

import Pyro5.api

ALL_FEATURES = [
    "length_url", "length_hostname", "ip", "https_token", "nb_subdomains",
    "prefix_suffix", "shortening_service", "ratio_digits_url", "phish_hints",
    "nb_hyperlinks", "domain_age", "web_traffic", "google_index",
    "nb_specialchars", "brand_flags", "tld_subdomain_abnormal",
    "nb_redirection_total", "suspicious_html_score",
]

ELBOW_THRESHOLD = 0.005   # 0.5% gain per round counts as "low gain"
STREAK_REQUIRED = 3       # consecutive low-gain rounds needed before locking in the elbow
RESULTS_CSV = "fmax_results.csv"


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class Manager:
    def __init__(self):
        # threading and state
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()

        self.workers = set()

        # task ID counter
        self._next_task_id = 0

        # round state
        self.round_num = 0
        self.round_results = {}
        self.expected_in_round = 0
        self.in_flight = {}

        # running states
        self.current_best = []
        self.best_accuracy = 0.0
        self.results_log = []

        # finalization
        self.finished = False
        self.final_feature_set = None

        # elbow point detection (streak-based, resistant to single noisy dips)
        self.elbow_found = False
        self.elbow_round = None
        self.elbow_features = None
        self.elbow_accuracy = None
        self.low_gain_streak = 0
        self.plateau_start_features = None
        self.plateau_start_accuracy = None

        self._start_round()

    def _start_round(self):
        self.round_num += 1
        remaining = [f for f in ALL_FEATURES if f not in self.current_best]
        if not remaining:
            self._finalize()
            return

        candidates = [self.current_best + [f] for f in remaining]
        self.round_results = {}
        self.expected_in_round = len(candidates)

        for feats in candidates:
            tid = self._next_task_id
            self._next_task_id += 1
            self.in_flight[tid] = feats
            self.task_queue.put((tid, self.round_num, feats))

        print(f"[Manager] Round {self.round_num}: {len(candidates)} candidates queued")

    def _finalize(self):
        self.finished = True
        self.final_feature_set = self.current_best

        if not self.elbow_found:
            # No plateau ever persisted long enough -- fall back to the full set.
            self.elbow_round = self.round_num
            self.elbow_features = list(self.current_best)
            self.elbow_accuracy = self.best_accuracy
            self.results_log.append({
                "round": self.elbow_round, "features": self.elbow_features,
                "accuracy": self.elbow_accuracy, "worker": "ELBOW_POINT",
            })

        print(f"[Manager] All rounds complete. Full feature set: "
              f"{self.final_feature_set} (accuracy={self.best_accuracy:.4f})")
        print(f"[Manager] Elbow point: round {self.elbow_round}, "
              f"features={self.elbow_features}, accuracy={self.elbow_accuracy:.4f}")
        self._write_csv()

    def _write_csv(self):
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "features", "accuracy", "worker"])
            for row in self.results_log:
                w.writerow([row["round"], "|".join(row["features"]),
                            row["accuracy"], row["worker"]])

    def _round_complete(self):
        best_tid = max(
            self.round_results,
            key=lambda tid: self.round_results[tid]["accuracy"],
        )
        best_acc = self.round_results[best_tid]["accuracy"]
        best_feats = self.round_results[best_tid]["features"]

        gain = best_acc - self.best_accuracy
        print(f"[Manager] Round {self.round_num} best: {best_feats} "
              f"acc={best_acc:.4f} (gain={gain:.4f})")

        if not self.elbow_found and self.round_num > 1:
            if gain <= ELBOW_THRESHOLD:
                # First low-gain round in a potential streak: remember where
                # the plateau would have started (i.e. the round BEFORE this
                # one, since that's the last round with a real improvement).
                if self.low_gain_streak == 0:
                    self.plateau_start_features = list(self.current_best)
                    self.plateau_start_accuracy = self.best_accuracy
                self.low_gain_streak += 1
            else:
                # Gain recovered -- this wasn't a real plateau, reset the streak.
                self.low_gain_streak = 0

            if self.low_gain_streak >= STREAK_REQUIRED:
                self.elbow_found = True
                self.elbow_round = self.round_num - self.low_gain_streak
                self.elbow_features = self.plateau_start_features
                self.elbow_accuracy = self.plateau_start_accuracy
                self.results_log.append({
                    "round": self.elbow_round, "features": self.elbow_features,
                    "accuracy": self.elbow_accuracy, "worker": "ELBOW_POINT",
                })
                print(f"[Manager] Elbow point detected: round {self.elbow_round} "
                      f"{self.elbow_features} acc={self.elbow_accuracy:.4f} "
                      f"({self.low_gain_streak} consecutive rounds under "
                      f"{ELBOW_THRESHOLD}). Continuing to run remaining rounds anyway.")

        self.current_best = best_feats
        self.best_accuracy = best_acc
        self._start_round()

    def register_worker(self, worker_id):
        with self.lock:
            self.workers.add(worker_id)
        print(f"[Manager] Worker registered: {worker_id} (total={len(self.workers)})")
        return True

    def get_task(self, worker_id):
        if self.finished:
            return None
        try:
            tid, rnd, feats = self.task_queue.get_nowait()
        except queue.Empty:
            return None
        return {"task_id": tid, "round": rnd, "features": feats}

    def report_result(self, task_id, worker_id, accuracy):
        with self.lock:
            feats = self.in_flight.pop(task_id, None)
            if feats is None:
                return "unknown task_id"
            self.results_log.append({
                "round": self.round_num, "features": feats,
                "accuracy": accuracy, "worker": worker_id,
            })
            self.round_results[task_id] = {"features": feats, "accuracy": accuracy}
            print(f"[Manager] {worker_id} -> {feats} acc={accuracy:.4f}")

            if len(self.round_results) >= self.expected_in_round:
                self._round_complete()
        return "ok"

    def is_finished(self):
        return self.finished

    def get_final_result(self):
        if not self.finished:
            return None
        return {
            "final_features": self.final_feature_set,
            "final_accuracy": self.best_accuracy,
            "elbow_round": self.elbow_round,
            "elbow_features": self.elbow_features,
            "elbow_accuracy": self.elbow_accuracy,
        }


def main():
    daemon = Pyro5.api.Daemon()  ################# <- host="IP_address"
    ns = Pyro5.api.locate_ns()  ################# <- IP_address, 9090
    manager = Manager()
    uri = daemon.register(manager)

    print("Manager URI:", uri)
    ns.register("fmax.manager", uri)
    print(f"[Manager] Ready. Waiting for Workers...")

    try:
        daemon.requestLoop(loopCondition=lambda: not manager.finished)
    finally:
        time.sleep(2)
        ns.remove("fmax.manager")
        print("[Manager] Shut down.")


if __name__ == "__main__":
    main()
