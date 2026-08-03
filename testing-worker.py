import argparse
import time
import uuid

import Pyro5.api
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_CSV = "dataset_phishing_trimmed.csv"
RANDOM_STATE = 42
POLL_INTERVAL = 0.5
N_ESTIMATORS = 100
N_JOBS = 1  # keep at 1-2 when running multiple worker processes on the same machine,
            # otherwise each RandomForest fights the others for all CPU cores.
N_FOLDS = 5  # 5-fold CV instead of a single 80/20 split, so one lucky/unlucky
             # split can't sway which feature "wins" a round.


def load_dataset(path):
    df = pd.read_csv(path)
    y = df["status"]
    X = df.drop(columns=["status"])
    return X, y


def train_and_score(X, y, features):
    """5-fold stratified CV: trains/evaluates 5 times on 5 different splits
    and returns the mean accuracy, so a single lucky/unlucky split can't
    decide which feature wins a round."""
    X_sub = X[features]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_sub, y, cv=cv, scoring="accuracy", n_jobs=1)
    return scores.mean()


def run_worker(worker_id, data_path):
    ns = Pyro5.api.locate_ns("localhost", 9090)  ################# <- IP_address, 9090
    manager = Pyro5.api.Proxy(ns.lookup("fmax.manager"))

    print(f"[Worker {worker_id}] Loading dataset from {data_path}...")
    X, y = load_dataset(data_path)
    print(f"[Worker {worker_id}] Loaded {len(X)} rows, {len(X.columns)} candidate features")

    manager.register_worker(worker_id)
    print(f"[Worker {worker_id}] Registered with manager. Polling for tasks...")

    tasks_done = 0
    while True:
        if manager.is_finished():
            print(f"[Worker {worker_id}] Manager finished. Tasks completed: {tasks_done}. Shutting down.")
            break

        task = manager.get_task(worker_id)
        if task is None:
            time.sleep(POLL_INTERVAL)
            continue

        task_id = task["task_id"]
        round_num = task["round"]
        features = task["features"]

        acc = train_and_score(X, y, features)
        tasks_done += 1
        print(f"[Worker {worker_id}] round={round_num} task={task_id} "
              f"n_features={len(features)} cv_acc(mean of {N_FOLDS} folds)={acc:.4f}")

        acc = float(acc)
        manager.report_result(task_id, worker_id, acc)


def main():
    parser = argparse.ArgumentParser(description="F-Max Worker")
    parser.add_argument("--id", default=None,
                         help="Worker id (default: random worker-XXXXXX)")
    parser.add_argument("--data", default=DATA_CSV,
                         help="Path to the trimmed dataset CSV")
    args = parser.parse_args()

    worker_id = args.id or f"worker-{uuid.uuid4().hex[:6]}"
    run_worker(worker_id, args.data)


if __name__ == "__main__":
    main()