# server.py

import os
import time
import csv
import argparse
import copy
import json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import flwr as fl
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad
from flwr.common import ndarrays_to_parameters
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from torchvision.models import resnet18, resnet50, ResNet50_Weights


# ─── Model factory ───────────────────────────────────────────────────────────────

def get_resnet50_custom(num_classes: int) -> nn.Module:
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for p in base.parameters():
        p.requires_grad = False
    layers = list(base.children())[:-2]
    backbone = nn.Sequential(*layers)
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(base.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return nn.Sequential(backbone, head)


def build_initial_model(class_count: int, model_name: str) -> nn.Module:
    if model_name == "SimpleCNN":
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3,32,3,padding=1)
                self.pool  = nn.MaxPool2d(2,2)
                self.fc1   = nn.Linear(32*64*64, class_count)
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                return self.fc1(x.view(-1,32*64*64))
        return SimpleCNN()

    elif model_name == "ResNet18":
        model = resnet18(weights='IMAGENET1K_V1')
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, class_count),
        )
        return model

    elif model_name == "ResNet50":
        return get_resnet50_custom(class_count)

    else:
        raise ValueError(f"Unknown model {model_name!r}")


# ─── Custom FedAvg strategy with CSV logging ────────────────────────────────────

class CustomFedAvg(FedAvg):
    def __init__(self, model, csv_path, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self._start_times = {}
        self.model = model

    # Note: Flower calls this with keyword `server_round=...`
    def configure_fit(self, server_round, parameters, client_manager):
        # Record start time
        self._start_times[server_round] = time.time()
        # Pass through to FedAvg
        return super().configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager,
        )

    # aggregate_evaluate still gets called with a positional `server_round`
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 0.0, {}

        losses = [res.loss for _, res in results]
        accs   = [res.metrics.get("accuracy", 0.0) for _, res in results]
        loss_agg = float(np.mean(losses))
        acc_agg  = float(np.mean(accs)) if accs else 0.0

        # Compute elapsed time
        start = self._start_times.get(server_round, None)
        training_time = time.time() - start if start is not None else 0.0

        # Append row
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.model,server_round, acc_agg, loss_agg, training_time])

        print(f"Round {server_round}: Loss={loss_agg:.4f}, "
              f"Acc={acc_agg:.4f}, Time={training_time:.1f}s")
        return loss_agg, {"accuracy": acc_agg}


class FedProx(CustomFedAvg):
    """FedProx: FedAvg + proximal term, with CSV logging inherited from CustomFedAvg."""

    def __init__(self, *, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self._global_ndarrays = None  # We'll store ndarrays here

    def aggregate_fit(self, server_round, results, failures):
        params, num_samples = super().aggregate_fit(
            server_round, results, failures
        )

        ndarrays = parameters_to_ndarrays(params)

        if self._global_ndarrays is None:
            self._global_ndarrays = deepcopy(ndarrays)

        proxed = [
            w_new - self.mu * (w_new - w_glob)
            for w_new, w_glob in zip(ndarrays, self._global_ndarrays)
        ]

        self._global_ndarrays = proxed
        proxed_params = ndarrays_to_parameters(proxed)
        return proxed_params, num_samples

class FedSGD(CustomFedAvg):
    """FedSGD: server-side SGD on the *updates* returned by clients."""
    def __init__(self, learning_rate: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.global_ndarrays = None  # will hold a list of np.ndarray

    def aggregate_fit(self, rnd, results, failures):
        #Run FedAvg to get new global parameters (Flower's Parameters object)
        params, num_samples = super().aggregate_fit(rnd, results, failures)

        #Unwrap into a list of numpy arrays
        new_ndarrays = parameters_to_ndarrays(params)

        #First round: just initialize
        if self.global_ndarrays is None:
            # Deep-copy so we don't alias
            self.global_ndarrays = [nd.copy() for nd in new_ndarrays]
            return params, num_samples

        #Compute per-layer “gradient” = new − old, using torch.tensor to handle scalars
        gradients = [
            torch.tensor(new) - torch.tensor(old)
            for new, old in zip(new_ndarrays, self.global_ndarrays)
        ]

        #Average each layer’s gradient
        avg_grads = [g / len(gradients) for g in gradients]

        #Apply server-side SGD update (convert back to numpy)
        updated_ndarrays = [
            old - self.lr * grad.numpy()
            for old, grad in zip(self.global_ndarrays, avg_grads)
        ]

        #Store for next round
        self.global_ndarrays = [nd.copy() for nd in updated_ndarrays]

        #Wrap back into a Parameters object
        updated_params = ndarrays_to_parameters(updated_ndarrays)
        return updated_params, num_samples


# ─── CLI & server startup ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FL Server")
    parser.add_argument("--strategies", nargs="+",
                        choices=["fedavg","fedsgd","fedprox"],
                        default=["fedavg","fedsgd","fedprox"],
                        help="List of strategies to run sequentially")
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--dn", type=str, default="skin_cancer_2024")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--fraction_fit", type=float, default=0.6)
    parser.add_argument("--fraction_evaluate", type=float, default=0.6)
    parser.add_argument("--min_fit_clients", type=int, default=14)
    parser.add_argument("--min_evaluate_clients", type=int, default=13)
    parser.add_argument("--min_available_clients", type=int, default=19)
    parser.add_argument("--m", choices=["SimpleCNN","ResNet18","ResNet50"],
                        default="ResNet50")
    parser.add_argument("--c", type=int, default=7)
    parser.add_argument("--t", type=str, default="IID",
                        help="Subfolder under ./results/ for CSVs")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    base_path = f"./results/{args.t}"
    os.makedirs(base_path, exist_ok=True)

    # Build initial global model once
    global_model = build_initial_model(args.c, args.m)
    weight_list = [val.cpu().numpy() for val in global_model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(weight_list)

    # Common kwargs
    strat_kwargs = dict(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        fraction_evaluate=args.fraction_evaluate,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
    )

    for strat_name in args.strategies:
        print(f"\n===== Running strategy: {strat_name.upper()} =====")
        # CSV per strategy
        csv_path = f"{base_path}/{args.dn}_{strat_name}_evaluation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model","round", "accuracy", "loss", "training_time"])

        # Build strategy instance
        if strat_name == "fedavg":
            strategy = CustomFedAvg(model=args.m,csv_path=csv_path, **strat_kwargs)

        elif strat_name == "fedprox":
            strategy = FedProx(
                model=args.m,
                csv_path=csv_path,
                mu=args.mu,
                initial_parameters=initial_parameters,
                **strat_kwargs,
            )

        elif strat_name == "fedsgd":
            strategy = FedSGD(
                model=args.m,
                csv_path=csv_path,
                initial_parameters=initial_parameters,
                learning_rate=args.lr,
                **strat_kwargs
            )
        else:
            raise ValueError(f"Unknown strategy {strat_name}")

        # Run the server (blocks until completion)
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )

