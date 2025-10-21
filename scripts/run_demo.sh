#!/usr/bin/env bash
python -m src.demo_synth --out data/demo
python -m src.train_gnn --config config.yaml --data data/demo --out runs/demo_full
python -m src.calibrate --pred runs/demo_full/preds_test.npy --labels runs/demo_full/labels_test.npy --out runs/demo_full/ts.json
python -m src.evaluate --pred runs/demo_full/preds_test.npy --labels runs/demo_full/labels_test.npy --cal runs/demo_full/ts.json --out runs/demo_full/figs
