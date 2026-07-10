"""
Training entry-point for the Federated Heterogeneous KG ADR model.

Usage:
    python scripts/train.py --formulation a --data_dir data/raw/ --output_dir results/hfl/
    python scripts/train.py --formulation b --data_dir data/raw/ --output_dir results/vfl/
    python scripts/train.py --model_config configs/ablation/no_edge_feat.yaml --output_dir results/no_edge_feat/
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import torch, yaml

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(path):
    with open(path) as f: return yaml.safe_load(f)

def merge_configs(*cfgs):
    merged = {}
    for c in cfgs: merged.update(c)
    return merged

def build_kg(cfg, data_dir):
    from src.data.ontology_normalizer import OntologyNormalizer
    from src.data.kg_builder import KGBuilder
    norm = OntologyNormalizer(
        rxnorm_path=cfg.get("rxnorm_path"),
        drugbank_path=cfg.get("drugbank_vocab_path"),
        meddra_path=cfg.get("meddra_path"),
        mesh_path=cfg.get("mesh_path"),
    )
    builder = KGBuilder(cfg, norm)
    kg = builder.build(data_dir)
    logger.info(kg.summary())
    return kg

def run_formulation_a(cfg, kg, output_dir, device):
    from src.data.dataset import DrugADRDataModule
    from src.federated.formulation_a import HFLClient, HFLServer, HFLTrainer, LocalKGFragment

    dm = DrugADRDataModule(kg=kg,
        train_frac=cfg["splits"]["train"], val_frac=cfg["splits"]["val"],
        test_frac=cfg["splits"]["test"], batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"])
    logger.info(f"Splits: {dm.split_sizes()}")

    adj = {r: (v[0], v[1]) for r, v in kg.edge_index.items()}
    fed_clients = cfg["federated"]["formulation_a"]["clients"]
    n = len(fed_clients)
    all_pos = dm.train_set.pairs
    chunk = all_pos.size(0) // n

    clients = []
    for i, cc in enumerate(fed_clients):
        fragment = LocalKGFragment(
            h=kg.node_features, adj=adj, edge_feat=kg.edge_attr,
            pos_pairs=all_pos[i*chunk:(i+1)*chunk],
            val_pairs=dm.val_set.pairs[i*(dm.val_set.pairs.size(0)//n):(i+1)*(dm.val_set.pairs.size(0)//n)],
        )
        clients.append(HFLClient(cc["id"], fragment, cfg, device))
        logger.info(f"Client '{cc['id']}': {chunk:,} training pairs")

    server = HFLServer(cfg=cfg, device=device)
    HFLTrainer(server, clients, cfg,
               checkpoint_dir=str(Path(output_dir)/"checkpoints")).train()
    final = str(Path(output_dir)/"best.pt")
    server.save_checkpoint(final, cfg["federated"]["formulation_a"]["aggregation"]["rounds"])
    with open(Path(output_dir)/"splits.json","w") as f: json.dump(dm.split_sizes(), f, indent=2)
    logger.info(f"Saved to {final}")

def run_formulation_b(cfg, kg, output_dir, device):
    from src.data.dataset import DrugADRDataModule
    from src.federated.formulation_b import EntityAligner, VFLClient, VFLServer, VFLTrainer
    import torch

    dm = DrugADRDataModule(kg=kg,
        train_frac=cfg["splits"]["train"], val_frac=cfg["splits"]["val"],
        test_frac=cfg["splits"]["test"], batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"])

    drug_sets, relation_types, clients = {}, [], {}
    for rc in cfg["federated"]["formulation_b"]["clients"]:
        rel = rc["id"]
        relation_types.append(rel)
        ei_key = {"client_adr":"DRUG_CAUSES_AE","client_disease":"ASSOCIATED_WITH",
                  "client_compound":"HAS_TARGET","client_ddi":"DRUG_CAUSES_AE"}.get(rel,"DRUG_CAUSES_AE")
        ei = kg.edge_index.get(ei_key, torch.zeros(2,0,dtype=torch.long))
        drug_sets[rel] = set(ei[0].tolist()) if ei.size(1)>0 else set(range(kg.num_nodes("Drug")))

    aligner = EntityAligner(drug_sets)
    logger.info(f"Shared drugs: {aligner.num_shared_drugs:,}")

    for rc in cfg["federated"]["formulation_b"]["clients"]:
        rel = rc["id"]
        ei_key = {"client_adr":"DRUG_CAUSES_AE","client_disease":"ASSOCIATED_WITH",
                  "client_compound":"HAS_TARGET","client_ddi":"DRUG_CAUSES_AE"}.get(rel,"DRUG_CAUSES_AE")
        ei = kg.edge_index.get(ei_key, torch.zeros(2,0,dtype=torch.long))
        ef = kg.edge_attr.get(ei_key)
        src = ei[0] if ei.size(1)>0 else torch.zeros(0,dtype=torch.long)
        dst = ei[1] if ei.size(1)>0 else torch.zeros(0,dtype=torch.long)
        clients[rel] = VFLClient(rel, rc.get("relation",rel), drug_sets[rel],
            kg.node_features["Drug"], kg.node_features.get("AdverseEffect"),
            (src, dst), ef, cfg, device)

    cfg["num_adrs"] = kg.num_nodes("AdverseEffect")
    server = VFLServer(relation_types, aligner, cfg, device)
    VFLTrainer(server, clients, aligner, dm.train_set.pairs.to(device), cfg,
               checkpoint_dir=str(Path(output_dir)/"checkpoints")).train()
    final = str(Path(output_dir)/"best.pt")
    server.save_checkpoint(final, cfg["federated"]["formulation_b"]["rounds"])
    with open(Path(output_dir)/"splits.json","w") as f: json.dump(dm.split_sizes(), f, indent=2)
    logger.info(f"Saved to {final}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config",   default="configs/model.yaml")
    p.add_argument("--train_config",   default="configs/training.yaml")
    p.add_argument("--fed_config",     default="configs/federated.yaml")
    p.add_argument("--dataset_config", default="configs/dataset.yaml")
    p.add_argument("--data_dir",       default="data/raw/")
    p.add_argument("--output_dir",     default="results/")
    p.add_argument("--formulation",    default="a", choices=["a","b"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = merge_configs(*[load_yaml(getattr(args, k)) for k in
                          ["model_config","train_config","fed_config","dataset_config"]])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device} | Formulation: {args.formulation.upper()}")

    kg = build_kg(cfg, args.data_dir)
    kg.to(device)
    if args.formulation == "a":
        run_formulation_a(cfg, kg, args.output_dir, device)
    else:
        run_formulation_b(cfg, kg, args.output_dir, device)

if __name__ == "__main__":
    main()
