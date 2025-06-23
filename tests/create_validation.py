#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sSuStaIn
import os
import copy
import warnings
warnings.filterwarnings('ignore')

def initialize_validation(seed):
    biomarker_labels = [f"Biomarker {i+1}" for i in range(4)]
    base_kwargs = {
        "biomarker_labels": biomarker_labels,
        "N_startpoints": 10,
        "N_S_max": 3,
        "N_iterations_MCMC": int(1e4),
        "dataset_name": "test",
        "use_parallel_startpoints": True,
        "seed": seed,
    }

    sEBM_kwargs = {
        "n_stages": 4,
        "stage_size_init": [1, 1, 1, 1],
        "min_stage_size": 1,
        "p_absorb": 0.1,
        "rep_opt": 20,
        "biomarker_labels": biomarker_labels,
        "N_startpoints": 50,
        "N_S_max": 4,
        "N_iterations_MCMC_init": int(1e4),
        "N_iterations_MCMC": int(1e4),
        "N_em": 100,
        "output_folder": None,
        "dataset_name": "test",
        "use_parallel_startpoints": True,
        "seed": 42
    }

    # random ground‐truth subtype assignment
    subtype_fractions = np.array([0.5, 0.30, 0.20])
    rng = np.random.RandomState(seed)
    ground_truth_subtypes = rng.choice(
        range(3), size=500, replace=True, p=subtype_fractions
    ).astype(int)

    return {
        "sustain_kwargs": base_kwargs,
        "sEBM_kwargs": sEBM_kwargs,
        "n_biomarkers": 4,
        "n_samples": 500,
        "n_subtypes": 3,
        "ground_truth_subtypes": ground_truth_subtypes
    }

def create_new_validation(seed, sustain_classes):
    vp = initialize_validation(seed)
    base_kwargs = vp["sustain_kwargs"]
    sEBM_kwargs = vp["sEBM_kwargs"]
    n_biomarkers = vp["n_biomarkers"]
    n_samples = vp["n_samples"]
    n_subtypes = vp["n_subtypes"]
    gt_subtypes = vp["ground_truth_subtypes"]
    subj_ids = list(map(str, np.arange(1, n_samples + 1)))

    for sustain_class in sustain_classes:
        class_name = sustain_class.__name__
        print(f"\nInitializing {class_name}...")

        results_csv = Path.cwd() / f"{class_name}_results.csv"
        if results_csv.exists():
            results_csv.unlink()

        output_folder = Path.cwd() / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Choose the correct kwargs template
        if class_name == "sEBMSustain":
            this_kwargs = copy.deepcopy(vp["sEBM_kwargs"])
        else:
            this_kwargs = copy.deepcopy(vp["sustain_kwargs"])

        this_kwargs["output_folder"] = output_folder

        # Unified call
        sustain_model = sustain_class.test_sustain(
           n_biomarkers, n_samples, n_subtypes,
           gt_subtypes, this_kwargs, seed=seed
        )

        try:
            (samples_sequence, samples_f, ml_subtype,
             prob_ml_subtype, ml_stage, prob_ml_stage,
             prob_subtype_stage) = sustain_model.run_sustain_algorithm()
        finally:
            shutil.rmtree(output_folder, ignore_errors=True)

        df = pd.DataFrame({
            "subj_id": list(map(int, subj_ids)),
            "ml_subtype": ml_subtype.flatten(),
            "prob_ml_subtype": prob_ml_subtype.flatten(),
            "ml_stage": ml_stage.flatten(),
            "prob_ml_stage": prob_ml_stage.flatten()
        })
        df.to_csv(results_csv, index=False)
        print(f"Saved {class_name} results to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create new validation results"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Seed number used for validation"
    )
    parser.add_argument(
        "-c", "--sustainclass", type=str, default=None,
        choices=[i.__name__ for i in sSuStaIn.AbstractSustain.__subclasses__()],
        help="Name of single class to create new validation"
    )
    args = parser.parse_args()

    # Protect from accidental overwrite
    if args.sustainclass:
        target = Path.cwd() / f"{args.sustainclass}_results.csv"
        if target.exists():
            resp = input(f"{target} exists. Override? [y/n] ")
            if resp.lower() != 'y':
                print("Aborting.")
                exit(0)
        cls_map = {i.__name__: i
                   for i in sSuStaIn.AbstractSustain.__subclasses__()}
        classes = [cls_map[args.sustainclass]]
    else:
        classes = sSuStaIn.AbstractSustain.__subclasses__()

    create_new_validation(args.seed, classes)
