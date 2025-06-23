###
# ReduXis: Reduced-dimensional X-modality Integrated Stage and Subtype Inference — a lightweight and powerful wrapper around the s-SuStaIn module
# for modeling complex disease progression.
#
# Author: Neel Sarkar
#
# This script was developed as part of the ReduXis project and implements a full custom workflow
# for subtype and stage inference, building upon the sEBMSuStaIn model architecture.
#
# While all code in this file was developed independently, it depends on and interfaces with the
# sEBMSuStaIn module, which in turn extends the pySuStaIn framework.
#
# For code reference, please see the following repositories:
# 1. The original pySuStaIn framework: https://github.com/ucl-pond/pySuStaIn
# 2. The ReduXis project: https://github.com/pathology-dynamics/ReduXis
#
# If you use ReduXis, please cite the following core papers:
# 1. The original SuStaIn paper:    https://doi.org/10.1038/s41467-018-05892-0
# 2. The pySuStaIn software paper:  https://doi.org/10.1016/j.softx.2021.100811
# 3. The s-SuStaIn software paper:  https://pmc.ncbi.nlm.nih.gov/articles/PMC11881980
# 4. The ReduXis software paper:    [DOI or arXiv link when available]
#
# Thank you for using ReduXis.
###

#!/usr/bin/env python3
# Array API is still not in standard support in scipy as of version 1.15.3
import os
os.environ["SCIPY_ARRAY_API"] = "1"

# System libraries - native to Python
import argparse, logging, sys, time, threading, readline, re, shutil, string, random
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import csv

# Arrays and DataFrames
import numpy as np
import numpy.ma as ma
import pandas as pd

# Data viz
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Scikit-learn and third-party packages for classification
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import unique_labels
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Statistics
from scipy.stats import chi2

# SuStaIn-dependent libraries
from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
from sSuStaIn.sEBMSustain import sEBMSustain

# Colorful CLI for the user
from colorama import Fore, Style, init
init(autoreset=True)  # auto-reset styles so no bleeding colors

# Ignore warnings, both from CLI and user and everywhere
import warnings
warnings.filterwarnings('ignore')

# Global variables and sets
CURRENT_PROMPT_LABEL = ""
EXEC_EXT   = {'.py', '.sh', '.exe', '.pl', '.rb', '.bat', '.c', '.cpp'}
IMGVID_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.mov', '.avi'}
ARCH_EXT   = {'.zip', '.tar', '.gz', '.bz2', '.tgz', '.tar.gz'}
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.CYAN
    }

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, "")
        reset = Style.RESET_ALL
        record.levelname = f"{color}[{record.levelname}]{reset}"  # Brackets AND color
        record.msg = f"{color}{record.msg}{reset}"  # Optional: color message too
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(
    fmt="%(asctime)s %(levelname)-12s %(message)s",  # <- Note: 12 chars to fit brackets
    datefmt="%H:%M:%S"
))

logging.basicConfig(level=logging.INFO, handlers=[handler])

# ──────────────────────────────────────────────────────────────────────────────
# Flexible Data Reading
def read_flexible_csv(filepath, **kwargs):
    # Detect delimiter using csv.Sniffer
    with open(filepath, 'r', newline='') as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t|')
            delimiter = dialect.delimiter
        except csv.Error:
            # Fallback to comma if detection fails
            delimiter = ','

    return pd.read_csv(filepath, sep=delimiter, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Validation Helpers
def check_file_format(path: Path, exts=("csv","tsv","psv")):
    if path.suffix.lstrip(".").lower() not in exts:
        logging.error(f"Bad format for {path.name}: must be one of {exts}")
        sys.exit(1)
    logging.info(f"✓ Format of {path.name} OK")

def check_missing_values(df: pd.DataFrame, name="DataFrame", min_fraction_passed=0.10):
    threshold = 0.2
    n_rows = len(df)

    # Identify columns (subjects) exceeding the missing threshold
    col_missing_fraction = df.isna().sum(axis=0) / n_rows
    bad_cols = list(df.columns[col_missing_fraction > threshold])

    if bad_cols:
        preview_cols = bad_cols[:10]
        logging.warning(f"{len(bad_cols)} feature columns exceed the missing data threshold of {threshold * 100:.0f}%."
            f"Consider extensive imputation or exclusion. "
            f"Example affected columns: {preview_cols}..."
        )

    # Drop bad columns (subjects)
    df.drop(columns=bad_cols, inplace=True)

    # Post-drop check
    n_total = len(bad_cols) + len(df.columns)
    n_retained = len(df.columns)

    if n_retained / n_total < min_fraction_passed:
        logging.error(f"Too few columns remain in {name} after filtering ({n_retained}/{n_total}) — exiting.")
        sys.exit(1)

    logging.info(f"✓ Retained {n_retained}/{n_total} columns in {name} after missing value filtering")
    return df

def check_numeric(df: pd.DataFrame, name="DataFrame"):
    nonnum = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum:
        logging.error(f"Non-numeric columns in {name}: {nonnum}")
        sys.exit(1)
    logging.info(f"✓ All columns in {name} numeric")

def check_numeric_range(df: pd.DataFrame, iqr_multiplier=1.5, name="DataFrame"):
    minv = 1e-8 # tiny positive number
    outlier_proportions = []
    df_cleaned = df.copy()
    for col in df.columns:
        series = df_cleaned[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue  # Skip non-numeric columns

        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + iqr_multiplier * iqr # Q3 + 1.5 IQR traditional for outlier

        outliers = (series < minv) | (series > upper)
        outlier_ratio = outliers.sum() / len(series)
        outlier_proportions.append(outlier_ratio)
        if outliers.any():
            df_cleaned[col] = series.clip(lower=minv, upper=upper)

    # Drop all-zero columns after clipping

    zero_cols = df_cleaned.columns[df_cleaned.eq(0).all()]
    if not zero_cols.empty:
        df_cleaned.drop(columns=zero_cols, inplace=True)
        logging.info(f"✓ Dropped {len(zero_cols)} all-zero columns in {name}")

    if outlier_proportions:
        max_outlier_pct = max(outlier_proportions) * 100
        logging.info(f"✓ Maximum outlier proportion in {name} across columns before clipping: {max_outlier_pct:.2f}%")
    else:
        logging.info(f"✓ No outliers detected in {name}")
    return df_cleaned

def check_columns(df: pd.DataFrame, required, name="DataFrame"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error(f"Missing required cols in {name}: {missing}")
        sys.exit(1)
    logging.info(f"✓ Required columns {required} exist in {name}")

# ──────────────────────────────────────────────────────────────────────────────
# Imputation & Scaling
def impute_data(df: pd.DataFrame, n_neighbors=5) -> pd.DataFrame:
    arr = df.T.values
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled = imputer.fit_transform(arr)
    logging.info("✓ Imputation complete")
    return pd.DataFrame(filled, index=df.columns, columns=df.index).T

# ──────────────────────────────────────────────────────────────────────────────────────
# Sample Weighted Feature Selection via Ensemble Feature Importance (With Oversampling)
def run_feature_selection(X: pd.DataFrame, labels: pd.DataFrame, valid_outcomes: list, output_dir: Path, min_features: int, max_features: int, run_selection: bool, min_stability_threshold=0.5):
    outcome_encoder = {v: i for i, v in enumerate(valid_outcomes)}
    y = labels["Outcome"].map(outcome_encoder)
    if y.isnull().any():
        total_missing = y.isnull().sum().sum()
        logging.warning(f"Dropping rows with missing labels: {total_missing}")
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    N_feats = X.shape[1]
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    outcome_weights = dict(zip(classes, weights))

    # Light GBM hyperparameters
    lgbm_common_params = {
        'objective':    'multiclass' if len(classes) > 2 else 'binary',
        'class_weight': outcome_weights,
        'n_estimators': 100,
        'random_state': 42,
        'verbosity':    -1
    }

    if len(classes) > 2:
        lgbm_common_params['num_class'] = len(classes)

    # 3 classifiers for ensemble-based feature selection
    lgbm = LGBMClassifier(**lgbm_common_params)
    rf = RandomForestClassifier(n_estimators=100, class_weight=outcome_weights, random_state=42)
    sgd = SGDClassifier(loss='log_loss', max_iter=1000, class_weight=outcome_weights, random_state=42)
    classifiers = {"LightGBM": lgbm, "Random Forest": rf, "Logistic Regression": sgd}

    # Distance penalty matrix (ordinal accuracy)
    dist_penalty = np.zeros((len(classes), len(classes)), dtype=float)
    for i in range(len(classes)):
        for j in range(len(classes)):
            dist_penalty[i, j] = 1.0 - abs(i - j) / (len(classes) - 1)
    if dist_penalty.max() > 1:
        dist_penalty = dist_penalty / dist_penalty.max()

    # SMOTE instance, to be used inside training only (confusion matrix on test set)
    smote = SMOTE(random_state=42)
    print("")

    if not run_selection:
        logging.info("Feature selection: DISABLED. Using all features...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Balanced train/test with SMOTE oversampling on train only
        X_tr_os, y_tr_os = smote.fit_resample(X_train, y_train)
        lgbm.fit(X_tr_os, y_tr_os)
        y_pred = lgbm.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))

        # Compute sample-aware, ordinal-based accuracy
        row_totals = cm.sum(axis=1)
        row_totals_safe = np.where(row_totals == 0, 1, row_totals)
        cm_normalized = cm / row_totals_safe[:, None]
        reward_mat = np.zeros_like(cm_normalized)
        for i in range(len(classes)):
            for j in range(len(classes)):
                distance = abs(i - j)
                reward = max(0, 1 - (distance / (len(classes) - 1))) if len(classes) > 1 else 1
                reward_mat[i, j] = reward

        weighted_sum = np.sum(cm_normalized * reward_mat * row_totals[:, None])
        total_samples = np.sum(row_totals)
        ordinal_acc = weighted_sum / total_samples
        full_report = classification_report(y_test, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        if ordinal_acc > 0.75:
            logging.info(f"Average balanced ordinal accuracy: {ordinal_acc:.2%}")
            logging.info("Model is highly robust.")
        else:
            logging.warning(f"Average balanced ordinal accuracy: {ordinal_acc:.2%}")
            logging.warning("Model may struggle with disease progression inference.")
        print("")
        final_stability_score = 1.0
        return X, y, full_report, cm, ordinal_acc, final_stability_score, N_feats, N_feats

    # Feature selection path
    logging.info("Feature selection: ENABLED. Implementing ensemble feature importance to prioritize most significant features...")
    feat_counts = defaultdict(int)
    total_runs = 5 * len(classifiers)
    candidate_k = min(max(int(N_feats * 0.05), max_features * 3), N_feats)

    for run in range(5):
        # Get balanced train/test split for this run (train oversampled)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=(42+run))
        X_tr_os, y_tr_os = smote.fit_resample(X_train, y_train)
        for name, clf in classifiers.items():
            clf.fit(X_tr_os, y_tr_os)
            # Get feature importances or coef absolute means
            if hasattr(clf, 'feature_importances_'):
                imps = clf.feature_importances_
            else:
                imps = np.mean(np.abs(clf.coef_), axis=0)

            # Top candidate_k features for this classifier & run
            top_idx = np.argsort(imps)[::-1][:candidate_k]
            for i in top_idx:
                feat_counts[X.columns[i]] += 1

    # Stability: frequency normalized by total runs
    stability = {f: cnt / total_runs for f, cnt in feat_counts.items()}
    sorted_feats = sorted(stability, key=lambda f: stability[f], reverse=True)
    selected = sorted_feats[:max_features]

    # Prune features with low stability below 0.5 but keep at least min_features
    while len(selected) > min_features and stability[selected[-1]] < min_stability_threshold:
        selected.pop()
    final_stability_score = np.mean([stability[f] for f in selected])
    logging.info(f"Features: {N_feats} → {len(selected)} selected")
    logging.info(f"Average stability score after pruning = {final_stability_score:.3f}")

    # Final evaluation on one fresh balanced train/test split with SMOTE oversampling only on training data
    X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X[selected], y, test_size=0.3, stratify=y, random_state=99)
    X_tr_os, y_tr_os = smote.fit_resample(X_red_train, y_red_train)
    lgbm.fit(X_tr_os, y_tr_os)
    y_pred = lgbm.predict(X_red_test)
    cm = confusion_matrix(y_red_test, y_pred, labels=range(len(classes)))

    # Compute sample-aware, ordinal-based accuracy after feature selection as well
    row_totals = cm.sum(axis=1)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    cm_normalized = cm / row_totals_safe[:, None]
    reward_mat = np.zeros_like(cm_normalized)
    for i in range(len(classes)):
        for j in range(len(classes)):
            distance = abs(i - j)
            reward = max(0, 1 - (distance / (len(classes) - 1))) if len(classes) > 1 else 1
            reward_mat[i, j] = reward
    weighted_sum = np.sum(cm_normalized * reward_mat * row_totals[:, None])
    total_samples = np.sum(row_totals)
    ordinal_acc = weighted_sum / total_samples
    reduced_report = classification_report(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")
    if ordinal_acc > 0.75:
        logging.info(f"Average balanced ordinal accuracy: {ordinal_acc:.2%}")
        logging.info("Model is highly robust even after feature reduction.")
    else:
        logging.warning(f"Average balanced ordinal accuracy: {ordinal_acc:.2%}")
        logging.warning("Model may struggle with disease progression inference. Consider loosening pruning criteria to enhance classification accuracy.")
    print("")
    return X[selected], y, reduced_report, cm, ordinal_acc, final_stability_score, N_feats, len(selected)

# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity-Specificity Analysis Helper Function
def run_sensitivity_analysis(X: pd.DataFrame, y: pd.DataFrame, valid_outcomes: list, output_dir: Path, min_features: int, max_features: int):
    # Initialize a list to track results
    thresholds = np.arange(0.4, 0.76, 0.05)  # 0.4 to 0.75 with step size 0.05
    feature_retention = []
    feature_counts = []
    for threshold in thresholds:
        # Run feature selection with the current threshold
        _, _, _, _, _, _, _, selected_feature_count = run_feature_selection(X, y, valid_outcomes, output_dir, min_features, max_features, X.shape[1] > max_features, threshold)
        # Record the proportion of features selected at this threshold
        retention = selected_feature_count / max_features
        logging.info(f"Threshold: {threshold:.2f}, Feature Retention: {retention:.2%}")
        feature_retention.append(retention)
        feature_counts.append(selected_feature_count)

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'lower_thresh': thresholds,
        'final_features': feature_retention
    })

    # Plot Inertia Curve
    optimal_threshold = plot_elbow_curve(df, output_dir, max_features)
    return df, optimal_threshold

# ──────────────────────────────────────────────────────────────────────────────
# Elbow Curve for Sensitivity Analysis
def plot_elbow_curve(df, output_dir, max_features=100):
    df = df.sort_values("lower_thresh")
    x = df["lower_thresh"]
    min_x = x.min() - 0.02
    max_x = x.max() + 0.02
    y = df["final_features"]
    max_y = y.max() + 0.05

    # Detect the "elbow": first noticeable increase in selected features
    dy = y.diff().fillna(0)
    elbow_idx = dy.abs().idxmax()  # Index of max magnitude change

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", color="tab:blue", label="Retention Curve")
    elbow_x = 0
    elbow_y = 1

    if pd.notna(elbow_idx):
        elbow_x = x.loc[elbow_idx]
        elbow_y = y.loc[elbow_idx]

        plt.scatter(elbow_x, elbow_y, color="red", zorder=5,
                    label=f"Elbow @ ({elbow_x:.2f}, {elbow_y:.2f})")

        ax = plt.gca()

        plt.vlines(elbow_x, ymin=0, ymax=elbow_y,
                   colors="red", linestyles="--", alpha=0.7)

        plt.hlines(elbow_y, xmin=min_x, xmax=elbow_x,
                   colors="red", linestyles="--", alpha=0.7)

    plt.title("Sensitivity–Specificity Analysis of Feature Retention")
    plt.xlabel("Minimum Stability Threshold")
    plt.ylabel("Proportion of Features Selected")
    plt.xlim(min_x, max_x)
    plt.ylim(0, max_y)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filename = "elbow_curve.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=600)
    plt.close()
    logging.info(f"Saved elbow curve of various minimum stability thresholds.")
    return elbow_x

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
def read_and_preprocess(data_f: Path, meta_f: Path, outcomes: list):
    # Paths
    for p in (data_f, meta_f):
        check_file_format(p)

    # Load
    X = read_flexible_csv(data_f, index_col=0)
    meta = read_flexible_csv(meta_f)

    # Checks
    check_numeric(X, name="main_df")
    X_filled = check_missing_values(X, name="main_df", min_fraction_passed=0.10)
    X_cleaned = check_numeric_range(X_filled, name="main_df")
    check_columns(meta, ["SubjectID","Outcome"], name="metadata_df")

    # Labels (account for case, ignore all characters except for letters, numbers, and spaces when searching for matching outcomes
    normalized_outcomes = [str(o).lower().strip().replace(" ", "").translate(str.maketrans('', '', string.punctuation)) for o in outcomes]
    meta['_Outcome_norm'] = meta['Outcome'].astype(str).str.lower()
    meta['_Outcome_norm'] = meta['_Outcome_norm'].str.replace(r'[^a-z0-9 ]+', '', regex=True).str.replace(' ', '')
    lab = meta[meta['_Outcome_norm'].isin(normalized_outcomes)][["SubjectID", "Outcome"]].copy()
    lab.set_index("SubjectID", inplace=True)
    meta.drop(columns=['_Outcome_norm'], inplace=True)

    # Impute and scale
    X_imp = impute_data(X_cleaned)
    X_scaled = pd.DataFrame(X_imp, index=X_cleaned.index, columns=X_cleaned.columns)

    # Remove all-zero columns
    X_scaled = X_scaled.loc[:, (X_scaled != 0).any(axis=0)]

    # Defensive check — raise error with guidance if mismatch of indices
    X_scaled.index.name = "SubjectID"
    lab.index.name = "SubjectID"
    common_ids = X_scaled.index.intersection(lab.index)
    if common_ids.empty:
        X_scaled = X_scaled.T  # One-time attempt to recover indices that may be found in rows rather than columns
        common_ids = X_scaled.index.intersection(lab.index)
        raise ValueError(
            f"No matching SubjectIDs between data and metadata!\n"
            f"Check if you set indexes correctly.\n"
            f"Data indices: {X_scaled.index[:5].tolist()}\n"
            f"Metadata indices: {lab.index[:5].tolist()}"
        )

    # Align and remove duplicate rows
    X_scaled = X_scaled.loc[common_ids]
    lab = lab.loc[common_ids]
    X_scaled = X_scaled[~X_scaled.index.duplicated(keep="first")]
    lab = lab[~lab.index.duplicated(keep="first")]

    # Final confirmation
    assert X_scaled.shape[0] == len(lab), (
        f"STILL mismatched: X has {X_scaled.shape[0]} features, but y has {len(lab)} features."
    )
    # Algorithm now able to accept 2, 3, even 4 outcomes in mapping
    normalized_input_outcomes = [o.lower().strip().replace(" ", "") for o in outcomes]
    lab['_Outcome_norm'] = lab['Outcome'].astype(str).str.lower().str.replace(r'[^a-z0-9]+', '', regex=True)

    # Match using normalized strings, return matched values from metadata
    matched_outcomes = lab.loc[lab['_Outcome_norm'].isin(normalized_input_outcomes), 'Outcome'].unique()
    valid_outcomes = list(matched_outcomes)

    # If < 2 outcomes left, crash
    if len(valid_outcomes) < 2:
        logging.error(f"Two valid matching outcomes required! Outcomes passed in: {outcomes}, metadata has: {matched_outcomes}. Try again.")
        sys.exit(1)

    # Clean up
    lab.drop(columns=['_Outcome_norm'], inplace=True)
    return X_scaled, lab, valid_outcomes

# ──────────────────────────────────────────────────────────────────────────────
# Heatmap Plot
def plot_biomarker_stage_heatmap(X: pd.DataFrame, biomarkers: list, outcomes: list, assigned_df: pd.DataFrame, output_dir: Path):
    # Compute counts per (Outcome, Stage)
    count_df = (assigned_df.groupby(["Outcome", "Assigned Stage"]).size().reset_index(name="n_subjects"))

    # Filter to those meeting threshold
    valid_pairs = count_df.loc[count_df["n_subjects"] >= 5, ["Outcome", "Assigned Stage"]] # 5 subjects to avoid sparse stages
    valid_pairs = set(zip(valid_pairs["Outcome"], valid_pairs["Assigned Stage"]))

    # Sort in the desired order of outcomes & increasing stage
    outcome_stage_pairs = sorted(valid_pairs, key=lambda x: (outcomes.index(x[0]), x[1]))
    if not outcome_stage_pairs:
        logging.warning("No outcome-stage pairs meet the minimum count threshold! Exiting plot.")
        return

    # Build heatmap columns & empty DataFrame
    heatmap_cols = [f"{out}\nStage {st}" for out, st in outcome_stage_pairs]
    mean_expr = pd.DataFrame(index=biomarkers, columns=heatmap_cols, dtype=float)

    # Fill with mean expressions
    for (outcome, stage), colname in zip(outcome_stage_pairs, heatmap_cols):
        subj_mask = ((assigned_df["Outcome"] == outcome) & (assigned_df["Assigned Stage"] == stage))
        subjects = assigned_df.loc[subj_mask, "Subject"]
        subset = X.loc[X.index.isin(subjects), biomarkers]

        if subset.empty:
            logging.warning(f"No data for {colname} (shouldn't happen if counted correctly).")
            mean_expr[colname] = np.nan
        else:
            mean_expr[colname] = subset.mean()

    # Sort biomarkers by first outcome’s mean across its valid stages
    first_outcome_cols = [c for c in mean_expr.columns if c.startswith(outcomes[0])]
    mean_expr = mean_expr.loc[mean_expr[first_outcome_cols].mean(axis=1).sort_values(ascending=False).index]

    # Plotting
    fig_width = max(15, len(heatmap_cols) * 1.5)
    fig_height = max(10, len(biomarkers) * 0.4)
    plt.figure(figsize=(fig_width, fig_height))
    safe_data = mean_expr.fillna(0).clip(lower=1e-6)  # avoid LogNorm errors

    ax = sns.heatmap(
        safe_data,
        cmap="bwr",
        norm=colors.LogNorm(),
        annot=True,
        fmt=".3f",
        annot_kws={"size": 12},
        linewidths=0.5,
        cbar_kws={'label': 'Mean Expression'}
    )

    # Colorbar tweaks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Mean Expression', rotation=270, fontsize=16, fontweight='bold', labelpad=20)

    # Titles & labels
    plt.title("Biomarker Expression by Outcome and Stage", fontsize=24, fontweight='bold')
    ax.set_ylabel("Biomarkers", fontsize=16, fontweight='bold')
    ax.set_xlabel("Outcome and Stage", fontsize=18, fontweight='bold')
    ax.xaxis.labelpad = 25
    ax.tick_params(axis='y', labelsize=12)

    # X axis: minor ticks = "Stage X"
    xticks = np.arange(len(outcome_stage_pairs)) + 0.5
    minor_labels = [f"Stage {st}" for (_, st) in outcome_stage_pairs]
    ax.set_xticks([], minor=False)
    ax.set_xticks(xticks, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=12, rotation=0)
    ax.tick_params(axis='x', which='minor', length=4, pad=2)

    # Secondary axis: major ticks = Outcome names
    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_frame_on(False)
    major_positions = []
    major_labels = []
    for out in outcomes:
        # find indices of this outcome in our filtered list
        idxs = [i for i, (o, _) in enumerate(outcome_stage_pairs) if o == out]
        if idxs:
            center = np.mean([xticks[i] for i in idxs])
            major_positions.append(center)
            major_labels.append(out)
    ax2.set_xticks(major_positions)
    ax2.set_xticklabels(major_labels, fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', which='major', length=10, pad=10)

    # Divider lines between outcomes
    for out in outcomes[1:]:
        idxs = [i for i, (o, _) in enumerate(outcome_stage_pairs) if o == out]
        if idxs:
            ax.axvline(idxs[0], color='black', linewidth=4)

    plt.tight_layout()
    output_path = output_dir / "biomarker_stage_heatmap.png"
    plt.savefig(output_path, dpi=600)
    plt.close()
    logging.info("Saved biomarker heatmap of expression levels across outcomes and inferred stages.")

# ──────────────────────────────────────────────────────────────────────────────
# Stage Assignment Bar Plot with Subplots for Subtypes
def plot_outcome_subtype_stage_fractions_subplots(
    y_outcomes,              # array-like: outcome for each subject (Pandas or NumPy)
    outcome_list,            # 1D array: outcomes that were given by the user, not to be confused with y_outcomes (Y-data)
    ml_subtype_assignments,  # 1D array: assigned subtype for each subject (ml_subtype)
    ml_stage_assignments,    # 1D array: assigned stage for each subject (ml_stage)
    num_stages_total,        # int: total number of stages (args.stages)
    output_dir,              # Path object: output directory
):

    y_outcomes = np.asarray(y_outcomes).reshape(-1)
    ml_subtype_assignments = np.asarray(ml_subtype_assignments).reshape(-1)
    ml_stage_assignments = np.asarray(ml_stage_assignments).reshape(-1)

    unique_subtypes = np.unique(ml_subtype_assignments)
    num_subtypes = len(unique_subtypes)

    if num_subtypes < 1:
        logging.info("No subtypes found. Skipping fraction plots.")
        return

    num_outcomes = len(outcome_list)

    if num_outcomes < 2:
        logging.info("No outcomes found. Skipping fraction plots.")
        return

    base_colors = ['#377eb8', '#4daf4a', '#ff7f00']  # blue, green, orange
    cmap = LinearSegmentedColormap.from_list("my_palette", base_colors)
    sampled_colors = [cmap(x) for x in np.linspace(0, 1, num_outcomes)]

    fig_height = 8 * num_subtypes
    fig, axes = plt.subplots(nrows=num_subtypes, ncols=1, figsize=(12, fig_height))

    if num_subtypes == 1:
        axes = [axes]

    bar_width = 0.8 / num_outcomes

    for i, subtype_to_filter_by in enumerate(unique_subtypes):
        subtype_mask = (ml_subtype_assignments == subtype_to_filter_by)
        selected_subjects_stages = ml_stage_assignments[subtype_mask]
        selected_subjects_outcomes = y_outcomes[subtype_mask]

        total_selected_subjects = len(selected_subjects_stages)
        if total_selected_subjects == 0:
            logging.info(f"No subjects found with subtype {subtype_to_filter_by}. Skipping plot for this subtype.")
            continue

        stage_counts_per_outcome = []
        for current_outcome in outcome_list:
            outcome_mask = (selected_subjects_outcomes == current_outcome)
            stages_this_outcome = selected_subjects_stages[outcome_mask]

            stage_counts = np.zeros(num_stages_total, dtype=int)
            unique_stages, counts = np.unique(stages_this_outcome, return_counts=True)
            for stage_val, count in zip(unique_stages, counts):
                stage_idx = int(stage_val) - 1
                stage_counts[stage_idx] += count
            stage_counts_per_outcome.append(stage_counts)

        stage_counts_per_outcome = np.array(stage_counts_per_outcome)
        non_empty_stage_mask = np.any(stage_counts_per_outcome > 0, axis=0)
        num_display_stages = np.sum(non_empty_stage_mask)

        if num_display_stages == 0:
            logging.warning(f"All stages empty for Subtype {int(subtype_to_filter_by)}. Skipping.")
            continue

        filtered_counts = stage_counts_per_outcome[:, non_empty_stage_mask]
        num_display_stages = filtered_counts.shape[1]

        # Re-label the stages sequentially
        stage_labels = [f"Stage {i + 1}" for i in range(num_display_stages)]
        ind = np.arange(num_display_stages)

        for k, current_outcome in enumerate(outcome_list):
            total_subjects_for_outcome = np.sum(selected_subjects_outcomes == current_outcome)
            counts = filtered_counts[k]
            fractions = counts / total_subjects_for_outcome if total_subjects_for_outcome > 0 else np.zeros_like(counts)

            offset = (k - (num_outcomes - 1) / 2) * bar_width
            x_positions = ind + offset

            bars = axes[i].bar(
                x_positions,
                fractions,
                width=bar_width,
                color=sampled_colors[k],
                edgecolor='black',
                label=f'{current_outcome}'
            )

            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2.0, yval + 0.01,
                             f'{yval:.2f}\n({counts[bar_idx]}/{total_subjects_for_outcome})',
                             ha='center', va='bottom', fontsize=14, rotation=90)


        axes[i].set_xticks(ind)
        axes[i].set_xticklabels(stage_labels, rotation=0, ha='center', fontsize=14)
        axes[i].set_title(f"Subtype {int(subtype_to_filter_by)}", fontsize=20, fontweight='semibold')
        axes[i].set_ylim(0, 1.0)
        axes[i].set_xlabel("Disease Stage", fontsize=18)
        axes[i].set_ylabel("Fraction of Subjects", fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].legend(title="Outcomes", title_fontsize=16, fontsize=14, loc='upper right', bbox_to_anchor=(1.25, 1))

    fig.suptitle(f"Fraction of Subjects per Stage by Subtype and Outcome", fontsize=24, y=1.02, fontweight='bold')
    fig.subplots_adjust(hspace=0.2)
    plot_filename = Path(output_dir) / f"fraction_subjects_by_subtype_stage_outcome.png"
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # reserve space below the title
    plt.savefig(plot_filename, dpi=600, bbox_inches='tight', pad_inches=0.2)
    logging.info("Saved grouped bar chart detailing fraction of subjects within inferred stages, subtypes, and outcomes.")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Classification Report Generation
def save_confusion_matrix(cm, class_labels, out_dir: Path):
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels)
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # Custom colormaps
    green_cmap = sns.light_palette("green", as_cmap=True) # outcomes match
    red_cmap = sns.light_palette("red", as_cmap=True) # outcomes are as far away from each other as possible
    blue_cmap = sns.light_palette("yellow", as_cmap=True) # intermediate, outcomes sort of matched

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_normalized[i, j]
            if i == j:
                color = green_cmap(val)
            elif abs(i - j) == 1 and cm.shape[0] > 2: # neighbors, let's say close = not as severe a penalty as far away
                color = blue_cmap(val)
            else:
                color = red_cmap(val)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            # Adaptive text color based on brightness
            r, g, b = color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=14)

    ax.set_xticks(np.arange(len(class_labels)) + 0.5)
    ax.set_yticks(np.arange(len(class_labels)) + 0.5)
    ax.set_xticklabels(class_labels, rotation=0, ha="center")
    ax.set_yticklabels(class_labels, rotation=0)
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("True Outcome")
    ax.set_title("Confusion Matrix")
    ax.set_xlim(0, len(class_labels))
    ax.set_ylim(len(class_labels), 0)  # Reverse y-axis to have first true label at top
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.9) # Give some room for the titles and labels
    filepath = Path(out_dir) / "visualized_confusion_matrix.png"
    plt.savefig(filepath, dpi=600, bbox_inches='tight', pad_inches=0.2)
    logging.info(f"Saved confusion matrix.")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Classification Report Helper
def save_classification_report(report, cm, acc, class_labels, initial_features, final_features, final_stability_score, out_dir: Path):
    filepath = Path(out_dir) / "general_classification_report.txt"
    with open(filepath, "w") as f:
        f.write("=== Confusion Matrix ===\n")
        f.write(f"{cm}\n\n")

        f.write("=== Balanced Ordinal Accuracy ===\n")
        f.write(f"{acc:.2%}\n\n")

        f.write("=== Feature Selection ===\n")
        f.write(f"Initial Number of Features Detected: {initial_features}\n")
        f.write(f"Final Number of Features Retained: {final_features}\n")
        f.write(f"Final Feature Selection Stability Score: {final_stability_score:.3f}")
        if final_stability_score > 0.7:
            f.write(" (Highly Stable, Highly Consistent Model)\n")
        elif final_stability_score > 0.5:
            f.write(" (Mostly Stable, Slightly Inconsistent Model)\n")
        else:
            f.write(" (Highly Unstable, Highly Inconsistent Model)\n")
        f.write("\n")

        f.write("=== Detailed Classification Report ===\n")
        f.write(report)

    logging.info(f"Saved classification report.")

# ──────────────────────────────────────────────────────────────────────────────
# Chi-Squared Analysis, Nested and Stratified
def perform_chi_squared_analysis(assigned_df, output_dir):
    # Unique levels
    stages   = assigned_df['Assigned Stage'].unique()
    subtypes = assigned_df['Assigned Subtype'].unique()
    outcomes = assigned_df['Outcome'].unique()

    # Tally observed counts and totals
    observed = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals   = defaultdict(lambda: defaultdict(int))
    for _, row in assigned_df.iterrows():
        st, out, stage = row['Assigned Subtype'], row['Outcome'], row['Assigned Stage']
        observed[st][out][stage] += 1
        totals[st][out] += 1

    # Compute expected (uniform)
    expected = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for st in subtypes:
        for out in outcomes:
            tot = totals[st][out]
            for stage in stages:
                expected[st][out][stage] = (tot / len(stages)) if tot > 0 else 0.0

    # Compute max outcome‐label length
    outcome_indent = 4
    all_labels = []
    for st in subtypes:
        for out in outcomes:
            lbl = " " * outcome_indent + f'Outcome "{out}" (N = {totals[st][out]})' + ":"
            all_labels.append(lbl)
    max_label_len = max(len(lbl) for lbl in all_labels)
    stage_indent = max_label_len

    # Determine column width to fit any Stage X or number + extra padding
    num_strs = []
    for st in subtypes:
        for out in outcomes:
            for stage in stages:
                num_strs.append(str(int(round(observed[st][out][stage]))))
                num_strs.append(f"{expected[st][out][stage]:.2f}")
    max_num_len = max(len(s) for s in num_strs)
    max_stage_label_len = max(len(f"Stage {s}") for s in stages)
    stage_col_width = max(max_num_len, max_stage_label_len) + 10  # extra breathing room

    # Helpers
    def center_val(s):
        return str(s).center(stage_col_width)
    def sig_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    def fmt_p(p):
        if p < 1e-300: return "< 10^-300"
        if np.isnan(p): return "nan"
        return f"{p:.3e}"

    total_width = stage_indent + stage_col_width * len(stages)

    # Build the lines
    lines = []
    # Title centered
    title = "CHI-SQUARED GOODNESS OF FIT TEST (Stratified Expected Counts)"
    lines.append(" " * ((total_width - len(title)) // 2) + title)

    for st in subtypes:
        # Subtype header
        lines.append(f"Subtype {st} (N = {sum(totals[st].values())}):")
        # Stage headers
        hdr = " " * stage_indent + "".join(f"Stage {s}".center(stage_col_width) for s in stages)
        lines.append(hdr)

        # Per‐outcome blocks
        for idx, out in enumerate(outcomes):
            if idx > 0:
                lines.append("")  # blank line between outcomes
            lines.append(" " * outcome_indent + f'Outcome "{out}" (N = {totals[st][out]})' + ":")
            # Observed row
            obs_vals = [observed[st][out][s] for s in stages]
            obs_line = " " * (outcome_indent + 4) + "Observed:".ljust(max_label_len - outcome_indent - 4) \
                       + "".join(center_val(v) for v in obs_vals)
            lines.append(obs_line)
            # Expected row
            exp_vals = [expected[st][out][s] for s in stages]
            exp_line = " " * (outcome_indent + 4) + "Expected:".ljust(max_label_len - outcome_indent - 4) \
                       + "".join(center_val(f"{v:.2f}") for v in exp_vals)
            lines.append(exp_line)

        # Append subtype χ² on last Expected line
        obs_flat = [observed[st][o][s] for o in outcomes for s in stages]
        exp_flat = [expected[st][o][s] for o in outcomes for s in stages]
        df       = (len(outcomes) - 1) * (len(stages) - 1)
        chi2_stat= np.sum((np.array(obs_flat) - np.array(exp_flat))**2 / (np.array(exp_flat) + 1e-10))
        p_val    = chi2.sf(chi2_stat, df)
        lines[-1] += " " * 4 + f"(χ^2 = {chi2_stat:.4f}, df = {df}, P = {fmt_p(p_val)} ({sig_stars(p_val)}))"

        # Subtype separator
        lines.append("-" * total_width)
        lines.append("")

    # Global statistics
    all_obs = [observed[st][o][s] for st in subtypes for o in outcomes for s in stages]
    all_exp = [expected[st][o][s] for st in subtypes for o in outcomes for s in stages]
    gdf     = max(1, len(subtypes) - 1) * (len(outcomes) - 1) * (len(stages) - 1)
    gchi    = np.sum((np.array(all_obs) - np.array(all_exp))**2 / (np.array(all_exp) + 1e-10))
    gp      = chi2.sf(gchi, gdf)
    lines.append(f"Global statistics:  χ^2 = {gchi:.4f}, df = {gdf}, P = {fmt_p(gp)} ({sig_stars(gp)})")

    # Print & save
    output = "\n".join(lines)
    print(output)
    output_path = Path(output_dir) / "chi_squared_nested_analysis.txt"
    with open(output_path, "w") as f:
        f.write(output)
    logging.info(f"Saved chi-squared nested analysis.")

# SuStaIn Orchestration
def run_sustain_pipeline(args):
    data_f = Path(args.data_file).expanduser()
    metadata_f = Path(args.metadata_file).expanduser()
    X_preprocessed, lab, valid_outcomes = read_and_preprocess(data_f, metadata_f, args.outcomes)

    # Fix and/or limit cluster size, number of stages and subtypes BEFORE we run the computationally expensive SuStaIn algorithm
    # For minimum cluster size, we expect to see at most around 20 different biomarkers that are deterministic of stage
    print("")
    if args.min_clust_size < 5 or args.min_clust_size > 20:
        logging.warning("Number of biomarkers per stage should be between 5 and 20 for biologically realistic modeling. Adjusting automatically.")
    args.min_clust_size = min(max(args.min_clust_size, 3), 20)
    logging.info(f"Cluster size validation complete. Cluster size inputted: {args.min_clust_size}")

    # General rule of thumb is to assume maximum sparsity.
    # To be conservative, around 30 subjects per stage per subtype on average should be considered too sparse
    # In most clinical and biomedical contexts, having more than 5 subtypes is unheard of, so having a hard limit of 5 subtypes is reasonable
    min_subtypes = 1
    n_subjects = X_preprocessed.shape[0] # Rows = subjects
    max_subtypes = min(5, n_subjects // (args.stages * 10))
    corrected_subtypes = min(max(args.subtypes, min_subtypes), max_subtypes)
    if args.subtypes < min_subtypes or args.subtypes > max_subtypes:
        logging.warning("Automatically constraining the number of progression patterns to between 1 and 5 for robust and consistent disease modeling.")
    args.subtypes = corrected_subtypes
    logging.info(f"Subtype validation complete. Total number of subtypes inputted: {args.subtypes}")

    max_possible_stages = 6
    min_reasonable_clust_size = 5
    optimal_threshold = 0.5  # default, may tune depending on specific analysis
    min_features = max_possible_stages * args.min_clust_size
    do_selection = X_preprocessed.shape[1] > args.max_features
    if args.run_analysis:
        if not do_selection:
            logging.info("Feature count already less than specified maximum number of features. Feature selection not necessary. Skipping sensitivity analysis.")
        else:
            min_features_broadened = max_possible_stages * min_reasonable_clust_size # broaden to increase range of feature counts for sensitivity analysis
            analysis, optimal_threshold = run_sensitivity_analysis(X_preprocessed, lab, valid_outcomes, Path(args.output_dir), min_features_broadened, args.max_features)
    logging.info(f"Optimal minimum stability threshold selected: {optimal_threshold}")
    min_features = max_possible_stages * args.min_clust_size
    X_reduced, y, report, cm, ordinal_acc, final_stability_score, initial_features, final_features = run_feature_selection(X_preprocessed, lab, valid_outcomes, Path(args.output_dir), min_features, args.max_features, do_selection, optimal_threshold)
    save_confusion_matrix(cm, valid_outcomes, args.output_dir)
    save_classification_report(report, cm, ordinal_acc, valid_outcomes, initial_features, final_features, final_stability_score, args.output_dir)
    n_features = X_reduced.shape[1] # Cols = biomarkers
    max_stages = min(6, n_features // args.min_clust_size)
    if args.stages < 2 or args.stages > max_stages:
        logging.warning(f"Recommended number of stages for disease progression modeling is between 2 and {max_stages}. Adjusting automatically.")
    corrected_stages = min(max(args.stages, 2), max_stages)
    args.stages = corrected_stages
    logging.info(f"Stage validation complete. Total number of stages inputted: {args.stages}")
    print("")
    min_features = args.min_clust_size * args.stages

    # Ensure the total number of biomarkers in all stages matches the number of biomarkers in reduced set
    base_clust_size = n_features // args.stages
    slack = n_features - (base_clust_size * args.stages)
    bm_clusters = [base_clust_size] * args.stages
    while slack > 0:
        idx = random.randint(0, args.stages - 1)
        bm_clusters[idx] += 1
        slack -= 1
    random.shuffle(bm_clusters)

    # Extract numpy array of X for mixture model, and biomarker labels from X
    X_np = X_reduced.to_numpy().astype(float)

    # Mixture models
    mm_fn = fit_all_kde_models if args.mixture_type == "kde" else fit_all_gmm_models
    mixtures = mm_fn(X_np, np.array(y))

    # Likelihoods
    L_no = np.zeros_like(X_np)
    L_yes = np.zeros_like(X_np)
    for i in range(X_np.shape[1]):
        if args.mixture_type == "gmm":
            L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, X_np[:, i])
        else:
            L_no[:, i], L_yes[:, i] = mixtures[i].pdf(X_np[:, i].reshape(-1, 1))

    process_L = lambda L, mv=0: ma.masked_less_equal(L, mv).filled(fill_value=ma.masked_less_equal(L, mv).min(axis=0))
    L_no, L_yes = process_L(L_no), process_L(L_yes)
    bm_labels = X_reduced.columns.tolist()

    # s-SuStaIn params
    sustain = sEBMSustain(
        L_yes, L_no,
        args.stages, bm_clusters,
        args.min_clust_size, args.p_absorb,
        args.rep, bm_labels,
        args.N_startpoints, args.subtypes,
        args.N_iter_init, args.N_iter,
        args.N_em, args.output_dir,
        args.dataset_name, args.parallel
    )

    # Only clears everything in the "pickle_files" subdirectory
    pickle_dir = Path(args.output_dir) / "pickle_files"
    if pickle_dir.exists() and pickle_dir.is_dir():
        for item in pickle_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)
    marker_sheet_output_path = Path(args.output_dir) / "assigned_subjects.tsv"

    # Initialize data structures
    rows = []
    subtype_counts = defaultdict(int)
    rows_by_subtype = defaultdict(list)

    # Determine if ml_stage is 0-based
    all_stage_values = [int(m[0]) if isinstance(m, (np.ndarray, list)) else int(m) for m in ml_stage]
    stage_offset = 1 if min(all_stage_values) == 0 else 0

    for subj, ms, ps, mt, pt in zip(X_reduced.index, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage):
        outcome = lab.loc[subj, "Outcome"]
        ms = min(max(int(ms[0]) + 1 if isinstance(ms, np.ndarray) else int(ms) + 1, 1), args.subtypes) # Add 1 for all subtype assignments universally

        # Apply globally-determined stage offset
        mt_val = int(mt[0]) if isinstance(mt, (np.ndarray, list)) else int(mt)
        mt = min(max(mt_val + stage_offset, 1), args.stages)
        rows.append((subj, outcome, ms, mt))
        subtype_counts[ms] += 1
        rows_by_subtype[ms].append((subj, outcome, ms, mt))

    # Step 2: Warn the user about sparse subtypes
    REQUIRED_SUBJECTS = 0.10 * n_subjects
    for subtype, count in subtype_counts.items():
        if count < REQUIRED_SUBJECTS:
            logging.warning(f"Subtype {subtype} has only {count} subjects, when {int(REQUIRED_SUBJECTS)} are recommended to have. Consider reducing the number of subtypes requested.")

    # Step 3: Sort the cleaned rows
    outcome_order = {outcome: i for i, outcome in enumerate(valid_outcomes)}
    rows.sort(key=lambda x: (outcome_order.get(x[1], float('inf')), x[2], x[3]))

    # Step 4: Save cleaned rows
    with open(marker_sheet_output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Subject", "Outcome", "Assigned Subtype", "Assigned Stage"])
        writer.writerows(rows)
    print("")
    logging.info("Exported subject stage and subtype assignments as a structured table.")

    # Supervised analysis of s-SuStaIn's inference - plot the distribution of subjects and features based on outcome and stage
    assigned_df = read_flexible_csv(marker_sheet_output_path)
    cleaned_outcomes = assigned_df["Outcome"].to_numpy()
    cleaned_ml_subtype = assigned_df["Assigned Subtype"].to_numpy()
    cleaned_ml_stage   = assigned_df["Assigned Stage"].to_numpy()

    plot_outcome_subtype_stage_fractions_subplots(
        y_outcomes=cleaned_outcomes,
        outcome_list=valid_outcomes,
        ml_subtype_assignments=cleaned_ml_subtype,
        ml_stage_assignments=cleaned_ml_stage,
        num_stages_total=args.stages,
        output_dir=Path(args.output_dir),
    )

    plot_biomarker_stage_heatmap(X_reduced, bm_labels, valid_outcomes, assigned_df, Path(args.output_dir))

    # Quantify the separability of the subjects, using goodness-of-fit test, stratify by outcome and subtype for best results
    perform_chi_squared_analysis(assigned_df, Path(args.output_dir))

# ──────────────────────────────────────────────────────────────────────────────
# Interactive CLI
class RichParser(argparse.ArgumentParser):
    def error(self, message):
        # On any parsing error: show banner, error, then help
        print_welcome()
        logging.error(f"{message}")
        self.print_help()
        sys.exit(2)

# ──────────────────────────────────────────────────────────────────────────────
# ANSI Handler for Computing Space Separation
def visible_len(s: str) -> int:
    """Length of string without ANSI color codes."""
    return len(ANSI_ESCAPE.sub('', s))

# ──────────────────────────────────────────────────────────────────────────────
# Colorizer
def colorize(name: str, base_path: str) -> str:
    """Colorize filename or directory with trailing slash for dirs."""
    full_path = os.path.expanduser(os.path.join(base_path, name))
    is_dir = os.path.isdir(full_path)
    base = os.path.basename(name.rstrip(os.sep))

    # Add trailing slash for directories for clarity & colorize blue
    if is_dir:
        display_name = name.rstrip(os.sep) + os.sep
        return f"{Style.BRIGHT}{Fore.BLUE}{display_name}{Style.RESET_ALL}"
    ext = os.path.splitext(base)[1].lower()
    if ext in EXEC_EXT or os.access(full_path, os.X_OK):
        return f"{Style.BRIGHT}{Fore.GREEN}{name}{Style.RESET_ALL}"
    if ext in IMGVID_EXT:
        return f"{Style.BRIGHT}{Fore.MAGENTA}{name}{Style.RESET_ALL}"
    if ext in ARCH_EXT:
        return f"{Style.BRIGHT}{Fore.RED}{name}{Style.RESET_ALL}"

    return name  # default no color

# ──────────────────────────────────────────────────────────────────────────────
# ANSI Display Formatter
def print_suggestions_columnwise(matches, base_dir, max_width=120, spacing=10):
    global CURRENT_PROMPT_LABEL

    # Prepare all items with ANSI
    colored_entries = [colorize(m, base_dir) for m in matches]
    raw_entries = [m for m in matches]

    # Compute max visible length for each column
    visible_lengths = [visible_len(s) for s in colored_entries]
    col_width = max(visible_lengths) + spacing

    cols = max_width // col_width
    if cols < 1:
        cols = 1

    # Print all in aligned rows
    print("\n")
    for i in range(0, len(colored_entries), cols):
        row = colored_entries[i:i+cols]
        padded = [s + ' ' * (col_width - visible_len(s)) for s in row]
        print("".join(padded))
    print()

    # Restore prompt and buffer
    buffer = readline.get_line_buffer()
    prompt = f"Please enter {CURRENT_PROMPT_LABEL}: "
    print(f"{Fore.MAGENTA}{prompt}{Style.RESET_ALL}{buffer}", end='', flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Common Prefix Handler
def common_prefix(strings):
    if not strings:
        return ''
    prefix = os.path.commonprefix(strings)
    return prefix

# ──────────────────────────────────────────────────────────────────────────────
# Path Completer and Special File Path Handler
def path_completer(text, state):
    buffer = readline.get_line_buffer()
    expanded = os.path.expanduser(buffer.strip() or ".")

    if buffer in ("", ".", "./"):
        base_dir = os.getcwd()
        prefix = ""
    elif buffer in ("~", "~/"):
        base_dir = os.path.expanduser("~")
        prefix = ""
    elif buffer.endswith("/"):
        base_dir = os.path.expanduser(buffer)
        prefix = ""
    else:
        base_dir = os.path.dirname(expanded)
        prefix = os.path.basename(expanded)

    try:
        entries = sorted(os.listdir(base_dir))

        if CURRENT_PROMPT_LABEL == "output directory":
            entries = [e for e in entries if os.path.isdir(os.path.join(base_dir, e))]

        matches = [e for e in entries if e.startswith(prefix)]

        if not matches:
            return None

        if len(matches) == 1:
            match = matches[0]
            suffix = match[len(prefix):]
            if state == 0 and suffix:
                readline.insert_text(suffix)
                readline.redisplay()
            return None

        if state == 0:
            # Handle partial common prefix autocompletion
            common = common_prefix(matches)
            if common and common != prefix:
                suffix = common[len(prefix):]
                readline.insert_text(suffix)
                readline.redisplay()
                return None

            # Otherwise, just display options
            print_suggestions_columnwise(matches, base_dir)
            readline.redisplay()
        return None

    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Tab Completed Interactive Prompt
def interactive_prompt(label, use_tab_completion=False, custom_prompt=None):
    global CURRENT_PROMPT_LABEL
    CURRENT_PROMPT_LABEL = label  # set global prompt label

    # Set completer only if tab completion is explicitly enabled
    readline.set_completer(None) # default behavior is to do nothing
    readline.parse_and_bind('')

    if use_tab_completion:
        readline.set_completer_delims('\t\n')
        readline.set_completer(path_completer)
        readline.parse_and_bind("tab: complete")
    else:
        readline.set_completer(None)
        readline.parse_and_bind('"\t": ""')

    try:
        prompt_text = f"{custom_prompt} {Style.RESET_ALL}" if custom_prompt else f"{Fore.MAGENTA}Please enter {label}: {Style.RESET_ALL}"
        return input(prompt_text)
    except KeyboardInterrupt:
        print("\n")
        logging.error(f"{Fore.RED}Aborted.{Style.RESET_ALL}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Welcome Screen
def print_welcome():
    banner = rf"""{Style.BRIGHT}
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                         ____          _      __  ___                           ║
    ║                        |  _ \ ___  __| |_   _\ \/ (_)___                       ║
    ║                        | |_) / _ \/ _` | | | |\  /| / __|                      ║
    ║                        |  _ <  __/ (_| | |_| |/  \| \__ \                      ║
    ║                        |_| \_\___|\__,_|\__,_/_/\_\_|___/                      ║
    ║                                                                                ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    {Fore.GREEN}║        🚀 Welcome to ReduXis: Ensemble Feature Reduction + s-SuStaIn 🚀        ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    {Fore.YELLOW}║                       📥 Input: Expression Data + Metadata 📥                  ║
    ║  📤 Output: Subtype + Stage Assignments, Chi-Squared + Sensitivity Analysis 📤 ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    {Fore.BLUE}║  💡 Purpose: Infer disease progression patterns in your bioinformatics data 💡 ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    {Fore.RED}║   ⚠️  Warning: For robust modeling, avoid including variant-level features ⚠️    ║
    ║    Examples of such features may include:                                      ║
    ║        - SNPs (Single Nucleotide Polymorphisms)                                ║
    ║        - INDELs (Insertions/Deletions)                                         ║
    ║        - Mutant peptides (e.g., KRAS_G12D, BRAF_V600E, APOE_C112R)             ║
    ║        - Allele-specific isoforms or transcripts                               ║
    ║    These may introduce spurious subtypes or confound temporal inference.       ║
    ║    Consider harmonizing to canonical forms or collapsing variant annotations.  ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    {Style.RESET_ALL}"""
    print(banner)


# ──────────────────────────────────────────────────────────────────────────────
# Help Screen
def print_help():

    # Interactive user guide
    print(f"""{Style.BRIGHT}{Fore.GREEN}⚙️  REQUIRED ARGUMENTS:{Style.RESET_ALL}
  {Fore.GREEN}--data-file{Style.RESET_ALL}          Path to data file (e.g., RNA counts)         {Fore.RED}(REQUIRED){Style.RESET_ALL}
  {Fore.GREEN}--metadata-file{Style.RESET_ALL}      Path to metadata file                        {Fore.RED}(REQUIRED){Style.RESET_ALL}
  {Fore.GREEN}--outcomes{Style.RESET_ALL}           Outcomes (e.g., Normal Tumor)                {Fore.RED}(REQUIRED){Style.RESET_ALL}
  {Fore.GREEN}--dataset-name{Style.RESET_ALL}       Dataset name (e.g., TCGA)                    {Fore.RED}(REQUIRED){Style.RESET_ALL}

{Style.BRIGHT}{Fore.CYAN}⚙️  OPTIONAL ARGUMENTS:{Style.RESET_ALL}
  {Fore.CYAN}-h{Style.RESET_ALL}, {Fore.CYAN}--help{Style.RESET_ALL}           Show this help message and exit
  {Fore.CYAN}-v{Style.RESET_ALL}, {Fore.CYAN}--version{Style.RESET_ALL}         Show the version number of ReduXis and exit
  {Fore.CYAN}--output-dir{Style.RESET_ALL}         Output directory
  {Fore.CYAN}--stages{Style.RESET_ALL}             Number of disease stages
  {Fore.CYAN}--subtypes{Style.RESET_ALL}           Number of subtypes
  {Fore.CYAN}--max-features{Style.RESET_ALL}       Maximum number of features to retain
  {Fore.CYAN}--N_startpoints{Style.RESET_ALL}      Startpoints for optimization
  {Fore.CYAN}--rep{Style.RESET_ALL}                Repetitions for stability
  {Fore.CYAN}--min-clust-size{Style.RESET_ALL}     Minimum cluster size
  {Fore.CYAN}--p-absorb{Style.RESET_ALL}           Probability to absorb feature
  {Fore.CYAN}--N-iter-init{Style.RESET_ALL}        Iterations (initial phase)
  {Fore.CYAN}--N-iter{Style.RESET_ALL}             Iterations (main phase)
  {Fore.CYAN}--N-em{Style.RESET_ALL}               EM steps
  {Fore.CYAN}--mixture-type{Style.RESET_ALL}       Mixture type (gmm or kde)
  {Fore.CYAN}--parallel{Style.RESET_ALL}           Enable or disable parallel startpoints for s-SuStaIn
  {Fore.CYAN}--run-analysis{Style.RESET_ALL}       Run sensitivity analysis to tune optimal stability threshold for feature selection
""")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Custom Argument Parser
def parse_args():
    parser = RichParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            f"{Fore.LIGHTMAGENTA_EX}🧬 Ensemble Feature Reduction + s-SuStaIn pipeline\n"
            f"{Fore.CYAN}➤ {Fore.RESET}Reveal subtype progression in omics datasets\n"
            f"{Fore.CYAN}➤ {Fore.RESET}Reduces dimensionality, stabilizes inference, plots results\n"
            f"{Fore.GREEN}🌟 Example Usage:\n"
            f"{Fore.YELLOW}        reduxis --data-file data.csv \\\n"
            f"{Fore.YELLOW}                 --metadata-file meta.csv \\\n"
            f"{Fore.YELLOW}                 --outcomes Normal Tumor \\\n"
            f"{Fore.YELLOW}                 --dataset-name MyCancerData\n"
            f"{Style.RESET_ALL}"
        ),
        add_help=False,
        allow_abbrev=False
    )

    # Define arguments
    parser.add_argument("--data-file",      help="Path to data file (e.g., RNA counts)")
    parser.add_argument("--metadata-file",  help="Path to metadata file (subject IDs + outcomes)")
    parser.add_argument("--outcomes",    nargs="+", help="Outcomes (e.g., Normal Tumor)")
    parser.add_argument("--dataset-name", help="Dataset name (e.g. 'Colorectal TCGA'")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "output", help="Output directory")
    parser.add_argument("--stages",      type=int, default=4, help="Number of disease stages")
    parser.add_argument("--subtypes",    type=int, default=1, help="Number of subtypes")
    parser.add_argument("--max-features", type=int, default=150, help="Maximum number of features for scaled EBM")
    parser.add_argument("--N_startpoints", type=int, default=25, help="Startpoints for optimization")
    parser.add_argument("--rep",         type=int, default=20, help="Repetitions for stability")
    parser.add_argument("--min-clust-size", type=int, default=10, help="Minimum cluster size")
    parser.add_argument("--p-absorb",    type=float, default=0.1, help="Probability to absorb feature")
    parser.add_argument("--N-iter-init", type=int, default=10000, help="Iterations (initial phase)")
    parser.add_argument("--N-iter",      type=int, default=100000, help="Iterations (main phase)")
    parser.add_argument("--N-em",        type=int, default=100, help="EM steps")
    parser.add_argument("--mixture-type", choices=["gmm","kde"], default="gmm", help="Mixture type")
    parser.add_argument("--parallel",    action="store_true", default=True, help="Enable or disable parallel startpoints for s-SuStaIn")
    parser.add_argument("--run-analysis",    action="store_true", default=True, help="Enable or disable feature selection sensitivity analysis")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    parser.add_argument("-v", "--version", action="version", version="ReduXis v1.0", help="Show the version number of ReduXis and exit")

    # Try parsing
    args, unknown = parser.parse_known_args()

    # If -h/--help or unknown garbage, print help
    if args.help or unknown:
        if unknown:
            logging.error(f"Unknown arguments: {' '.join(unknown)}\n")
        print_help()
        sys.exit(0)
    return args

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
def main():

    # Welcome screen
    print_welcome()

    # First parse whatever was given
    args = parse_args()

    # Interactively prompt for missing important arguments
    for req in ["data_file", "metadata_file", "output_dir", "outcomes", "dataset_name", "stages", "subtypes", "max_features", "min_clust_size", "parallel", "run_analysis"]:
        attr = req.replace("-", "_")
        current_val = getattr(args, attr)
        override = req in ("output_dir", "stages", "subtypes", "max_features", "min_clust_size", "parallel", "run_analysis")
        prompt_name = req.replace("_", " ")
        default_str = f" (default: {current_val})" if current_val not in (None, []) else ""

        # Data and metadata file - use tab completion, ensure file name exists
        if current_val in (None, []) or override:
            if req in ("data_file", "metadata_file"):
                while True:
                    value = interactive_prompt(prompt_name + default_str, use_tab_completion=True)
                    if not value and current_val:
                        break  # Keep existing
                    expanded = os.path.expanduser(value)
                    if not os.path.isfile(expanded):
                        logging.warning("The file you put in does not exist. Try again.")
                    else:
                        setattr(args, attr, expanded)
                        break
            elif req == "output_dir":
                prompt_name  = "output directory"
                default_str  = " (default: 'output' subdirectory within current directory)"
                value = interactive_prompt(prompt_name + default_str, use_tab_completion=True)
                if not value:
                    if not os.path.exists(current_val):
                        os.makedirs(current_val, exist_ok=True)
                        logging.info("Created default 'output' subdirectory in current directory.")
                    else:
                        logging.info("Storing output in existing default output directory.")
                    setattr(args, attr, current_val)
                else:
                    expanded = os.path.expanduser(value)
                    if not os.path.exists(expanded):
                        os.makedirs(expanded, exist_ok=True)
                        logging.info("Created new directory at specified custom path.")
                    else:
                        logging.info("Storing output in specified path, which already exists.")
                        setattr(args, attr, expanded)
            elif req in ("stages", "subtypes", "min_clust_size", "max_features"):
                while True:
                    if req == "min_clust_size":
                        value = interactive_prompt(prompt_name + default_str, use_tab_completion=False,
                            custom_prompt=f"{Fore.MAGENTA}Please enter the minimum number of biomarkers to designate in each stage outside of sensitivity analysis (default: {current_val}):")
                    elif req == "max_features":
                        value = interactive_prompt(prompt_name + default_str, use_tab_completion=False,
                            custom_prompt=f"{Fore.MAGENTA}Please enter the maximum number of features to hold in the model (default: {current_val}):")
                    else:
                        value = interactive_prompt(prompt_name + default_str, use_tab_completion=False,
                            custom_prompt=f"{Fore.MAGENTA}Please enter the number of {req} desired (default: {current_val}):")
                    if not value:
                        break  # keep current default
                    if value.isdigit():
                        setattr(args, attr, int(value))
                        break
                    else:
                        logging.warning(f"'{value}' isn't a valid number.")

            elif req == "outcomes":
                while True:
                    logging
                    value = interactive_prompt(prompt_name + default_str, use_tab_completion=False,
                                               custom_prompt = f"{Fore.MAGENTA}Please enter a list of disease outcomes, ordered from least to most severe, separated by spaces (i.e. CN MCI AD):")
                    final_val = value.split()
                    if len(final_val) < 2:
                        logging.warning("You need to enter at least two outcomes. Try again.")
                        continue
                    if len(final_val) > 10:
                        logging.info("Only the first 10 outcomes will be used.")
                        final_val = final_val[:10]
                    setattr(args, attr, final_val)
                    break

            elif req == "parallel":
                value = interactive_prompt(prompt_name + default_str, use_tab_completion=False, custom_prompt=(f"{Fore.MAGENTA}Enable parallel startpoints for s-SuStaIn? [yes/no] (default: yes):"))
                if not value:
                    user_choice = True # keep parallel startpoints turned on for better performance
                user_choice = value.strip().lower() not in ("n", "no", "false", "f", "0")
                setattr(args, attr, user_choice)
                if user_choice:
                    logging.info(f"Parallel startpoints for s-SuStaIn ENABLED.")
                else:
                    logging.warning(f"Parallel startpoints DISABLED. Performance may be significantly impacted; consider enabling for optimal runtime.")
            elif req == "run_analysis":
                value = interactive_prompt(prompt_name + default_str, use_tab_completion=False, custom_prompt=(f"{Fore.MAGENTA}Perform feature selection sensitivity analysis? [yes/no] (default: no):"))
                if not value:
                    selection = False # keep sensitivity analysis turned off for brevity
                selection = value.strip().lower() in ("y", "yes", "true", "t", "1")
                setattr(args, attr, selection)
                logging.info(f"Sensitivity analysis of stability score thresholds for feature selection {'ENABLED.' if selection else 'DISABLED.'}")
                if selection:
                    logging.info("Setting minimum cluster size temporarily to 5 in order to enable broader minimum feature range.")
            else:
                while True:
                    value = interactive_prompt(prompt_name + default_str, use_tab_completion=False, custom_prompt=f"{Fore.MAGENTA}Please enter {prompt_name}:")
                    if value:
                        setattr(args, attr, value)
                        break
                    else:
                        logging.warning(f"{prompt_name} is required. Please do not leave it blank.")

    print("")
    logging.info(f"Running ReduXis with dataset: {args.dataset_name}")
    run_sustain_pipeline(args)

if __name__ == "__main__":
    main()
