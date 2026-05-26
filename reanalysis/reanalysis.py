"""
LLMemotions Reanalysis Script
Runs sentiment analysis on valid dialogue subsets (excluding "---" outputs and NaN labels)
and compares results to the full dataset analysis.

Usage:
    python reanalysis.py --data_dir /path/to/csvs --output_dir ./reanalysis_output

CSV naming convention expected:
    event_log_1_baseline.csv, event_log_1_trauma.csv
    event_log_12_baseline.csv, event_log_12_trauma.csv
    event_log_52_baseline.csv, event_log_52_trauma.csv
    event_log_365_baseline.csv, event_log_365_trauma.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ── Sentiment libraries ──────────────────────────────────────────────────────
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

# ── Config ───────────────────────────────────────────────────────────────────
DPY_CONDITIONS = [1, 12, 52, 365]

DATA_DIR   = "."          # override with --data_dir
OUTPUT_DIR = "./reanalysis_output"


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def load_condition(data_dir, dpy, condition):
    """Load one CSV file."""
    fname = os.path.join(data_dir, f"event_log_{dpy}_{condition}.csv")
    if not os.path.exists(fname):
        print(f"  [WARN] Missing: {fname}")
        return None
    df = pd.read_csv(fname)
    df["dpy"] = dpy
    df["condition"] = condition
    return df


def extract_dialogue_text(content_series):
    """
    Pull the actual LLM output from the content column.
    Format: 'Agent_X talks with Agent_Y: "<dialogue>"'
    Returns the part after the first colon+space+quote.
    """
    def _extract(s):
        if not isinstance(s, str):
            return ""
        idx = s.find(': "')
        if idx == -1:
            return s
        return s[idx + 3:].rstrip('"')
    return content_series.apply(_extract)


def is_valid_dialogue(row):
    """
    A dialogue row is valid for analysis if:
      1. The extracted text is not '---' and not empty
      2. mental_state is not NaN
    """
    text = row["dialogue_text"]
    ms   = row["mental_state"]
    if not isinstance(text, str) or text.strip() in ("", "---"):
        return False
    if pd.isna(ms) or str(ms).strip().upper() == "N/A":
        return False
    return True


def prepare_dialogues(df):
    """Extract and filter dialogue rows, add validity flag."""
    d = df[df["event"] == "dialogue"].copy()
    d["dialogue_text"] = extract_dialogue_text(d["content"])
    d["valid"] = d.apply(is_valid_dialogue, axis=1)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT SCORING
# ═════════════════════════════════════════════════════════════════════════════

def score_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def score_vader(text):
    vs = vader.polarity_scores(text)
    return vs["compound"]


def add_sentiment_scores(df):
    """Add TextBlob and VADER scores to a dialogue DataFrame."""
    tb = df["dialogue_text"].apply(score_textblob)
    df["tb_polarity"]     = [x[0] for x in tb]
    df["tb_subjectivity"] = [x[1] for x in tb]
    df["vader_compound"]  = df["dialogue_text"].apply(score_vader)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 3. STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def cliff_delta(x, y):
    """Cliff's delta effect size."""
    x, y = np.array(x), np.array(y)
    n = len(x) * len(y)
    if n == 0:
        return np.nan
    greater = sum(xi > yj for xi in x for yj in y)
    less    = sum(xi < yj for xi in x for yj in y)
    return (greater - less) / n


def compare_conditions(base_scores, trauma_scores, metric_name):
    """Mann-Whitney U + Cliff's delta for one metric."""
    if len(base_scores) < 5 or len(trauma_scores) < 5:
        return {
            "metric": metric_name,
            "n_base": len(base_scores),
            "n_trauma": len(trauma_scores),
            "median_base": np.nan,
            "median_trauma": np.nan,
            "delta_median": np.nan,
            "mwu_p": np.nan,
            "cliffs_d": np.nan,
            "note": "insufficient data"
        }
    stat, p = mannwhitneyu(base_scores, trauma_scores, alternative="two-sided")
    cd = cliff_delta(base_scores, trauma_scores)
    return {
        "metric": metric_name,
        "n_base": len(base_scores),
        "n_trauma": len(trauma_scores),
        "median_base": round(np.median(base_scores), 4),
        "median_trauma": round(np.median(trauma_scores), 4),
        "delta_median": round(np.median(trauma_scores) - np.median(base_scores), 4),
        "mean_base": round(np.mean(base_scores), 4),
        "mean_trauma": round(np.mean(trauma_scores), 4),
        "delta_mean": round(np.mean(trauma_scores) - np.mean(base_scores), 4),
        "mwu_p": round(p, 4),
        "cliffs_d": round(cd, 4),
        "note": ""
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. MAIN ANALYSIS LOOPS
# ═════════════════════════════════════════════════════════════════════════════

def run_aggregate_comparison(all_dialogues_by_dpy):
    """
    For each DPY: compare baseline vs trauma on TextBlob polarity and VADER compound.
    Runs on BOTH full set and valid-only subset.
    Returns two DataFrames (full, valid).
    """
    rows_full  = []
    rows_valid = []

    for dpy in DPY_CONDITIONS:
        pair = all_dialogues_by_dpy.get(dpy)
        if pair is None:
            continue
        base_full,  trauma_full  = pair["base_full"],  pair["trauma_full"]
        base_valid, trauma_valid = pair["base_valid"], pair["trauma_valid"]

        for metric in ["tb_polarity", "vader_compound"]:
            # full set
            rows_full.append({
                "dpy": dpy,
                **compare_conditions(
                    base_full[metric].values,
                    trauma_full[metric].values,
                    metric
                )
            })
            # valid subset
            rows_valid.append({
                "dpy": dpy,
                **compare_conditions(
                    base_valid[metric].values,
                    trauma_valid[metric].values,
                    metric
                )
            })

    return pd.DataFrame(rows_full), pd.DataFrame(rows_valid)


def run_state_personality_comparison(all_dialogues_by_dpy):
    """
    For each DPY × mental_state × personality:
    compare baseline vs trauma (valid subset only, since NaN-labeled rows
    have no state/personality info).
    Returns a long-format DataFrame.
    """
    rows = []

    for dpy in DPY_CONDITIONS:
        pair = all_dialogues_by_dpy.get(dpy)
        if pair is None:
            continue
        base_valid  = pair["base_valid"]
        trauma_valid = pair["trauma_valid"]

        states = set(base_valid["mental_state"].dropna().unique()) | \
                 set(trauma_valid["mental_state"].dropna().unique())
        personalities = set(base_valid["personality"].dropna().unique()) | \
                        set(trauma_valid["personality"].dropna().unique())

        for state in sorted(states):
            for pers in sorted(personalities):
                b = base_valid[
                    (base_valid["mental_state"] == state) &
                    (base_valid["personality"]  == pers)
                ]
                t = trauma_valid[
                    (trauma_valid["mental_state"] == state) &
                    (trauma_valid["personality"]  == pers)
                ]
                for metric in ["tb_polarity", "vader_compound"]:
                    res = compare_conditions(b[metric].values, t[metric].values, metric)
                    rows.append({
                        "dpy": dpy,
                        "mental_state": state,
                        "personality": pers,
                        **res
                    })

    return pd.DataFrame(rows)


def run_data_audit(all_dialogues_by_dpy):
    """Summary table of data counts and exclusion rates."""
    rows = []
    for dpy in DPY_CONDITIONS:
        pair = all_dialogues_by_dpy.get(dpy)
        if pair is None:
            continue
        for cond_name, df_full, df_valid in [
            ("baseline", pair["base_full"],  pair["base_valid"]),
            ("trauma",   pair["trauma_full"], pair["trauma_valid"])
        ]:
            total   = len(df_full)
            n_dash  = df_full["dialogue_text"].str.strip().eq("---").sum()
            n_nan   = df_full["mental_state"].isna().sum()
            n_valid = len(df_valid)
            rows.append({
                "dpy": dpy,
                "condition": cond_name,
                "total_dialogue": total,
                "n_dash_output": n_dash,
                "pct_dash": round(100 * n_dash / max(total, 1), 1),
                "n_nan_label": n_nan,
                "pct_nan_label": round(100 * n_nan / max(total, 1), 1),
                "n_valid": n_valid,
                "pct_valid": round(100 * n_valid / max(total, 1), 1),
            })
    return pd.DataFrame(rows)


def run_temporal_sentiment(all_dialogues_by_dpy, bin_size=50):
    """
    Compute rolling mean sentiment over time (by day bins).
    Valid subset only.
    """
    rows = []
    for dpy in DPY_CONDITIONS:
        pair = all_dialogues_by_dpy.get(dpy)
        if pair is None:
            continue
        for cond_name, df in [("baseline", pair["base_valid"]),
                               ("trauma",   pair["trauma_valid"])]:
            if df.empty:
                continue
            df2 = df.copy()
            df2["day_bin"] = (df2["day"] // bin_size) * bin_size
            grouped = df2.groupby("day_bin")[["tb_polarity", "vader_compound"]].mean().reset_index()
            grouped["dpy"] = dpy
            grouped["condition"] = cond_name
            rows.append(grouped)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# 5. DIRECTION CONSISTENCY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def direction_summary(df_full, df_valid):
    """
    For each DPY × metric: show direction of delta_mean in full vs valid.
    Flags reversals (opposite signs).
    """
    rows = []
    for dpy in DPY_CONDITIONS:
        for metric in ["tb_polarity", "vader_compound"]:
            f = df_full[(df_full["dpy"] == dpy) & (df_full["metric"] == metric)]
            v = df_valid[(df_valid["dpy"] == dpy) & (df_valid["metric"] == metric)]
            if f.empty or v.empty:
                continue
            dm_full  = f["delta_mean"].values[0]
            dm_valid = v["delta_mean"].values[0]
            reversal = (np.sign(dm_full) != np.sign(dm_valid)) and \
                       not (np.isnan(dm_full) or np.isnan(dm_valid))
            rows.append({
                "dpy": dpy,
                "metric": metric,
                "delta_mean_full":  round(dm_full,  4),
                "delta_mean_valid": round(dm_valid, 4),
                "reversal": "⚠ REVERSAL" if reversal else "consistent"
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print("LLMemotions Reanalysis")
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load all CSVs ────────────────────────────────────────────────────────
    all_dialogues_by_dpy = {}

    for dpy in DPY_CONDITIONS:
        base_df   = load_condition(data_dir, dpy, "baseline")
        trauma_df = load_condition(data_dir, dpy, "trauma")
        if base_df is None or trauma_df is None:
            continue

        base_d   = prepare_dialogues(base_df)
        trauma_d = prepare_dialogues(trauma_df)

        base_d   = add_sentiment_scores(base_d)
        trauma_d = add_sentiment_scores(trauma_d)

        all_dialogues_by_dpy[dpy] = {
            "base_full":   base_d,
            "trauma_full": trauma_d,
            "base_valid":  base_d[base_d["valid"]].copy(),
            "trauma_valid": trauma_d[trauma_d["valid"]].copy(),
        }
        print(f"DPY={dpy:>3}  base: {len(base_d)} dialogues "
              f"({base_d['valid'].sum()} valid)  |  "
              f"trauma: {len(trauma_d)} dialogues "
              f"({trauma_d['valid'].sum()} valid)")

    print()

    # ── Data audit ───────────────────────────────────────────────────────────
    audit = run_data_audit(all_dialogues_by_dpy)
    print("── DATA AUDIT ──────────────────────────────────────────────")
    print(audit.to_string(index=False))
    audit.to_csv(os.path.join(output_dir, "data_audit.csv"), index=False)

    # ── Aggregate sentiment comparison ───────────────────────────────────────
    df_full, df_valid = run_aggregate_comparison(all_dialogues_by_dpy)
    print("\n── AGGREGATE COMPARISON: FULL DATASET ─────────────────────")
    print(df_full[["dpy","metric","n_base","n_trauma",
                   "median_base","median_trauma","delta_mean",
                   "mwu_p","cliffs_d"]].to_string(index=False))
    print("\n── AGGREGATE COMPARISON: VALID SUBSET ─────────────────────")
    print(df_valid[["dpy","metric","n_base","n_trauma",
                    "median_base","median_trauma","delta_mean",
                    "mwu_p","cliffs_d"]].to_string(index=False))
    df_full.to_csv(os.path.join(output_dir, "aggregate_full.csv"), index=False)
    df_valid.to_csv(os.path.join(output_dir, "aggregate_valid.csv"), index=False)

    # ── Direction consistency ────────────────────────────────────────────────
    direction = direction_summary(df_full, df_valid)
    print("\n── DIRECTION CONSISTENCY (full vs valid subset) ────────────")
    print(direction.to_string(index=False))
    direction.to_csv(os.path.join(output_dir, "direction_consistency.csv"), index=False)

    # ── State × Personality comparison ──────────────────────────────────────
    sp_comp = run_state_personality_comparison(all_dialogues_by_dpy)
    sp_comp.to_csv(os.path.join(output_dir, "state_personality_comparison.csv"), index=False)
    print(f"\n── STATE×PERSONALITY: {len(sp_comp)} rows saved to state_personality_comparison.csv")

    # ── Temporal sentiment ───────────────────────────────────────────────────
    temporal = run_temporal_sentiment(all_dialogues_by_dpy)
    if not temporal.empty:
        temporal.to_csv(os.path.join(output_dir, "temporal_sentiment.csv"), index=False)
        print(f"── TEMPORAL: {len(temporal)} rows saved to temporal_sentiment.csv")

    print(f"\nAll outputs written to: {output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=".",
                        help="Directory containing event_log_*.csv files")
    parser.add_argument("--output_dir", default="./reanalysis_output",
                        help="Where to write result CSVs")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
