#!/usr/bin/env python3
"""
Baseline Logistics Incident Triage Engine (Python)
- Fast-path keyword rule engine with weighted scoring and margin checks.
- Urgency classification logic.
- Automated CSV output and Markdown summary report generation.
"""

import os
import sys
import argparse
import time
import re
from datetime import datetime
import pandas as pd

# Categories definition
CATEGORIES = [
    "Theft/Security",
    "Cold Chain/Temperature",
    "Delay/Transit",
    "Mispick/Order Error",
    "Damage/Handling"
]
CAT_PREFIXES = ["THF", "CLD", "DLY", "MSP", "DMG"]

# Keyword definitions: (category_index, weight, pattern)
# Patterns ending with '*' are treated as prefix matches.
KW_DEFS = [
    # ---- Theft / Security ----
    (0, 3, "stol*"), (0, 3, "steal*"), (0, 3, "theft"), (0, 3, "hijack*"), (0, 3, "burglar*"),
    (0, 3, "break in"), (0, 3, "broke into"), (0, 3, "tamper*"), (0, 3, "padlock"), (0, 3, "cctv"),
    (0, 2, "lock cut"), (0, 2, "unauthorized"), (0, 2, "missing"), (0, 2, "drained"), (0, 1, "guard"),
    
    # ---- Cold Chain / Temperature ----
    (1, 3, "reefer"), (1, 3, "cold chain"), (1, 3, "refrigerat*"), (1, 3, "fridge"), (1, 3, "chiller"),
    (1, 3, "freezer"), (1, 3, "thaw*"), (1, 3, "melted"), (1, 3, "temperature*"), (1, 3, "excursion*"),
    (1, 2, "warm"), (1, 2, "ice"), (1, 1, "sensor"), (1, 1, "alarm"), (1, 1, "power"),

    # ---- Delay / Transit ----
    (2, 3, "traffic"), (2, 3, "delayed"), (2, 3, "delay*"), (2, 3, "detour"), (2, 3, "gridlock"),
    (2, 3, "stranded"), (2, 3, "eta"), (2, 3, "breakdown"), (2, 3, "accident"), (2, 2, "stuck"),
    (2, 1, "late"), (2, 1, "route"), (2, 1, "highway"), (2, 1, "weather"), (2, 1, "driver*"),

    # ---- Mispick / Order Error ----
    (3, 3, "mispick*"), (3, 3, "picker*"), (3, 3, "pick list"), (3, 3, "wrong item"), (3, 3, "sku"),
    (3, 3, "packing slip"), (3, 3, "swapped"), (3, 3, "mismatch*"), (3, 3, "incorrect"), (3, 2, "wrong"),
    (3, 1, "extra"), (3, 1, "quantity"), (3, 1, "barcode*"), (3, 1, "missing"), (3, 1, "short"),

    # ---- Damaged packaging ----
    (4, 3, "crush*"), (4, 3, "dent*"), (4, 3, "torn"), (4, 3, "ripped"), (4, 3, "punctur*"),
    (4, 3, "crack*"), (4, 3, "shatter*"), (4, 3, "unsellable"), (4, 3, "damage*"), (4, 3, "leak*"),
    (4, 2, "broken"), (4, 2, "wet"), (4, 2, "spill*"), (4, 1, "pallet*"), (4, 1, "forklift")
]

# Urgency cues
URG_UP = ["vaccine", "medical", "hospital", "dialysis", "organ", "insulin", "emergency", "penalty", 
          "production line", "shutdown", "grounded", "critical", "perishable", "urgent", "asap", "immediately"]
URG_DOWN = ["flexible", "not urgent", "no impact", "acceptable", "routine", "minor", "showers", "rescheduled"]

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and replace non-alphanumeric with spaces
    cleaned = re.sub(r'[^a-zA-Z0-9*]', ' ', text.lower())
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return f" {cleaned} "

def compile_matchers():
    matchers = []
    for cat, weight, pat in KW_DEFS:
        if pat.endswith('*'):
            needle = " " + pat[:-1]
            match_type = "prefix"
        else:
            needle = f" {pat} "
            match_type = "exact"
        matchers.append({"cat": cat, "w": weight, "needle": needle, "type": match_type, "pat": pat})
    return matchers

def rule_classify(norm_text, matchers):
    scores = [0] * len(CATEGORIES)
    hits = []
    
    for m in matchers:
        if m["type"] == "prefix":
            # Check if any word starts with the pattern
            words = norm_text.strip().split()
            prefix_val = m["needle"].strip()
            if any(w.startswith(prefix_val) for w in words):
                scores[m["cat"]] += m["w"]
                hits.append(m["pat"])
        else:
            if m["needle"] in norm_text:
                scores[m["cat"]] += m["w"]
                hits.append(m["pat"])
                
    sorted_scores = sorted([(s, i) for i, s in enumerate(scores)], reverse=True)
    best_score, best_cat = sorted_scores[0]
    second_score, _ = sorted_scores[1]
    
    margin = best_score - second_score
    return best_cat, best_score, margin, scores, list(set(hits))

def predict_urgency(cat_idx, norm_text):
    # Theft & Cold Chain default to urgent; Mispick & Damage default to standard
    if cat_idx in [0, 1]:
        return True
    if cat_idx in [3, 4]:
        return False
    
    # For delays, check keyword net score
    up_hits = sum(1 for u in URG_UP if f" {u} " in norm_text)
    down_hits = sum(1 for d in URG_DOWN if f" {d} " in norm_text)
    return (up_hits - down_hits) >= 1

def main():
    parser = argparse.ArgumentParser(description="Baseline Python Logistics Incident Triage Engine")
    parser.add_argument("--csv", default="logistics_incident_reports.csv", help="Input CSV file path")
    parser.add_argument("--out", default="python_triage", help="Output file prefix for results and reports")
    parser.add_argument("--min-top", type=int, default=5, help="Minimum score to trust rule classification")
    parser.add_argument("--min-margin", type=int, default=2, help="Minimum score margin over runner-up")
    args = parser.parse_args()

    print("==================================================")
    print("   PYTHON BASELINE INCIDENT TRIAGE ENGINE         ")
    print("==================================================")

    if not os.path.exists(args.csv):
        print(f"Error: Could not find input file '{args.csv}'.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} records from {args.csv}\n")

    matchers = compile_matchers()
    
    results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        inc_id = row.get("incident_id", f"INC-{idx}")
        text = row.get("incident_text", row.get("text", ""))
        true_urgency_str = row.get("urgency", "standard")
        
        # Determine true category from ID prefix if available
        true_cat = -1
        for c_idx, pref in enumerate(CAT_PREFIXES):
            if str(inc_id).startswith(pref):
                true_cat = c_idx
                break

        norm = normalize_text(text)
        best_cat, top_score, margin, scores, hits = rule_classify(norm, matchers)
        
        confident = (top_score >= args.min_top) and (margin >= args.min_margin)
        # Check confidence rules
        confident = (top_score >= args.min_top) and (margin >= args.min_margin)
        
        if confident:
            pred_cat = best_cat
            source = "rules"
            review_status = "Auto-Resolved"
        elif best_cat >= 0:
            pred_cat = best_cat
            source = "rules-lowconf"
            review_status = "Human Review (Low Confidence)"
        else:
            pred_cat = -1
            source = "unclassified"
            review_status = "Human Review (Unclassified)"

        # Define urgency prediction before evaluation checks
        pred_urgent = predict_urgency(pred_cat if pred_cat >= 0 else 2, norm)
        
        # Ground truth evaluation checks
        true_urg_bool = 0 if "urgent" in str(true_urgency_str).lower() else 1
        urg_correct = (pred_urgent == (true_urg_bool == 0))
        cat_correct = (pred_cat == true_cat) if true_cat >= 0 else None
        results.append({
            "incident_id": inc_id,
            "true_category": CATEGORIES[true_cat] if true_cat >= 0 else "unknown",
            "pred_category": CATEGORIES[pred_cat] if pred_cat >= 0 else "Unclassified",
            "category_correct": "yes" if cat_correct else ("no" if cat_correct is not None else "n/a"),
            "source": source,
            "rule_top_score": top_score,
            "rule_margin": margin,
            "true_urgency": true_urgency_str,
            "pred_urgency": "Urgent" if pred_urgent else "Standard",
            "urgency_correct": "yes" if urg_correct else "no",
            "matched_keywords": "; ".join(hits)
        })

    elapsed_ms = (time.time() - start_time) * 1000.0

    # Output Results CSV
    res_df = pd.DataFrame(results)
    csv_filename = f"{args.out}_results.csv"
    res_df.to_csv(csv_filename, index=False)

    # Compute high-level statistics
    total_records = len(results)
    known_cats = res_df[res_df["true_category"] != "unknown"]
    cat_accuracy = (known_cats["category_correct"] == "yes").mean() * 100 if len(known_cats) > 0 else 0
    urg_accuracy = (res_df["urgency_correct"] == "yes").mean() * 100

    # Output Markdown Report
    md_filename = f"{args.out}_report.md"
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("# Python Baseline Logistics Triage - Run Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Run Summary\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Dataset File | `{args.csv}` |\n")
        f.write(f"| Total Incidents Processed | {total_records} |\n")
        f.write(f"| Execution Time | {elapsed_ms:.2f} ms ({elapsed_ms/total_records:.3f} ms/record) |\n")
        f.write(f"| Category Accuracy | {cat_accuracy:.1f}% |\n")
        f.write(f"| Urgency Accuracy | {urg_accuracy:.1f}% |\n\n")
        f.write("## 2. Classification Breakdown by Source\n\n")
        source_counts = res_df["source"].value_counts()
        f.write("| Source Strategy | Count | Percentage |\n|---|---|---|\n")
        for src, count in source_counts.items():
            f.write(f"| `{src}` | {count} | {(count/total_records)*100:.1f}% |\n")

    print("Batch Run Completed Successfully!")
    print(f" -> Results saved to: {csv_filename}")
    print(f" -> Report generated: {md_filename}")
    print(f" -> Overall Category Accuracy: {cat_accuracy:.1f}%")
    print(f" -> Overall Urgency Accuracy:  {urg_accuracy:.1f}%")

if __name__ == "__main__":
    main()

