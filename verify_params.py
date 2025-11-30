import pandas as pd
import numpy as np
import os

# Path to your reference dataset
DATASET_PATH = os.path.join('data', '243_specificenergy.csv')


def verify_parameters_with_llm(P, mf, v, df, do, target_depth):
    """
    Validates the proposed parameters by comparing them against the
    reference dataset '243_specificenergy.csv'.
    """
    print(f"\n--- Dataset Validation Phase (243_specificenergy.csv) ---")

    if not os.path.exists(DATASET_PATH):
        return {"error": f"Dataset not found at {DATASET_PATH}"}

    try:
        # Load the reference dataset
        ref_df = pd.read_csv(DATASET_PATH)

        # Standardizing column names for this logic
        # We look for columns that contain these keywords
        col_map = {
            'P': None,
            'mf': None,
            'v': None,
            'h': None
        }

        # Heuristic to find the right columns
        for col in ref_df.columns:
            lower_col = col.lower()
            if 'p (' in lower_col or 'pressure' in lower_col:
                col_map['P'] = col
            elif 'mf' in lower_col or 'flow' in lower_col:
                # Distinguish between mass flow (kg/min) vs water flow if needed
                if 'kg/min' in lower_col:
                    col_map['mf'] = col
            elif ('v (' in lower_col or 'speed' in lower_col) and 'mm/min' in lower_col:
                col_map['v'] = col
            elif 'h (' in lower_col or 'depth' in lower_col:
                col_map['h'] = col

        # If mapping failed for critical columns, return error
        missing_cols = [k for k, v in col_map.items() if v is None and k != 'h']
        if missing_cols:
            return {"error": f"Could not map columns for {missing_cols}. Found in CSV: {ref_df.columns.tolist()}"}

        # Find closest input parameters using normalized Euclidean distance
        min_distance = float('inf')
        best_match_idx = -1

        # We only compare the control inputs (P, mf, v)
        for idx, row in ref_df.iterrows():
            # Calculate distance (handling zeros to avoid div by zero)
            d_p = ((row[col_map['P']] - P) / (P + 1e-6)) ** 2
            d_mf = ((row[col_map['mf']] - mf) / (mf + 1e-6)) ** 2
            d_v = ((row[col_map['v']] - v) / (v + 1e-6)) ** 2

            dist = d_p + d_mf + d_v

            if dist < min_distance:
                min_distance = dist
                best_match_idx = idx

        closest_row = ref_df.iloc[best_match_idx]

        # 2. Extract Data from Closest Match
        ref_P = closest_row[col_map['P']]
        ref_mf = closest_row[col_map['mf']]
        ref_v = closest_row[col_map['v']]
        ref_h = closest_row[col_map['h']] if col_map['h'] else "N/A"

        # 3. Construct the Output
        # We mimic the LLM JSON response structure for compatibility

        verdict = "SAFE"
        # Check for deviation if we have depth data
        if isinstance(ref_h, (int, float)):
            error = abs(ref_h - target_depth)
            # If the closest experimental setup yielded a depth very different from our target
            # it implies our GA might be extrapolating or finding a solution that contradicts data.
            if error > 5.0:
                verdict = "WARNING"
                confidence = "LOW (Deviation from Exp. Data)"
            else:
                confidence = "HIGH (Matches Exp. Data)"
        else:
            confidence = "MEDIUM (No depth data to compare)"

        reasoning = (
            f"**Validation against Experimental Dataset (Row #{best_match_idx}):**\n\n"
            f"The optimized parameters are compared to the nearest real-world experiment:\n"
            f"* **Pressure:** {ref_P:.2f} MPa (vs {P:.2f} MPa)\n"
            f"* **Flow Rate:** {ref_mf:.4f} kg/min (vs {mf:.4f} kg/min)\n"
            f"* **Traverse Speed:** {ref_v:.2f} mm/min (vs {v:.2f} mm/min)\n\n"
            f"**Outcome:** The experimental setup produced a depth of **{ref_h} mm**.\n"
            f"*(Target was {target_depth} mm)*"
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning
        }

    except Exception as e:
        return {"error": str(e)}