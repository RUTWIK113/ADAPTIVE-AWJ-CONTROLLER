import streamlit as st
import time
import numpy as np
import os
import json
import sys
import subprocess
import pandas as pd
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('data', 'awj_model.keras')
SCALER_PATH = os.path.join('data', 'scaler.pkl')
DATA_PATH = os.path.join('data', 'awj_training_data.csv')
REF_DATA_PATH = os.path.join('data', '243_specificenergy.csv')

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Adaptive AWJ Controller",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: TEAL & EMERALD THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Ubuntu', sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
    }

    header {visibility: hidden;}
    .sticky-nav {
        position: fixed; top: 0; left: 0; width: 100%;
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border-bottom: 2px solid #10b981;
        color: #f1f5f9; padding: 8px 20px; z-index: 99999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        display: flex; align-items: center; justify-content: space-between;
        height: 50px;
    }
    .main-content { margin-top: 60px; }

    .stExpander, div[data-testid="stContainer"] {
        background-color: #1e293b; border-radius: 6px;
        border: 1px solid #334155; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .stNumberInput input, .stTextInput input {
        color: #f1f5f9; background-color: #0f172a; border: 1px solid #334155;
    }

    .big-metric {
        background-color: #1e293b; border-left: 4px solid #14b8a6;
        padding: 15px; border-radius: 6px; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 10px;
    }
    .metric-value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; }
    .metric-label { color: #14b8a6; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 2px; }
    .metric-unit { color: #94a3b8; font-size: 0.85rem; font-weight: 400; }

    .handout-card {
        background-color: #1e293b; border: 1px solid #10b981;
        border-radius: 6px; padding: 20px; margin-top: 15px;
    }
    .handout-title {
        color: #10b981; font-weight: 700; border-bottom: 2px solid #10b981;
        padding-bottom: 8px; margin-bottom: 15px; font-size: 1.1rem;
    }

    /* Comparison Table Styling */
    .comp-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-family: monospace;
        font-size: 0.9em;
    }
    .comp-table th { text-align: left; color: #94a3b8; padding: 5px; border-bottom: 1px solid #334155; }
    .comp-table td { padding: 8px 5px; color: #f1f5f9; border-bottom: 1px solid #334155; }
    .highlight { color: #10b981; font-weight: bold; }

    .snail-wrapper { width: 100%; margin: 20px 0; text-align: center; }
    .snail-track { position: relative; width: 100%; height: 6px; background-color: #334155; border-radius: 10px; margin-top: 15px; overflow: hidden; }
    .snail-trail { position: absolute; top: 0; left: 0; height: 100%; background: linear-gradient(90deg, #14b8a6, #10b981); width: 0%; border-radius: 10px; }
    .snail-mover { position: relative; top: -28px; left: 0%; width: 35px; height: 35px; font-size: 24px; line-height: 35px; transition: left 0.5s ease-out; z-index: 10; }

    .running .snail-mover { animation: crawl 12s forwards linear; }
    .running .snail-trail { animation: trailFill 12s forwards linear; }
    .done .snail-mover { left: 95% !important; animation: none; }
    .done .snail-trail { width: 100% !important; animation: none; }

    @keyframes crawl { 0% { left: 0%; } 100% { left: 95%; } }
    @keyframes trailFill { 0% { width: 0%; } 100% { width: 95%; } }

    .status-text { color: #14b8a6; margin-top: 5px; font-weight: 500; font-size: 0.9em; }
    .gen-badge { background-color: #10b981; color: #0f172a; padding: 4px 10px; border-radius: 4px; font-size: 0.8em; font-weight: 700; margin-bottom: 5px; display: inline-block; }

    .stButton>button {
        background: linear-gradient(45deg, #10b981, #14b8a6);
        color: #0f172a; border: none; border-radius: 4px; font-weight: 600;
        transition: transform 0.2s; height: 40px;
    }
    .stButton>button:hover { transform: translateY(-1px); color: #fff; }

    .stStatus { background-color: #1e293b; color: #f1f5f9; border-color: #334155; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- COMPACT STICKY NAVBAR ---
st.markdown("""
<div class="sticky-nav">
    <div style="display:flex; align-items:center; gap:10px;">
        <h1 style="margin:0; font-size:1.1rem; color: #f1f5f9; font-weight:700;">🌊 Adaptive AWJ Controller</h1>
        <span style="font-size: 0.8rem; color: #94a3b8; border-left:1px solid #334155; padding-left:10px;">Hybrid AI Optimization</span>
    </div>
    <div style="font-size: 0.75rem; font-weight:600; color: #10b981; background:rgba(16, 185, 129, 0.1); padding:4px 8px; border-radius:4px; border: 1px solid #10b981;">
        STATUS: ONLINE
    </div>
</div>
<div class="main-content"></div>
""", unsafe_allow_html=True)

# --- SYSTEM HEALTH CHECK ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.container()
    st.warning("⚠️ System Not Initialized")
    st.info(
        "Required model artifacts (awj_model.keras, scaler.pkl) are missing. Please run the training pipeline manually.")
    st.stop()

# --- MAIN APP LOGIC ---
try:
    from control.ga_optimizer import run_genetic_algorithm
    from verify_params import verify_parameters_with_llm
    from vision.monitoring import measure_nozzle_diameter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()


# --- HELPER: FIND NEAREST NEIGHBOR IN CSV ---
def find_closest_experiment(P, mf, v):
    """
    Scans the experimental CSV to find the row with the most similar inputs.
    Returns the row data and the depth it achieved.
    """
    if not os.path.exists(REF_DATA_PATH):
        return None, "Dataset not found."

    try:
        df = pd.read_csv(REF_DATA_PATH)

        # Normalize weights for distance calculation (approximate ranges)
        # Pressure range ~200, Flow ~0.5, Speed ~2000
        # We weigh them to treat them somewhat equally in the distance metric

        # Simple Euclidean distance on normalized inputs
        # (Assuming columns exist, we use heuristics to find them)
        p_col = next((c for c in df.columns if 'P' in c), None)
        mf_col = next((c for c in df.columns if 'mf' in c and 'kg/min' in c), None)
        v_col = next((c for c in df.columns if 'v' in c and 'mm/min' in c), None)
        h_col = next((c for c in df.columns if 'h' in c or 'depth' in c.lower()), None)

        if not all([p_col, mf_col, v_col, h_col]):
            return None, "Column mapping failed."

        min_dist = float('inf')
        closest_row = None

        for idx, row in df.iterrows():
            # Normalized Distance Squared
            dist = ((row[p_col] - P) / 400) ** 2 + ((row[mf_col] - mf) / 1.0) ** 2 + ((row[v_col] - v) / 5000) ** 2

            if dist < min_dist:
                min_dist = dist
                closest_row = row

        return closest_row, {
            'P': closest_row[p_col],
            'mf': closest_row[mf_col],
            'v': closest_row[v_col],
            'h': closest_row[h_col]
        }

    except Exception as e:
        return None, str(e)


# --- ANIMATION HELPERS ---
def render_snail_progress():
    return st.markdown("""
    <div class="snail-wrapper running">
        <span class="gen-badge">Running Genetic Algorithm</span>
        <div style="position: relative;">
            <div class="snail-track"><div class="snail-trail"></div></div>
            <div class="snail-mover">🐌</div>
        </div>
        <p class="status-text">Evolving Generations...</p>
    </div>
    """, unsafe_allow_html=True)


def render_snail_complete():
    return st.markdown("""
    <div class="snail-wrapper done">
        <span class="gen-badge" style="background-color: #10b981; color: #0f172a;">Optimization Complete</span>
        <div style="position: relative;">
            <div class="snail-track"><div class="snail-trail"></div></div>
            <div class="snail-mover">🏁</div>
        </div>
        <p class="status-text" style="color:#10b981;">Solution Found!</p>
    </div>
    """, unsafe_allow_html=True)


# --- MACHINE SETUP SECTION ---
with st.expander("⚙️ Machine Configuration", expanded=True):
    col_setup1, col_setup2 = st.columns(2)

    with col_setup1:
        do_input = st.number_input(
            "Orifice Diameter (do) [mm]",
            min_value=0.1, max_value=0.5, value=0.24, step=0.01
        )

    with col_setup2:
        st.markdown("**📏 Focusing Tube (df)**")
        df_mode = st.radio("Source:", ["Manual Input", "Vision System"], index=0, horizontal=True)

        df_input = 0.72

        if df_mode == "Manual Input":
            df_input = st.number_input(
                "Enter Diameter [mm]",
                min_value=0.5, max_value=1.5, value=0.72, step=0.01
            )
        else:
            uploaded_file = st.file_uploader("Upload Nozzle Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Nozzle Tip", width=150)
                with open("temp_nozzle.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Analyzing..."):
                    PIXELS_TO_MM = 0.01
                    inner, outer = measure_nozzle_diameter("temp_nozzle.jpg", PIXELS_TO_MM)
                    if inner is not None:
                        st.success(f"Detected: {inner:.4f} mm")
                        df_input = inner
                    else:
                        st.error("Failed. Using Default.")
                        df_input = 0.72
            else:
                st.info("Awaiting Image...")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🎯 Control Target")
    target_depth = st.number_input(
        "Desired Depth of Cut (mm)",
        min_value=0.1, max_value=80.0, value=8.10, step=0.1
    )
    # Spacing handled by CSS margins now
    run_btn = st.button("🚀 Optimize Parameters", type="primary", use_container_width=True)

# --- OPTIMIZATION LOGIC ---
if run_btn:
    param_ranges = {
        "P (MPa)": [100.0, 400.0],
        "mf (kg/min)": [0.1, 1.0],
        "v (mm/min)": [100.0, 5000.0],
        "df (mm)": [0.76, 1.6],
        "do (mm)": [0.1, 0.3]
    }

    # 1. SHOW RUNNING ANIMATION
    with col2:
        placeholder = st.empty()
        with placeholder.container():
            render_snail_progress()
            st.caption("Generation Nevals: Simulating 50 Generations...")

    # 2. RUN GA
    static_inputs = [df_input, do_input]
    start_time = time.time()
    optimal_params = run_genetic_algorithm(
        param_ranges=param_ranges,
        static_inputs=static_inputs,
        desired_depth=target_depth
    )

    # 3. CLEAR RUNNING & SHOW COMPLETE
    placeholder.empty()

    # 4. RESULTS DISPLAY
    with col2:
        render_snail_complete()
        st.caption(f"Computation Time: {time.time() - start_time:.2f}s | 50 Generations × 50 Pop")

        opt_P = optimal_params['pressure']
        opt_mf = optimal_params['flow_rate']
        opt_v = optimal_params['traverse_rate']

        # --- BIG METRIC CARDS ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="big-metric">
                <div class="metric-label">Water Pressure</div>
                <div class="metric-value">{opt_P:.1f}</div>
                <div class="metric-unit">MPa</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="big-metric">
                <div class="metric-label">Abrasive Flow</div>
                <div class="metric-value">{opt_mf:.3f}</div>
                <div class="metric-unit">kg/min</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="big-metric">
                <div class="metric-label">Traverse Speed</div>
                <div class="metric-value">{opt_v:.1f}</div>
                <div class="metric-unit">mm/min</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # --- EXPERIMENTAL VALIDATION HANDOUT ---
    st.markdown('<h3 style="color:#f1f5f9;">🧪 Experimental Validation Handout</h3>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="handout-card">', unsafe_allow_html=True)
        st.markdown('<div class="handout-title">📋 Dataset Comparison (Validation)</div>', unsafe_allow_html=True)

        with st.spinner("Finding Nearest Experimental Match..."):
            # Find closest row in CSV
            row_data, closest_vals = find_closest_experiment(opt_P, opt_mf, opt_v)

            if closest_vals and isinstance(closest_vals, dict):

                exp_depth = closest_vals['h']

                # Create Comparison Table HTML
                table_html = f"""
                <table class="comp-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>AI Optimized</th>
                            <th>Closest Experiment</th>
                            <th>Delta</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Pressure (P)</td>
                            <td class="highlight">{opt_P:.1f} MPa</td>
                            <td>{closest_vals['P']:.1f} MPa</td>
                            <td>{abs(opt_P - closest_vals['P']):.1f}</td>
                        </tr>
                        <tr>
                            <td>Flow (mf)</td>
                            <td class="highlight">{opt_mf:.3f} kg/min</td>
                            <td>{closest_vals['mf']:.3f} kg/min</td>
                            <td>{abs(opt_mf - closest_vals['mf']):.3f}</td>
                        </tr>
                        <tr>
                            <td>Speed (v)</td>
                            <td class="highlight">{opt_v:.1f} mm/min</td>
                            <td>{closest_vals['v']:.1f} mm/min</td>
                            <td>{abs(opt_v - closest_vals['v']):.1f}</td>
                        </tr>
                    </tbody>
                </table>
                """

                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown("---")

                col_verdict, col_result = st.columns([1, 2])

                with col_result:
                    st.write(f"**Closest Experiment Result:** Depth = `{exp_depth:.2f} mm`")
                    st.write(f"**Your Target:** Depth = `{target_depth:.2f} mm`")

                with col_verdict:
                    # Logic: Is the experimental depth close to our target?
                    # Note: Since the inputs are only "closest", the depth might differ significantly.
                    # This shows if the AI is interpolating well or extrapolating.
                    diff = abs(exp_depth - target_depth)

                    if diff < 5.0:
                        st.success("✅ **VALIDATED**")
                        st.caption("Matches historical data.")
                    else:
                        st.warning("⚠️ **EXTRAPOLATION**")
                        st.caption("No exact historical match.")
            else:
                st.error("Could not load experimental dataset for comparison.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    with col2:
        st.info("👈 Please configure Machine Setup above and click 'Optimize Parameters' to begin.")
        st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 50px;">
            <h3 style="color:#14b8a6;">Ready for Input</h3>
            <p>Select target depth to initialize the AI-Physics engine.</p>
        </div>
        """, unsafe_allow_html=True)