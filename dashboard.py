import streamlit as st
import fitparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np
import spm1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Interval Lab", layout="wide")

# --- SESSION STATE MANAGEMENT ---
if 'spm_window_val' not in st.session_state:
    st.session_state.spm_window_val = (0.0, 60.0)
if 'reg_window_val' not in st.session_state:
    st.session_state.reg_window_val = (0.0, 60.0)

# --- CALLBACKS ---
def sync_spm_from_slider():
    st.session_state.spm_window_val = st.session_state.spm_slider_key

def sync_spm_from_box():
    new_duration = st.session_state.spm_box_key
    st.session_state.spm_window_val = (0.0, float(new_duration))

def sync_reg_from_slider():
    st.session_state.reg_window_val = st.session_state.reg_slider_key

def sync_reg_from_box():
    new_duration = st.session_state.reg_box_key
    st.session_state.reg_window_val = (0.0, float(new_duration))

# --- HELPER FUNCTIONS ---

def load_fit_data(file_bytes):
    fitfile = fitparse.FitFile(io.BytesIO(file_bytes))
    records = []
    laps = []
    for record in fitfile.get_messages("lap"):
        laps.append({
            'start_time': record.get_value('start_time'),
            'timestamp': record.get_value('timestamp'),
            'total_distance': record.get_value('total_distance'),
            'avg_speed': record.get_value('enhanced_avg_speed')
        })
    for record in fitfile.get_messages("record"):
        ts = record.get_value('timestamp')
        speed = record.get_value('enhanced_speed')
        if ts and speed is not None:
            records.append({
                'timestamp': ts,
                'speed_mps': speed
            })
    return pd.DataFrame(records), pd.DataFrame(laps)

def process_data_10hz(df):
    if df.empty: return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to 10Hz
    df_resampled = df.resample('100ms').asfreq()
    df_10hz = df_resampled.copy()
    
    # Interpolate Speed
    df_10hz['speed_mps'] = df_resampled['speed_mps'].interpolate(method='cubic')
    
    # Calculate Acceleration
    df_10hz['acceleration'] = df_10hz['speed_mps'].diff() / 0.1
    df_10hz['acceleration'] = df_10hz['acceleration'].rolling(window=5, center=True).mean()

    # Units
    df_10hz['speed_kph'] = df_10hz['speed_mps'] * 3.6
    
    return df_10hz

def get_stats(df_chunk):
    if df_chunk.empty: return None
    return {
        "Peak Speed": round(df_chunk['speed_kph'].max(), 2),
        "Avg Speed": round(df_chunk['speed_kph'].mean(), 2),
        "Max Accel": round(df_chunk['acceleration'].max(), 2),
        "Duration": round(len(df_chunk) / 10, 1)
    }

def normalize_laps_strict(df_10hz, laps_df, selected_lap_names, start_trim, end_trim, n_points=100):
    normalized_curves = []
    valid_lap_names = []
    warnings = []
    tolerance = 1.0 
    
    for lap_name in selected_lap_names:
        lap_idx = int(lap_name.split(" ")[1]) - 1
        lap_data = laps_df.iloc[lap_idx]
        start_ts = pd.to_datetime(lap_data['start_time'])
        end_ts = pd.to_datetime(lap_data['timestamp'])
        
        chunk = df_10hz[(df_10hz.index >= start_ts) & (df_10hz.index <= end_ts)].copy()
        
        if not chunk.empty:
            chunk['seconds'] = (chunk.index - chunk.index[0]).total_seconds()
            
            # Check length
            if chunk['seconds'].max() < (end_trim - tolerance):
                warnings.append(f"❌ **{lap_name}** dropped: Too short.")
                continue
                
            # Trim
            trimmed_chunk = chunk[(chunk['seconds'] >= start_trim) & (chunk['seconds'] <= end_trim)]
            
            if len(trimmed_chunk) > 10: 
                y = trimmed_chunk['speed_kph'].values # Only Speed used now
                x = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, n_points)
                y_new = np.interp(x_new, x, y)
                normalized_curves.append(y_new)
                valid_lap_names.append(lap_name)
                
    return np.array(normalized_curves), valid_lap_names, warnings

def lap_checkbox_grid(options, key_prefix, num_columns=4, pre_selected_indices=None):
    selected_items = []
    cols = st.columns(num_columns)
    for i, option in enumerate(options):
        default = False
        if pre_selected_indices and i in pre_selected_indices:
            default = True
        col = cols[i % num_columns]
        if col.checkbox(option, value=default, key=f"{key_prefix}_{i}"):
            selected_items.append(option)
    return selected_items

def find_structural_break(x, y):
    n = len(x)
    if n < 10: return None, None, None, None
    best_err = np.inf
    best_k = -1
    search_start = int(n * 0.15)
    search_end = int(n * 0.85)
    for k in range(search_start, search_end):
        p1 = np.polyfit(x[:k], y[:k], 1)
        p2 = np.polyfit(x[k:], y[k:], 1)
        fit1 = np.polyval(p1, x[:k])
        fit2 = np.polyval(p2, x[k:])
        err = np.sum((y[:k] - fit1)**2) + np.sum((y[k:] - fit2)**2)
        if err < best_err:
            best_err = err
            best_k = k
            best_p1 = p1
            best_p2 = p2
    return best_k, best_p1, best_p2, best_err

# --- APP LAYOUT ---

st.title("Interval Lab")

st.markdown("""
### How to use this app
This tool is designed to analyze the physics of your speed during interval sessions.

1.  **Upload Data:** Drag and drop your Garmin **.FIT** file below.
2.  **Performance Tab:**
    * **Single Lap:** Use this to see your "Acceleration Profile." The app will auto-detect when you stopped driving and started holding (Break Point).
    * **Overlay:** Select multiple laps to see them stacked on top of each other.
3.  **Group Hypothesis Tab:** * Select a "Group A" (e.g., Early Laps) and "Group B" (e.g., Late Laps).
    * The app will run a statistical test (SPM) to prove exactly *where* in the interval you are getting slower.
4.  **Fatigue Trend Tab:**
    * Select a full sequence (e.g., Laps 1 to 10).
    * The app calculates a "Slope of Decay" for every second of the rep to show how much speed you lose per lap.
""")

st.divider()

uploaded_file = st.file_uploader("Upload .FIT file", type=['fit'])

if uploaded_file is not None:
    with st.spinner('Crunching numbers...'):
        try:
            raw_df, laps_df = load_fit_data(uploaded_file.read())
            
            if raw_df.empty:
                st.error("No speed data found in file.")
            else:
                df_10hz = process_data_10hz(raw_df)
                lap_options = [f"Lap {i+1}" for i in range(len(laps_df))]

                tab1, tab2, tab3 = st.tabs(["📊 Performance", "🧬 Group Hypothesis (SPM)", "📉 Fatigue Trend (Regression)"])

                # ================= TAB 1: PERFORMANCE =================
                with tab1:
                    mode = st.radio("View Mode", ["Single Lap (Breakpoint Analysis)", "Multi-Lap Overlay"], horizontal=True)
                    
                    if mode == "Single Lap (Breakpoint Analysis)":
                        with st.expander("ℹ️ How to interpret Structural Breakpoint"):
                            st.markdown("This finds the exact moment your strategy changed from 'Accelerating' to 'Holding'. Ideally, the second phase should be flat (Slope 0.00).")

                        selected_lap_name = st.selectbox("Select Interval:", lap_options)
                        lap_idx = int(selected_lap_name.split(" ")[1]) - 1
                        lap_data = laps_df.iloc[lap_idx]
                        start_ts = pd.to_datetime(lap_data['start_time'])
                        end_ts = pd.to_datetime(lap_data['timestamp'])
                        lap_chunk = df_10hz[(df_10hz.index >= start_ts) & (df_10hz.index <= end_ts)].copy()
                        lap_chunk['seconds'] = (lap_chunk.index - lap_chunk.index[0]).total_seconds()
                        
                        start_sec, end_sec = st.slider("Trim Window", 0.0, float(lap_chunk['seconds'].max()), (0.0, float(lap_chunk['seconds'].max())), 0.1)
                        zoom_chunk = lap_chunk[(lap_chunk['seconds'] >= start_sec) & (lap_chunk['seconds'] <= end_sec)]
                        stats = get_stats(zoom_chunk)
                        
                        if stats:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Peak Speed", f"{stats['Peak Speed']} km/h")
                            c2.metric("Avg Speed", f"{stats['Avg Speed']} km/h")
                            c3.metric("Max Accel", f"{stats['Max Accel']} m/s²")
                            c4.metric("Time", f"{stats['Duration']} s")
                        
                        # BREAKPOINT
                        x_arr = zoom_chunk['seconds'].values
                        y_arr = zoom_chunk['speed_kph'].values
                        
                        bk_idx, p1, p2, err = find_structural_break(x_arr, y_arr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='lines', name='Actual Speed', line=dict(color='#00CC96', width=3)))
                        
                        if bk_idx:
                            break_time = x_arr[bk_idx]
                            fit1_x = x_arr[:bk_idx]
                            fit1_y = np.polyval(p1, fit1_x)
                            fit2_x = x_arr[bk_idx:]
                            fit2_y = np.polyval(p2, fit2_x)
                            
                            fig.add_trace(go.Scatter(x=fit1_x, y=fit1_y, mode='lines', name='Phase 1 Trend', line=dict(color='yellow', dash='dash')))
                            fig.add_trace(go.Scatter(x=fit2_x, y=fit2_y, mode='lines', name='Phase 2 Trend', line=dict(color='orange', dash='dash')))
                            fig.add_vline(x=break_time, line_width=2, line_dash="dot", line_color="white", annotation_text="Break Point")
                            
                            slope1 = p1[0]
                            slope2 = p2[0]
                            st.info(f"⚡ **Structural Break Detected at {break_time:.1f}s**")
                            c1, c2 = st.columns(2)
                            c1.metric("Phase 1 Slope", f"{slope1:.2f}")
                            c2.metric("Phase 2 Slope", f"{slope2:.2f}", delta=f"{slope2-slope1:.2f}")

                        fig.update_layout(template="plotly_dark", hovermode="x unified", title="Single Lap Structural Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.subheader("Select Laps to Overlay")
                        with st.expander("Show/Hide Lap Selection Grid", expanded=True):
                            selected_laps = lap_checkbox_grid(lap_options, "perf_grid", num_columns=5, pre_selected_indices=[0, 1])
                        
                        if selected_laps:
                            max_dur = 0
                            for l in selected_laps:
                                l_idx = int(l.split(" ")[1]) - 1
                                dur = (pd.to_datetime(laps_df.iloc[l_idx]['timestamp']) - pd.to_datetime(laps_df.iloc[l_idx]['start_time'])).total_seconds()
                                if dur > max_dur: max_dur = dur
                            start_trim, end_trim = st.slider("Global Trim (Seconds)", 0.0, float(max_dur), (0.0, float(max_dur)), 0.1)
                            
                            fig = go.Figure()
                            for lap_name in selected_laps:
                                lap_idx = int(lap_name.split(" ")[1]) - 1
                                chunk = df_10hz[(df_10hz.index >= pd.to_datetime(laps_df.iloc[lap_idx]['start_time'])) & 
                                                (df_10hz.index <= pd.to_datetime(laps_df.iloc[lap_idx]['timestamp']))].copy()
                                chunk['seconds'] = (chunk.index - chunk.index[0]).total_seconds()
                                chunk = chunk[(chunk['seconds'] >= start_trim) & (chunk['seconds'] <= end_trim)]
                                if not chunk.empty:
                                    fig.add_trace(go.Scatter(x=chunk['seconds'], y=chunk['speed_kph'], mode='lines', name=lap_name))
                            fig.update_layout(template="plotly_dark", title="Lap Speed Overlay (Aligned)", xaxis_title="Seconds", yaxis_title="Speed (km/h)")
                            st.plotly_chart(fig, use_container_width=True)

                # ================= TAB 2: SPM GROUP TEST =================
                with tab2:
                    st.header("Group Hypothesis Testing (A vs B)")
                    with st.expander("ℹ️ How to interpret SPM Analysis"):
                        st.markdown("""
                        **Statistical Parametric Mapping (SPM)** compares the entire speed curve of Group A vs Group B.
                        * **Grey Bars:** Indicate where the speed difference is statistically significant (p < 0.05).
                        * **Purple Chart:** Shows Effect Size (Cohen's d). Values > 0.8 indicate a physically large difference, even if the P-value says otherwise (common in small sample sizes).
                        """)

                    c1, c2 = st.columns(2)
                    with c1: 
                        st.markdown("#### Group A (Control)")
                        ga = lap_checkbox_grid(lap_options, "ga_grid", num_columns=3, pre_selected_indices=[0, 1])
                    with c2: 
                        st.markdown("#### Group B (Test)")
                        gb = lap_checkbox_grid(lap_options, "gb_grid", num_columns=3, pre_selected_indices=[len(lap_options)-2, len(lap_options)-1])
                    
                    all_sel = ga + gb
                    max_dur = 60.0
                    if all_sel:
                         for l in all_sel:
                            lid = int(l.split(" ")[1]) - 1
                            dur = (pd.to_datetime(laps_df.iloc[lid]['timestamp']) - pd.to_datetime(laps_df.iloc[lid]['start_time'])).total_seconds()
                            if dur > max_dur: max_dur = dur
                    
                    col_slider, col_box = st.columns([3, 1])
                    with col_box:
                        box_val = st.number_input("Quick Duration (s)", 
                                        min_value=1.0, max_value=float(max_dur), 
                                        value=st.session_state.spm_window_val[1], 
                                        key="spm_box_key", on_change=sync_spm_from_box)
                    with col_slider:
                        slider_val = st.slider("Window (s)", 0.0, float(max_dur), 
                                  value=st.session_state.spm_window_val, 
                                  key="spm_slider_key", on_change=sync_spm_from_slider)
                    
                    spm_win = st.session_state.spm_window_val

                    test_type = st.radio("Test Type:", ["Parametric (T-Test)", "Non-Parametric (SnPM - Permutation)"], horizontal=True)

                    if st.button("Run Statistical Test"):
                        if len(ga)<2 or len(gb)<2: st.error("Need 2+ laps per group.")
                        else:
                            Y1, v1, w1 = normalize_laps_strict(df_10hz, laps_df, ga, spm_win[0], spm_win[1], n_points=100)
                            Y2, v2, w2 = normalize_laps_strict(df_10hz, laps_df, gb, spm_win[0], spm_win[1], n_points=100)
                            if w1+w2: 
                                with st.expander("Warnings"): 
                                    for w in w1+w2: st.write(w)
                            
                            if len(Y1)<2 or len(Y2)<2: st.error("Not enough data.")
                            else:
                                if "Parametric" in test_type:
                                    t = spm1d.stats.ttest2(Y1, Y2, equal_var=False)
                                    ti = t.inference(alpha=0.05, two_tailed=True)
                                else:
                                    with st.spinner("Running 1000 Permutations..."):
                                        t = spm1d.stats.nonparam.ttest2(Y1, Y2)
                                        ti = t.inference(alpha=0.05, two_tailed=True, iterations=1000)

                                st.markdown("### 📝 Conclusion")
                                if ti.h0reject:
                                    ranges = []
                                    for c in ti.clusters:
                                        s,e = c.endpoints
                                        ts = spm_win[0]+(s/100)*(spm_win[1]-spm_win[0])
                                        te = spm_win[0]+(e/100)*(spm_win[1]-spm_win[0])
                                        ranges.append(f"**{ts:.1f}s to {te:.1f}s**")
                                    st.error(f"⚠️ **Significant Difference Detected!** Range: {' and '.join(ranges)}.")
                                else:
                                    st.success(f"✅ **No Significant Difference.**")

                                fig, ax = plt.subplots(figsize=(10,5))
                                x = np.linspace(spm_win[0], spm_win[1], 100)
                                ax.plot(x, Y1.mean(0), 'b', label="Group A Mean")
                                ax.fill_between(x, Y1.mean(0)-Y1.std(0), Y1.mean(0)+Y1.std(0), color='b', alpha=0.1)
                                ax.plot(x, Y2.mean(0), 'r', label="Group B Mean")
                                ax.fill_between(x, Y2.mean(0)-Y2.std(0), Y2.mean(0)+Y2.std(0), color='r', alpha=0.1)
                                if ti.h0reject:
                                    for c in ti.clusters:
                                        s,e = c.endpoints
                                        ax.axvspan(spm_win[0]+(s/100)*(spm_win[1]-spm_win[0]), spm_win[0]+(e/100)*(spm_win[1]-spm_win[0]), color='gray', alpha=0.3)
                                ax.legend(); ax.set_title("Speed Comparison (km/h)"); st.pyplot(fig)

                                # --- EFFECT SIZE (ALWAYS VISIBLE) ---
                                st.markdown("### 🔍 Effect Size (Magnitude of Difference)")
                                
                                mean1, mean2 = Y1.mean(0), Y2.mean(0)
                                sd1, sd2 = Y1.std(0, ddof=1), Y2.std(0, ddof=1)
                                n1, n2 = len(Y1), len(Y2)
                                pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
                                pooled_sd = np.where(pooled_sd == 0, 1e-9, pooled_sd)
                                cohens_d = (mean1 - mean2) / pooled_sd
                                
                                fig_d, ax_d = plt.subplots(figsize=(10, 3))
                                ax_d.plot(x, np.abs(cohens_d), color='purple', linewidth=2, label="Cohen's d (Abs)")
                                ax_d.axhline(0.8, color='orange', linestyle='--', label="Large Effect (0.8)")
                                ax_d.axhline(1.2, color='red', linestyle='--', label="Very Large Effect (1.2)")
                                ax_d.fill_between(x, 0, np.abs(cohens_d), color='purple', alpha=0.1)
                                ax_d.set_ylabel("Effect Size (d)")
                                ax_d.set_xlabel("Time (s)")
                                ax_d.legend(loc='upper right')
                                st.pyplot(fig_d)

                # ================= TAB 3: REGRESSION (FATIGUE TREND) =================
                with tab3:
                    st.header("📉 Fatigue Trend (1D Linear Regression)")
                    with st.expander("ℹ️ How to interpret 1D Regression"):
                        st.markdown("""
                        **What does this do?** Calculates the linear rate of change (Slope) for every second of the interval.
                        * **Purple Line:** Shows the change in km/h *per lap*.
                        * **Below 0:** You are getting slower (Fatigue).
                        * **Above 0:** You are getting faster (Pacing drift).
                        * **Shaded Zones:** The trend is statistically significant.
                        """)
                    
                    st.subheader("Select Sequence")
                    with st.expander("Select Laps for Trend Analysis", expanded=True):
                        reg_laps = lap_checkbox_grid(lap_options, "reg_grid", num_columns=5, pre_selected_indices=list(range(5)))
                    
                    max_reg = 60.0
                    if reg_laps:
                         for l in reg_laps:
                            lid = int(l.split(" ")[1]) - 1
                            dur = (pd.to_datetime(laps_df.iloc[lid]['timestamp']) - pd.to_datetime(laps_df.iloc[lid]['start_time'])).total_seconds()
                            if dur > max_reg: max_reg = dur
                    
                    col_reg_slider, col_reg_box = st.columns([3, 1])
                    with col_reg_box:
                        st.number_input("Quick Duration (s)", 
                                        min_value=1.0, max_value=float(max_reg), 
                                        value=st.session_state.reg_window_val[1], 
                                        key="reg_box_key", on_change=sync_reg_from_box)
                    with col_reg_slider:
                        st.slider("Analysis Window (s)", 0.0, float(max_reg), 
                                  value=st.session_state.reg_window_val, 
                                  key="reg_slider_key", on_change=sync_reg_from_slider)
                    
                    reg_win = st.session_state.reg_window_val

                    if st.button("Run Regression"):
                        if len(reg_laps) < 3: st.error("Need 3+ laps.")
                        else:
                            Y, valid_laps, warns = normalize_laps_strict(df_10hz, laps_df, reg_laps, reg_win[0], reg_win[1], n_points=100)
                            if len(Y) < 3: st.error("Not enough valid laps.")
                            else:
                                if warns: 
                                    with st.expander("Warnings"): 
                                        for w in warns: st.write(w)
                                x = np.array([int(l.split(" ")[1]) for l in valid_laps])
                                t = spm1d.stats.regress(Y, x)
                                ti = t.inference(alpha=0.05)

                                manual_slopes = []
                                n_timepoints = Y.shape[1]
                                for i in range(n_timepoints):
                                    m, c = np.polyfit(x, Y[:, i], 1)
                                    manual_slopes.append(m)
                                manual_slope_curve = np.array(manual_slopes)

                                st.markdown("### 📝 Detailed Interpretation")
                                if ti.h0reject:
                                    ranges = []
                                    for c in ti.clusters:
                                        s,e = c.endpoints
                                        s, e = int(s), int(e)
                                        ts = reg_win[0] + (s/100)*(reg_win[1]-reg_win[0])
                                        te = reg_win[0] + (e/100)*(reg_win[1]-reg_win[0])
                                        
                                        cluster_slope = manual_slope_curve[s:e].mean()
                                        direction = "📉 DECLINE" if cluster_slope < 0 else "📈 INCREASE"
                                        insight = "(Fatigue)" if cluster_slope < 0 else ""
                                        
                                        st.error(f"**{direction} Detected {insight}:** From **{ts:.1f}s to {te:.1f}s**, speed changes by an average of **{cluster_slope:.1f} km/h per lap**.")
                                else:
                                    st.success("✅ **Stable Performance.** No statistically significant linear trend.")

                                fig_reg, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                                time_axis = np.linspace(reg_win[0], reg_win[1], 100)
                                for curve in Y: ax1.plot(time_axis, curve, alpha=0.3, color='gray')
                                ax1.plot(time_axis, Y.mean(0), color='black', linewidth=2); ax1.set_title(f"Raw Curves"); ax1.set_ylabel("Speed (km/h)")
                                
                                ax2.plot(time_axis, manual_slope_curve, color='purple', linewidth=2, label="Rate of Change (Slope)"); ax2.axhline(0, color='black', linestyle='--')
                                if ti.h0reject:
                                    for c in ti.clusters:
                                        s, e = c.endpoints
                                        ax2.axvspan(reg_win[0]+(s/100)*(reg_win[1]-reg_win[0]), reg_win[0]+(e/100)*(reg_win[1]-reg_win[0]), color='purple', alpha=0.2)
                                ax2.set_ylabel("Change per Lap"); ax2.set_xlabel("Time (s)"); st.pyplot(fig_reg)

        except Exception as e:
            st.error(f"Error: {e}")