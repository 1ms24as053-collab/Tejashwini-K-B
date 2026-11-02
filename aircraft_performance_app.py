import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Aircraft Performance Analysis",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Aircraft Performance Analysis")
st.caption(
    "Compare aircraft performance at two air-density conditions. "
    "All calculations and default assumptions are based on the supplied script."
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("Analysis inputs")

st.sidebar.subheader("Air density")
rho_1 = st.sidebar.number_input(
    "Air density 1 (kg/m³)", min_value=0.1, max_value=2.0,
    value=1.112, step=0.001, format="%.3f"
)
rho_2 = st.sidebar.number_input(
    "Air density 2 (kg/m³)", min_value=0.1, max_value=2.0,
    value=0.819, step=0.001, format="%.3f"
)
densities = [rho_1, rho_2]

st.sidebar.subheader("Aircraft parameters")
S = st.sidebar.number_input("Wing area S (m²)", min_value=1.0, value=47.0)
AR = st.sidebar.number_input("Aspect ratio AR", min_value=0.1, value=6.5)
e = st.sidebar.number_input(
    "Oswald efficiency factor e", min_value=0.1, max_value=1.0,
    value=0.87, step=0.01
)
CD0 = st.sidebar.number_input(
    "Zero-lift drag coefficient CD₀", min_value=0.001,
    max_value=1.0, value=0.032, step=0.001, format="%.3f"
)
W = st.sidebar.number_input(
    "Aircraft weight W (N)", min_value=100.0, value=103047.0, step=100.0
)
num_engines = st.sidebar.number_input(
    "Number of engines", min_value=1, max_value=8, value=2, step=1
)
thrust_per_engine = st.sidebar.number_input(
    "Thrust per engine (N)", min_value=100.0, value=40298.0, step=100.0
)
velocity_min = st.sidebar.number_input(
    "Minimum velocity (m/s)", min_value=1.0, value=30.0
)
velocity_max = st.sidebar.number_input(
    "Maximum velocity (m/s)", min_value=velocity_min + 1.0, value=350.0
)
velocity_points = st.sidebar.slider(
    "Velocity points", min_value=100, max_value=1000, value=500, step=50
)

# =========================================================
# COMMON AIRCRAFT CALCULATIONS
# =========================================================
rho_sl = 1.225
T_A_sea_level = num_engines * thrust_per_engine
K = 1.0 / (np.pi * e * AR)
TR_min = 2 * W * np.sqrt(K * CD0)
velocity_range = np.linspace(velocity_min, velocity_max, velocity_points)

st.markdown("### Aircraft configuration")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wing area", f"{S:.2f} m²")
c2.metric("Aircraft weight", f"{W:,.0f} N")
c3.metric("Induced drag factor K", f"{K:.5f}")
c4.metric("Sea-level thrust available", f"{T_A_sea_level / 1000:.2f} kN")

# =========================================================
# 1. THRUST REQUIRED AND THRUST AVAILABLE
# =========================================================
st.header("1. Thrust required and thrust available")

fig1, ax1 = plt.subplots(figsize=(12, 7))
summary_thrust = []

for rho in densities:
    T_A_at_altitude = T_A_sea_level * (rho / rho_sl)
    TR = (
        0.5 * rho * velocity_range**2 * S * CD0
        + (2 * K * W**2) / (rho * velocity_range**2 * S)
    )
    V_TR_min = np.sqrt((2 * W) / (rho * S) * np.sqrt(K / CD0))

    ax1.plot(
        velocity_range, TR / 1000, linewidth=2,
        label=f"TR, ρ = {rho:.3f} kg/m³"
    )
    ax1.plot(
        V_TR_min, TR_min / 1000, "o", markersize=6,
        label=f"Min TR (V={V_TR_min:.1f} m/s, ρ={rho:.3f})"
    )
    ax1.plot(
        velocity_range,
        np.full(len(velocity_range), T_A_at_altitude / 1000),
        "--", linewidth=2.5,
        label=f"TA, ρ = {rho:.3f} (TA = {T_A_at_altitude / 1000:.2f} kN)"
    )

    summary_thrust.append({
        "Density (kg/m³)": rho,
        "TR_min (kN)": round(TR_min / 1000, 2),
        "V_TR_min (m/s)": round(V_TR_min, 2),
        "TA_at_altitude (kN)": round(T_A_at_altitude / 1000, 2),
    })

ax1.set_title("Thrust Required (TR) and Thrust Available (TA) vs Velocity")
ax1.set_xlabel("Velocity (m/s)")
ax1.set_ylabel("Thrust (kN)")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)
st.dataframe(pd.DataFrame(summary_thrust), use_container_width=True)

# =========================================================
# 2. POWER REQUIRED AND POWER AVAILABLE
# =========================================================
st.header("2. Power required and power available")

fig2, ax2 = plt.subplots(figsize=(12, 7))
summary_power = []

for rho in densities:
    T_A_at_altitude = T_A_sea_level * (rho / rho_sl)
    TR = (
        0.5 * rho * velocity_range**2 * S * CD0
        + (2 * K * W**2) / (rho * velocity_range**2 * S)
    )
    PR = TR * velocity_range
    PA = T_A_at_altitude * velocity_range

    V_PR_min = (
        (2 * K * W**2) /
        (1.5 * rho**2 * S**2 * CD0)
    ) ** (1 / 4)

    PR_min_val = (
        0.5 * rho * V_PR_min**3 * S * CD0
        + (2 * K * W**2) / (rho * V_PR_min * S)
    )

    ax2.plot(
        velocity_range, PR / 1000, linewidth=2,
        label=f"PR, ρ = {rho:.3f} kg/m³"
    )
    ax2.plot(
        velocity_range, PA / 1000, "--", linewidth=2.5,
        label=f"PA, ρ = {rho:.3f} kg/m³"
    )
    ax2.plot(
        V_PR_min, PR_min_val / 1000, "x", markersize=7,
        label=f"Min PR (V={V_PR_min:.1f} m/s, ρ={rho:.3f})"
    )

    summary_power.append({
        "Density (kg/m³)": rho,
        "PR_min (kW)": round(PR_min_val / 1000, 2),
        "V_PR_min (m/s)": round(V_PR_min, 2),
    })

ax2.set_title("Power Required (PR) and Power Available (PA) vs Velocity")
ax2.set_xlabel("Velocity (m/s)")
ax2.set_ylabel("Power (kW)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
st.dataframe(pd.DataFrame(summary_power), use_container_width=True)

# =========================================================
# 3. RATE OF CLIMB
# =========================================================
st.header("3. Rate of climb")

fig3, ax3 = plt.subplots(figsize=(12, 7))
summary_roc = []

for rho in densities:
    T_A_at_altitude = T_A_sea_level * (rho / rho_sl)
    TR = (
        0.5 * rho * velocity_range**2 * S * CD0
        + (2 * K * W**2) / (rho * velocity_range**2 * S)
    )
    PR = TR * velocity_range
    PA = T_A_at_altitude * velocity_range
    RoC = (PA - PR) / W

    valid_roc_indices = ~np.isnan(RoC)
    if np.any(valid_roc_indices):
        max_roc = np.max(RoC[valid_roc_indices])
        v_at_max_roc = velocity_range[valid_roc_indices][
            np.argmax(RoC[valid_roc_indices])
        ]
    else:
        max_roc = np.nan
        v_at_max_roc = np.nan

    ax3.plot(
        velocity_range, RoC, linewidth=2,
        label=f"RoC, ρ = {rho:.3f} kg/m³"
    )
    if not np.isnan(v_at_max_roc):
        ax3.plot(
            v_at_max_roc, max_roc, "s", markersize=6,
            label=f"Max RoC (V={v_at_max_roc:.1f} m/s, ρ={rho:.3f})"
        )

    summary_roc.append({
        "Density (kg/m³)": rho,
        "Max RoC (m/s)": round(max_roc, 2) if not np.isnan(max_roc) else "N/A",
        "V_at_Max_RoC (m/s)": (
            round(v_at_max_roc, 2) if not np.isnan(v_at_max_roc) else "N/A"
        ),
    })

ax3.set_title("Rate of Climb vs Velocity at Different Densities")
ax3.set_xlabel("Velocity (m/s)")
ax3.set_ylabel("Rate of Climb (m/s)")
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend()
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)
st.dataframe(pd.DataFrame(summary_roc), use_container_width=True)

# =========================================================
# 4. V-n DIAGRAM
# =========================================================
st.header("4. V-n diagram comparison")

W_N = W
S_ref = S
CL_max_pos = st.sidebar.number_input(
    "Positive maximum lift coefficient", min_value=0.1, value=1.5, step=0.1
)
CL_max_neg = st.sidebar.number_input(
    "Negative maximum lift coefficient", min_value=-2.0, max_value=-0.1,
    value=-0.8, step=0.1
)
n_max_pos = st.sidebar.number_input(
    "Positive structural limit (g)", min_value=0.1, value=5.0, step=0.5
)
n_max_neg = st.sidebar.number_input(
    "Negative structural limit (g)", min_value=-10.0, max_value=-0.1,
    value=-3.0, step=0.5
)

fig_vn_alt, ax_vn_alt = plt.subplots(figsize=(12, 8))
Vs_pos_min_rho = np.sqrt(
    (2 * W_N) / (min(densities) * S_ref * CL_max_pos)
)
Va_min_rho = Vs_pos_min_rho * np.sqrt(n_max_pos)
Vc_min_rho = 1.5 * Va_min_rho
Vne_min_rho = 1.2 * Vc_min_rho
V_limit = Vne_min_rho + 20

ax_vn_alt.axhline(
    n_max_pos, color="red", linestyle="-", linewidth=1.5,
    label=f"Pos. Limit Factor (+{n_max_pos:g}g)"
)
ax_vn_alt.axhline(
    n_max_neg, color="red", linestyle="--", linewidth=1.5,
    label=f"Neg. Limit Factor ({n_max_neg:g}g)"
)
ax_vn_alt.axhline(
    1.0, color="gray", linestyle=":", alpha=0.5, label="1g Level Flight"
)
ax_vn_alt.axhline(0.0, color="black", linestyle="-", alpha=0.3)

colors = ["blue", "purple"]
vn_summary = []

for rho, color in zip(densities, colors):
    Vs_pos = np.sqrt((2 * W_N) / (rho * S_ref * CL_max_pos))
    Vs_neg = np.sqrt((2 * W_N) / (rho * S_ref * abs(CL_max_neg)))
    Va = Vs_pos * np.sqrt(n_max_pos)
    Vc = 1.5 * Va
    Vne = 1.2 * Vc

    V_pos = np.linspace(0, Va, 200)
    n_stall_pos = (0.5 * rho * V_pos**2 * S_ref * CL_max_pos) / W_N

    V_neg_end = Vs_neg * np.sqrt(abs(n_max_neg))
    V_neg = np.linspace(0, V_neg_end, 200)
    n_stall_neg = -(
        0.5 * rho * V_neg**2 * S_ref * abs(CL_max_neg)
    ) / W_N

    ax_vn_alt.plot(
        V_pos, n_stall_pos, color=color, linestyle="-", linewidth=2,
        label=f"Pos Stall (ρ = {rho:.3f})"
    )
    ax_vn_alt.plot(
        V_neg, n_stall_neg, color=color, linestyle="--", linewidth=2,
        label=f"Neg Stall (ρ = {rho:.3f})"
    )
    ax_vn_alt.plot(
        [Vne, Vne], [n_max_neg, n_max_pos], color=color,
        linestyle=":", linewidth=2,
        label=f"Vne Limit ({Vne:.1f} m/s, ρ={rho:.3f})"
    )
    ax_vn_alt.plot(
        [Va, Vne], [n_max_pos, n_max_pos],
        color=color, linestyle="-.", alpha=0.7
    )
    ax_vn_alt.plot(
        [V_neg_end, Vc], [n_max_neg, n_max_neg],
        color=color, linestyle="-.", alpha=0.7
    )
    ax_vn_alt.plot(
        [Vc, Vne], [n_max_neg, 0],
        color=color, linestyle="-.", alpha=0.7
    )

    vn_summary.append({
        "Density (kg/m³)": rho,
        "Positive stall speed (m/s)": round(Vs_pos, 2),
        "Negative stall speed (m/s)": round(Vs_neg, 2),
        "Corner velocity Va (m/s)": round(Va, 2),
        "Cruise/design velocity Vc (m/s)": round(Vc, 2),
        "Vne (m/s)": round(Vne, 2),
    })

ax_vn_alt.set_title(
    "V-n Diagram Comparison for Different Densities",
    fontsize=14, fontweight="bold"
)
ax_vn_alt.set_xlabel("Velocity V (m/s)", fontsize=12)
ax_vn_alt.set_ylabel("Load Factor n (g)", fontsize=12)
ax_vn_alt.set_xlim(0, V_limit)
ax_vn_alt.set_ylim(n_max_neg - 0.5, n_max_pos + 0.5)
ax_vn_alt.grid(True, which="both", linestyle=":", alpha=0.5)
ax_vn_alt.legend(
    loc="upper left", bbox_to_anchor=(1.01, 1),
    borderaxespad=0, facecolor="white", framealpha=0.9
)
fig_vn_alt.tight_layout()
st.pyplot(fig_vn_alt)
plt.close(fig_vn_alt)
st.dataframe(pd.DataFrame(vn_summary), use_container_width=True)

# =========================================================
# 5. GLIDING FLIGHT PERFORMANCE
# =========================================================
st.header("5. Gliding flight performance")

cd0_val = st.number_input(
    "Glide analysis CD₀", min_value=0.001, value=0.032,
    step=0.001, format="%.3f"
)
k_val = st.number_input(
    "Glide analysis induced drag factor k", min_value=0.001,
    value=0.0563, step=0.0001, format="%.4f"
)

rho_sl_std = 1.225
T0 = 288.15
L = 0.0065
g_std = 9.80665
R_air = 287.05
exponent = (g_std / (R_air * L)) - 1.0


def density_to_altitude_km(rho):
    """Approximate tropospheric altitude in km from air density."""
    temp_ratio = (rho / rho_sl_std) ** (1.0 / exponent)
    h_m = (T0 / L) * (1.0 - temp_ratio)
    return h_m / 1000.0


def calculate_gliding_flight(cd0, k, h_km):
    """Calculate maximum L/D, minimum glide angle and maximum glide range."""
    cl_cd_max = 1.0 / (2.0 * math.sqrt(cd0 * k))
    theta_min_rad = math.atan(1.0 / cl_cd_max)
    theta_min_deg = math.degrees(theta_min_rad)
    r_max = h_km * cl_cd_max
    return {
        "cl_cd_max": cl_cd_max,
        "theta_min_deg": theta_min_deg,
        "r_max_km": r_max,
    }


gliding_results = []
for rho in densities:
    h_approx_km = density_to_altitude_km(rho)
    res = calculate_gliding_flight(cd0_val, k_val, h_approx_km)
    gliding_results.append({
        "Density (kg/m³)": rho,
        "Approx. Altitude (km)": round(h_approx_km, 3),
        "(CL/CD)_max": round(res["cl_cd_max"], 4),
        "Min Glide Angle (°)": round(res["theta_min_deg"], 2),
        "Max Glide Range (km)": round(res["r_max_km"], 3),
    })

st.dataframe(pd.DataFrame(gliding_results), use_container_width=True)

# =========================================================
# 6. TAKEOFF TRAJECTORY COMPARISON
# =========================================================
st.header("6. Takeoff trajectory comparison")

st.caption(
    "Ground-roll, transition-radius and airborne-distance values are scaled "
    "from the baseline values supplied in the original script."
)

CL_max_takeoff = st.number_input(
    "Takeoff maximum lift coefficient", min_value=0.1,
    value=2.8, step=0.1
)
S_g_baseline = 789.32
rho_baseline = 0.819
R_baseline = 1356.64
S_a_baseline = 364.92
g = 9.81

takeoff_results = []
for rho in densities:
    V_stall = math.sqrt((2.0 * W) / (rho * S * CL_max_takeoff))
    S_g = S_g_baseline * (rho_baseline / rho)
    R = R_baseline * (rho_baseline / rho)
    S_a = S_a_baseline * (rho_baseline / rho)
    S_to = S_g + S_a

    takeoff_results.append({
        "Density (kg/m³)": rho,
        "Stall Velocity (m/s)": round(V_stall, 2),
        "Ground Roll S_g (m)": round(S_g, 2),
        "Transition Radius R (m)": round(R, 2),
        "Airborne Distance S_a (m)": round(S_a, 2),
        "Total Takeoff S_to (m)": round(S_to, 2),
    })

st.dataframe(pd.DataFrame(takeoff_results), use_container_width=True)

st.divider()
st.caption(
    "Note: This app reproduces the equations, assumptions and baseline scaling "
    "from the supplied Python script. Validate the underlying aerodynamic and "
    "takeoff-performance assumptions before using results for engineering or "
    "operational decisions."
)
