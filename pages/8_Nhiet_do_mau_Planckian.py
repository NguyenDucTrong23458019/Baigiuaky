import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import colour

st.set_page_config(layout="wide")
st.write("## Nhiệt độ màu (CCT) và quỹ tích Planckian")

CCS = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
mode = st.sidebar.radio("Chế độ", ["Nhập nhiệt độ màu (K)", "Nhập toạ độ xy"])

if mode == "Nhập nhiệt độ màu (K)":
    T = st.sidebar.slider("Nhiệt độ màu (K)", 1700, 25000, 6500, 100)
    xy = np.array(colour.CCT_to_xy(T, method="Kang 2002"))
else:
    x = st.sidebar.number_input("x", 0.0, 0.8, 0.3457, 0.001, format="%.4f")
    y = st.sidebar.number_input("y", 0.0, 0.9, 0.3585, 0.001, format="%.4f")
    xy = np.array([x, y])
    T = float(colour.xy_to_CCT(xy, method="McCamy 1992"))

XYZ = colour.xy_to_XYZ(xy)
rgb = np.clip(colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=True), 0, None)
rgb = np.clip(rgb / rgb.max(), 0, 1)
hexs = "#%02X%02X%02X" % tuple((rgb * 255).round().astype(int))

c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(show=False)
    ax.set_xlim([-0.1, 0.85])
    ax.set_ylim([-0.1, 0.90])
    Ts = np.linspace(1700, 25000, 200)
    loc = np.array([colour.CCT_to_xy(t, method="Kang 2002") for t in Ts])
    ax.plot(loc[:, 0], loc[:, 1], "k", lw=1.5, label="Quỹ tích Planckian")
    for ten, mk in (("D50", "s"), ("D65", "^"), ("A", "D")):
        ax.plot(*CCS[ten], mk, ms=8, mfc="white", mec="k", label=ten)
    ax.plot(*xy, "o", ms=10, mfc=rgb, mec="k", label="Màu đang chọn")
    ax.set_title("Quỹ tích Planckian trên CIE 1931")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)
with c2:
    st.markdown(f"<div style='background:{hexs};height:90px;border:1px solid #888;border-radius:8px;"
                f"text-align:center;line-height:90px;font-size:20px;'>{hexs}</div>", unsafe_allow_html=True)
    st.write(f"**CCT ≈ {T:.0f} K**,  x = {xy[0]:.4f},  y = {xy[1]:.4f}")
    if mode == "Nhập nhiệt độ màu (K)":
        sd = colour.sd_blackbody(float(T), colour.SpectralShape(360, 830, 5))
        v = np.array(sd.values)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(sd.wavelengths, v / v.max(), color="tab:red")
        ax2.set_xlabel("Bước sóng (nm)")
        ax2.set_ylabel("Công suất tương đối")
        ax2.set_title(f"Phổ vật đen tuyệt đối {T:.0f} K")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2, clear_figure=True)
    bang = pd.DataFrame({
        "Nguồn sáng": ["D50", "D65", "A"],
        "x": [CCS[t][0] for t in ("D50", "D65", "A")],
        "y": [CCS[t][1] for t in ("D50", "D65", "A")],
        "CCT (K)": [round(float(colour.xy_to_CCT(CCS[t], method="McCamy 1992"))) for t in ("D50", "D65", "A")]})
    st.table(bang)
st.caption("Ánh sáng ngành in thường dùng D50 (~5000 K) để xem màu; màn hình thường dùng D65 (~6500 K).")
