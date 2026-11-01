import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import streamlit as st

st.set_page_config(layout="wide")
st.write("## Tăng giá trị tông (dot gain) của máy in nửa tông")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = scipy.io.loadmat(os.path.join(ROOT, "halftone.mat"))
P = np.vstack((data["W"], data["P"]))
wl = np.arange(400, 701, 10)
a_target = np.linspace(0, 1, 11)

yellow = P[:11, :]
magenta = P[0:111:11, :]            # hàng 0, 11, 22, ..., 110

muc = st.sidebar.radio("Chọn mực", ["Vàng", "Tím (Magenta)"])
n = st.sidebar.slider("Hệ số Yule–Nielsen n (n = 1 là Murray–Davies)", 1.0, 30.0, 20.0, 0.5)
R = yellow if muc == "Vàng" else magenta
lam = st.sidebar.slider("Bước sóng đo mật độ (nm)", 400, 700, 450 if muc == "Vàng" else 550, 10)


def phu_yn(R, n):
    """Diện tích hạt hiệu dụng theo Yule–Nielsen, dùng toàn bộ phổ."""
    W, S = R[0] ** (1 / n), R[-1] ** (1 / n)
    Rn = R ** (1 / n)
    c = np.array([np.sum((S - r) * (S - W)) / np.sum((S - W) ** 2) for r in Rn])
    return 1 - c


def phu_md_mat_do(R, lam):
    """Murray–Davies theo mật độ tại 1 bước sóng."""
    i = (lam - 400) // 10
    D = -np.log10(np.clip(R[:, i], 1e-4, None))
    Dp, Ds = D[0], D[-1]
    return (1 - 10 ** (-(D - Dp))) / (1 - 10 ** (-(Ds - Dp)))


a_yn = phu_yn(R, n)
a_md = phu_md_mat_do(R, lam)

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k:", label="Lý tưởng")
    ax.plot(a_target, a_yn, "o-", color="tab:orange", label=f"Yule–Nielsen (n={n:g})")
    ax.plot(a_target, a_md, "s--", color="tab:blue", label=f"Murray–Davies @ {lam} nm")
    ax.set_xlabel("Tỷ lệ hạt trame mục tiêu")
    ax.set_ylabel("Tỷ lệ hạt thực tế")
    ax.set_title(f"Đường cong tái tạo tông – mực {muc}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)
with g2:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(a_target * 100, (a_yn - a_target) * 100, "o-", color="tab:orange", label="Yule–Nielsen")
    ax.plot(a_target * 100, (a_md - a_target) * 100, "s--", color="tab:blue", label="Murray–Davies")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Tỷ lệ hạt trame mục tiêu (%)")
    ax.set_ylabel("Dot gain (%)")
    ax.set_title("Tăng giá trị tông (dot gain)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

bang = pd.DataFrame({"Mục tiêu (%)": (a_target * 100).round(0).astype(int),
                     "Thực tế YN (%)": (a_yn * 100).round(1),
                     "Dot gain YN (%)": ((a_yn - a_target) * 100).round(1),
                     "Thực tế MD (%)": (a_md * 100).round(1),
                     "Dot gain MD (%)": ((a_md - a_target) * 100).round(1)})
st.dataframe(bang, hide_index=True)
i50 = 5
st.success(f"Tại tông 50%: dot gain Yule–Nielsen = {(a_yn[i50]-0.5)*100:.1f}%, "
           f"Murray–Davies = {(a_md[i50]-0.5)*100:.1f}%.")
st.caption("Chênh lệch giữa hai đường cho thấy ảnh hưởng của hiện tượng tán xạ ánh sáng trong giấy (optical dot gain).")
