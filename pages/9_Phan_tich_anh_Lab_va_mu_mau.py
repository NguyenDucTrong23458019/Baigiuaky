import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import colour
from PIL import Image

st.set_page_config(layout="wide")
st.write("## Phân tích ảnh: kênh L\\*a\\*b\\* và mô phỏng mù màu")

D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
up = st.sidebar.file_uploader("Upload ảnh (png/jpg)", type=["png", "jpg", "jpeg"])

if up is not None:
    img = Image.open(up).convert("RGB")
    img.thumbnail((480, 480))
    arr = np.asarray(img) / 255.0
else:
    h, w = 240, 480
    H = np.tile(np.linspace(0, 1, w), (h, 1))
    V = np.tile(np.linspace(1, 0.25, h)[:, None], (1, w))
    arr = colour.HSV_to_RGB(np.dstack([H, np.ones((h, w)), V]))
    st.info("Chưa upload ảnh, đang dùng ảnh mẫu (dải màu cầu vồng).")

Lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(arr), illuminant=D65)
L, A, B = Lab[..., 0], Lab[..., 1], Lab[..., 2]

tab1, tab2 = st.tabs(["Các kênh L*, a*, b*", "Mô phỏng mù màu"])

with tab1:
    cmap_a = LinearSegmentedColormap.from_list("ga", ["green", "lightgray", "red"])
    cmap_b = LinearSegmentedColormap.from_list("by", ["blue", "lightgray", "gold"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].imshow(arr); axes[0].set_title("Ảnh gốc")
    axes[1].imshow(L, cmap="gray", vmin=0, vmax=100); axes[1].set_title("L* (độ sáng)")
    axes[2].imshow(A, cmap=cmap_a, vmin=-80, vmax=80); axes[2].set_title("a* (lục ↔ đỏ)")
    axes[3].imshow(B, cmap=cmap_b, vmin=-80, vmax=80); axes[3].set_title("b* (lam ↔ vàng)")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig, clear_figure=True)

    g1, g2 = st.columns(2)
    with g1:
        rng = np.random.default_rng(0)
        idx = rng.choice(L.size, size=min(4000, L.size), replace=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(A.ravel()[idx], B.ravel()[idx], c=arr.reshape(-1, 3)[idx], s=6)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("a*"); ax.set_ylabel("b*")
        ax.set_title("Phân bố màu của ảnh trên mặt phẳng a*b*")
        st.pyplot(fig, clear_figure=True)
    with g2:
        st.write("### Thống kê Lab của ảnh")
        st.table({"Kênh": ["L*", "a*", "b*"],
                  "Trung bình": [round(float(L.mean()), 2), round(float(A.mean()), 2), round(float(B.mean()), 2)],
                  "Độ lệch chuẩn": [round(float(L.std()), 2), round(float(A.std()), 2), round(float(B.std()), 2)]})
        C = np.hypot(A, B)
        st.write(f"Độ bão hoà trung bình C\\* = **{C.mean():.1f}**")

with tab2:
    loai = st.radio("Loại rối loạn thị giác màu",
                    ["Protanomaly", "Deuteranomaly", "Tritanomaly"], horizontal=True)
    muc = st.slider("Mức độ (0 = bình thường, 1 = nặng nhất)", 0.0, 1.0, 1.0, 0.1)
    M = colour.matrix_cvd_Machado2009(loai, muc)
    lin = colour.cctf_decoding(arr)
    out = np.einsum("ij,hwj->hwi", M, lin)
    out = np.clip(colour.cctf_encoding(np.clip(out, 0, 1)), 0, 1)
    c1, c2 = st.columns(2)
    c1.image(arr, caption="Ảnh gốc", clamp=True)
    c2.image(out, caption=f"Mô phỏng {loai} (mức {muc:.1f})", clamp=True)
    st.caption("Dùng mô hình Machado (2009). Ứng dụng: kiểm tra thiết kế bao bì có phân biệt được khi người xem "
               "bị rối loạn thị giác màu hay không.")
