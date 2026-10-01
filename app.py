import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

# Konfigurasi Halaman Streamlit
st.set_page_config(layout="wide", page_title="Simulator CT-Scan - Modul 1")
plt.style.use('dark_background')

st.markdown("<h2 style='text-align: center; color: #00e5ff;'>SIMULATOR AKUISISI CT-SCAN & SINOGRAM</h2>", unsafe_allow_html=True)
st.write("---")

# Sidebar Kontrol Parameter
st.sidebar.header("⚙️ Parameter Akuisisi")

jumlah_sudut = st.sidebar.slider("Jumlah Proyeksi", min_value=10, max_value=360, value=180, step=10)
sudut_maksimal = st.sidebar.slider("Rentang Sudut (°)", min_value=10, max_value=360, value-180, step=10)
#sudut_maksimal = st.sidebar.selectbox("Rentang Sudut (°)", options=[180, 360], index=0)
sudut_aktif = st.sidebar.slider("Sudut Aktif (°)", min_value=0, max_value=int(sudut_maksimal - 1), value=45, step=1)

# 1. Penyiapan Phantom
@st.cache_data
def get_phantom():
    img = shepp_logan_phantom()
    return rescale(img, scale=0.4, mode='reflect', channel_axis=None)

image = get_phantom()
center = image.shape[0] / 2

# 2. Kalkulasi Sinogram Dinamis
theta = np.linspace(0.0, float(sudut_maksimal), int(jumlah_sudut), endpoint=False)
sinogram = radon(image, theta=theta)

# Indeks sudut aktif saat ini
idx_sudut = int((sudut_aktif / sudut_maksimal) * jumlah_sudut)
idx_sudut = min(idx_sudut, jumlah_sudut - 1)
sudut_sekarang = theta[idx_sudut]
profil_1d = sinogram[:, idx_sudut]

# 3. Plotting Visualisasi 3 Panel
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1, 1, 1]})
fig.patch.set_facecolor('#0e1117')

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#161b22')

# Panel 1: Objek Phantom + Berkas Radiasi
ax1.set_title(f"1. Pemindaian Sinar-X ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
ax1.imshow(image, cmap='bone')

rad = np.deg2rad(sudut_sekarang + 90)
length = center * 0.95
x_line = [center - length * np.cos(rad), center + length * np.cos(rad)]
y_line = [center - length * np.sin(rad), center + length * np.sin(rad)]

ax1.plot(x_line, y_line, color='#ff1744', linewidth=2, linestyle='--', label='Berkas Radiasi')
ax1.scatter([x_line[0]], [y_line[0]], color='#ffea00', s=70, zorder=5, label='Sumber Sinar-X')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_axis_off()

# Panel 2: Profil Proyeksi 1D
ax2.set_title(f"2. Profil Proyeksi 1D ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
ax2.plot(profil_1d, color='#00e5ff', linewidth=1.8)
ax2.fill_between(range(len(profil_1d)), profil_1d, color='#00e5ff', alpha=0.2)
ax2.set_xlabel("Posisi Detektor (t)", color='white', fontsize=9)
ax2.set_ylabel("Jumlah Atenuasi", color='white', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.tick_params(colors='white', labelsize=8)

# Panel 3: Sinogram 2D
ax3.set_title(f"3. Sinogram ({jumlah_sudut} Proyeksi, 0–{sudut_maksimal}°)", color='#00e5ff', fontsize=11, fontweight='bold')
ax3.imshow(sinogram, cmap='bone', extent=(0, sudut_maksimal, 0, sinogram.shape[0]), aspect='auto', interpolation='nearest')
ax3.axvline(x=sudut_sekarang, color='#ff1744', linewidth=1.8, linestyle='--', label='Posisi Pemindaian')
ax3.set_xlabel(r"Sudut Proyeksi $\theta$ (°)", color='white', fontsize=9)
ax3.set_ylabel("Posisi Detektor (t)", color='white', fontsize=9)
ax3.tick_params(colors='white', labelsize=8)
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
st.pyplot(fig)
