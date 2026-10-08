import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import time

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulator Praktikum Pencitraan Biomedik - CT-Scan",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('dark_background')

# Header Utama Aplikasi
st.markdown("<h1 style='text-align: center; color: #00e5ff; margin-bottom: 0px;'>Simulator Praktikum Pencitraan Biomedik</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #ffffff; margin-top: 0px;'>Modul Pembelajaran Interaktif CT-Scan</h3>", unsafe_allow_html=True)
st.write("---")

# ---------------------------------------------------------
# 2. NAVIGASI UTAMA
# ---------------------------------------------------------
menu_terpilih = st.radio(
    label="Pilih Halaman / Modul:",
    options=[
        "📖 Panduan & Teori Dasar", 
        "🔬 Modul 1: Akuisisi & Sinogram", 
        "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)", 
        "🎨 Modul 3: Manipulasi & Visualisasi"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.write("")

# ---------------------------------------------------------
# HALAMAN 0: PANDUAN & TEORI DASAR
# ---------------------------------------------------------
if menu_terpilih == "📖 Panduan & Teori Dasar":
    st.markdown("### 📚 Dasar Teori Computed Tomography (CT-Scan)")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.write("""
        **Computed Tomography (CT)** adalah teknik pencitraan medis yang memanfaatkan sinar-X untuk menghasilkan citra potongan melintang (tomografi) dari area tubuh tertentu.
        
        Pada laboratorium virtual ini, Anda akan mempelajari 3 tahapan utama dalam pemrosesan citra CT-Scan:
        1. **Akuisisi Data & Sinogram (Modul 1):** Proses pemindaian sinar-X dari berbagai sudut $(\\theta)$ dan penyusunan sinyal atenuasi 1D menjadi matriks 2D yang disebut **Sinogram** (Transformasi Radon).
        2. **Rekonstruksi Citra 2D (Modul 2):** Mengubah data sinogram kembali menjadi citra anatomi 2D menggunakan metode *Simple Backprojection* (SBP) dan *Filtered Backprojection* (FBP).
        3. **Manipulasi & Visualisasi (Modul 3):** Pengolahan pasca-rekonstruksi seperti pengaturan *Window Level/Width* (Hounsfield Unit) dan peningkatan kualitas citra.
        """)
    with col_t2:
        st.info("""
        💡 **Petunjuk Praktikum:**
        - Pilih Modul Praktikum melalui navigasi di atas.
        - Sesuaikan parameter fisik dan geometris pada *sidebar*.
        - Klik **Start Scan** untuk melihat proses pemindaian akumulatif secara real-time.
        """)

# ---------------------------------------------------------
# HALAMAN 1: MODUL 1 (EXACT GEOMETRY MATCHING WITH VIDEO)
# ---------------------------------------------------------
elif menu_terpilih == "🔬 Modul 1: Akuisisi & Sinogram":
    st.subheader("Modul 1: Akuisisi Data Sinar-X & Pembentukan Sinogram")
    
    # ------------------ SIDEBAR PARAMETER ------------------
    st.sidebar.header("⚙️ 1. Geometri Pemindaian")
    
    jenis_phantom = st.sidebar.selectbox(
        "Pilih Objek / Phantom", 
        options=["Titik Tunggal (Off-Center Dot)", "Shepp-Logan (Anatomi Otak)", "Dua Titik (Multi-Dot)", "Lingkaran Konsentris"]
    )
    
    jumlah_sudut = st.sidebar.slider("Jumlah Proyeksi (Sampling Sudut)", min_value=10, max_value=360, value=180, step=10)
    sudut_maksimal = st.sidebar.slider("Rentang Sudut Total (°)", min_value=10, max_value=360, value=180, step=10)
    
    st.sidebar.header("📻 2. Kondisi Fisika Sinar-X")
    tambah_noise = st.sidebar.checkbox("Simulasi Derau Foton (Poisson Noise)")
    level_noise = 0
    if tambah_noise:
        level_noise = st.sidebar.slider("Tingkat Intensitas Noise", min_value=1, max_value=10, value=3)
        
    st.sidebar.header("🎬 3. Kontrol Akuisisi")
    btn_start = st.sidebar.button("▶ Mulai Pemindaian (Start Scan)")
    sudut_aktif = st.sidebar.slider("Sudut Manual (°)", min_value=0, max_value=int(sudut_maksimal - 1), value=0, step=1)

    # ------------------ GENERASI PHANTOM ------------------
    @st.cache_data
    def generate_phantom(tipe):
        N = 160
        img = np.zeros((N, N))
        center = N // 2
        
        if tipe == "Titik Tunggal (Off-Center Dot)":
            y, x = np.ogrid[-center:N-center, -center:N-center]
            mask = (x - 30)**2 + (y - 20)**2 <= 5**2
            img[mask] = 1.0
        elif tipe == "Shepp-Logan (Anatomi Otak)":
            raw_img = shepp_logan_phantom()
            img = rescale(raw_img, scale=N/raw_img.shape[0], mode='reflect', channel_axis=None)
        elif tipe == "Dua Titik (Multi-Dot)":
            y, x = np.ogrid[-center:N-center, -center:N-center]
            mask1 = (x - 30)**2 + (y - 20)**2 <= 5**2
            mask2 = (x + 25)**2 + (y + 25)**2 <= 5**2
            img[mask1] = 1.0
            img[mask2] = 0.7
        elif tipe == "Lingkaran Konsentris":
            y, x = np.ogrid[-center:N-center, -center:N-center]
            mask1 = x**2 + y**2 <= 60**2
            mask2 = x**2 + y**2 <= 35**2
            mask3 = x**2 + y**2 <= 15**2
            img[mask1] = 0.3
            img[mask2] = 0.7
            img[mask3] = 1.0
            
        return img

    image = generate_phantom(jenis_phantom)
    center = image.shape[0] / 2.0

    # ------------------ PRE-CALCULATE FULL RADON ------------------
    theta = np.linspace(0.0, float(sudut_maksimal), int(jumlah_sudut), endpoint=False)
    sinogram_full = radon(image, theta=theta)
    
    if tambah_noise:
        noise = np.random.normal(0, level_noise * 0.1, sinogram_full.shape)
        sinogram_full = np.clip(sinogram_full + noise, 0, None)

    max_attenuation = np.max(sinogram_full) * 1.15 if np.max(sinogram_full) > 0 else 10.0

    # ------------------ FUNGSI RENDER PLOT STABIL ------------------
    def render_scan_frame(curr_idx, is_partial=False):
        sudut_sekarang = theta[curr_idx]
        profil_1d = sinogram_full[:, curr_idx]
        
        # Matrix sinogram akumulatif
        sino_display = np.zeros_like(sinogram_full)
        if is_partial:
            sino_display[:, :curr_idx+1] = sinogram_full[:, :curr_idx+1]
        else:
            sino_display = sinogram_full

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={'width_ratios': [1, 1, 1]})
        fig.patch.set_facecolor('#0e1117')
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#161b22')

        # ---------------- Panel 1: Sinar-X ----------------
        ax1.set_title(f"1. Pemindaian Sinar-X ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
        ax1.imshow(image, cmap='bone', origin='lower')
        
        rad = np.deg2rad(sudut_sekarang)
        dir_x = -np.sin(rad)
        dir_y = np.cos(rad)
        norm_x = np.cos(rad)
        norm_y = np.sin(rad)
        
        length = center * 1.3
        
        # Gambar garis berkas paralel tipis
        offsets = np.linspace(-center*0.8, center*0.8, 7)
        for off in offsets:
            x_c = center + off * norm_x
            y_c = center + off * norm_y
            ax1.plot([x_c - length * dir_x, x_c + length * dir_x],
                     [y_c - length * dir_y, y_c + length * dir_y],
                     color='#ff1744', linewidth=0.7, alpha=0.35, linestyle=':')

        # Garis Berkas Radiasi Aktif
        if np.max(profil_1d) > 0.05:
            idx_peak = np.argmax(profil_1d)
        else:
            idx_peak = int(center)
            
        t_offset = idx_peak - center
        x0 = center + t_offset * norm_x
        y0 = center + t_offset * norm_y
        
        x_main = [x0 - length * dir_x, x0 + length * dir_x]
        y_main = [y0 - length * dir_y, y0 + length * dir_y]
        
        ax1.plot(x_main, y_main, color='#ff1744', linewidth=2, linestyle='--', label='Berkas Radiasi Aktif')
        ax1.scatter([x_main[1]], [y_main[1]], color='#ffea00', s=70, zorder=5, label='Sumber Sinar-X')
        
        ax1.set_xlim(0, image.shape[1])
        ax1.set_ylim(0, image.shape[0])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_axis_off()

        # ---------------- Panel 2: Profil Proyeksi 1D ----------------
        ax2.set_title(f"2. Profil Proyeksi 1D ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
        ax2.plot(profil_1d, color='#00e5ff', linewidth=1.8)
        ax2.fill_between(range(len(profil_1d)), profil_1d, color='#00e5ff', alpha=0.2)
        
        if np.max(profil_1d) > 0.05:
            ax2.axvline(x=idx_peak, color='#ff1744', linestyle='--', linewidth=1.2, alpha=0.8, label='Posisi Peak')
        
        ax2.set_xlabel("
