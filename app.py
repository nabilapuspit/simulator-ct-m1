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
        - Amati profil proyeksi dan jejak sinusoid pada sinogram secara *real-time*.
        """)

# ---------------------------------------------------------
# HALAMAN 1: MODUL 1 (AKUISISI DATA & SINOGRAM ENHANCED)
# ---------------------------------------------------------
elif menu_terpilih == "🔬 Modul 1: Akuisisi & Sinogram":
    st.subheader("Modul 1: Akuisisi Data Sinar-X & Pembentukan Sinogram")
    
    # ------------------ SIDEBAR PARAMETER ------------------
    st.sidebar.header("⚙️ 1. Geometri Pemindaian")
    
    jenis_phantom = st.sidebar.selectbox(
        "Pilih Objek / Phantom", 
        options=["Shepp-Logan (Anatomi Otak)", "Titik Tunggal (Off-Center Dot)", "Dua Titik (Multi-Dot)", "Lingkaran Konsentris"]
    )
    
    jumlah_sudut = st.sidebar.slider("Jumlah Proyeksi (Sampling Sudut)", min_value=10, max_value=360, value=180, step=10)
    sudut_maksimal = st.sidebar.slider("Rentang Sudut Total (°)", min_value=10, max_value=360, value=180, step=10)
    #sudut_maksimal = st.sidebar.selectbox("Rentang Sudut Total (°)", options=[180, 360], index=0)
    
    st.sidebar.header("📻 2. Kondisi Fisika Sinar-X")
    tambah_noise = st.sidebar.checkbox("Simulasi Derau Foton (Poisson Noise)")
    level_noise = 0
    if tambah_noise:
        level_noise = st.sidebar.slider("Tingkat Intensitas Noise", min_value=1, max_value=10, value=3)
        
    st.sidebar.header("🎬 3. Kontrol Animasi")
    auto_play = st.sidebar.button("▶ Start Scanning (Animasi)")
    sudut_aktif = st.sidebar.slider("Sudut Aktif Manual (°)", min_value=0, max_value=int(sudut_maksimal - 1), value=45, step=1)

    # ------------------ GENERASI PHANTOM ------------------
    @st.cache_data
    def generate_phantom(tipe):
        N = 160
        img = np.zeros((N, N))
        center = N // 2
        
        if tipe == "Shepp-Logan (Anatomi Otak)":
            raw_img = shepp_logan_phantom()
            img = rescale(raw_img, scale=N/raw_img.shape[0], mode='reflect', channel_axis=None)
        elif tipe == "Titik Tunggal (Off-Center Dot)":
            # Membuat titik bundar kecil (radius 3 piksel) di offset (x=+25, y=-20)
            y, x = np.ogrid[-center:N-center, -center:N-center]
            mask = (x - 25)**2 + (y + 20)**2 <= 4**2
            img[mask] = 1.0
        elif tipe == "Dua Titik (Multi-Dot)":
            y, x = np.ogrid[-center:N-center, -center:N-center]
            mask1 = (x + 25)**2 + (y + 20)**2 <= 4**2
            mask2 = (x - 20)**2 + (y - 25)**2 <= 4**2
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
    center = image.shape[0] / 2

    # ------------------ KALKULASI RADON / SINOGRAM ------------------
    theta = np.linspace(0.0, float(sudut_maksimal), int(jumlah_sudut), endpoint=False)
    sinogram_clean = radon(image, theta=theta)
    
    # Penambahan Noise
    if tambah_noise:
        noise = np.random.normal(0, level_noise * 0.1, sinogram_clean.shape)
        sinogram = np.clip(sinogram_clean + noise, 0, None)
    else:
        sinogram = sinogram_clean

    # ------------------ HANDLING ANIMASI SCANNING ------------------
    placeholder = st.empty()
    
    def render_plots(curr_angle_idx):
        sudut_sekarang = theta[curr_angle_idx]
        profil_1d = sinogram[:, curr_angle_idx]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={'width_ratios': [1, 1, 1]})
        fig.patch.set_facecolor('#0e1117')
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#161b22')

        # Panel 1: Pemindaian Sinar-X (Garis Sumbu Proyeksi Utama theta)
        ax1.set_title(f"1. Pemindaian Sinar-X ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
        ax1.imshow(image, cmap='bone', origin='upper')
        
        # Konvensi Geometri Radon scikit-image di Matplotlib (origin='upper')
        # Sinar-X sejajar memindai tegak lurus terhadap sumbu proyeksi theta
        rad = np.deg2rad(sudut_sekarang)
        
        # Arah garis berkas sinar-X utama (Pusat Rotasi t=0)
        # Pada Matplotlib y-down, arah garis berkas adalah (-sin(rad), -cos(rad))
        dir_x = -np.sin(rad)
        dir_y = -np.cos(rad)
        
        length = center * 1.3
        x_line = [center - length * dir_x, center + length * dir_x]
        y_line = [center - length * dir_y, center + length * dir_y]
        
        # Plot Garis Sinar-X Utama yang Melintasi Pusat Rotasi
        ax1.plot(x_line, y_line, color='#ff1744', linewidth=2, linestyle='--', label='Berkas Utama (t=0)')
        ax1.scatter([x_line[0]], [y_line[0]], color='#ffea00', s=70, zorder=5, label='Sumber Sinar-X')
        ax1.plot(center, center, 'r+', markersize=10, markeredgewidth=2, label='Pusat Rotasi')
        
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

        # Panel 3: Sinogram
        ax3.set_title(f"3. Sinogram ({jumlah_sudut} Proyeksi, 0–{sudut_maksimal}°)", color='#00e5ff', fontsize=11, fontweight='bold')
        ax3.imshow(sinogram, cmap='bone', extent=(0, sudut_maksimal, 0, sinogram.shape[0]), aspect='auto', interpolation='nearest')
        ax3.axvline(x=sudut_sekarang, color='#ff1744', linewidth=1.8, linestyle='--', label='Posisi Pemindaian')
        ax3.set_xlabel(r"Sudut Proyeksi $\theta$ (°)", color='white', fontsize=9)
        ax3.set_ylabel("Posisi Detektor (t)", color='white', fontsize=9)
        ax3.tick_params(colors='white', labelsize=8)
        ax3.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig, profil_1d, sudut_sekarang

    if auto_play:
        for idx in range(0, jumlah_sudut, max(1, jumlah_sudut // 30)):
            fig, profil_1d, sudut_sekarang = render_plots(idx)
            with placeholder.container():
                st.pyplot(fig)
            time.sleep(0.05)
    else:
        idx_sudut = int((sudut_aktif / sudut_maksimal) * jumlah_sudut)
        idx_sudut = min(idx_sudut, jumlah_sudut - 1)
        fig, profil_1d, sudut_sekarang = render_plots(idx_sudut)
        with placeholder.container():
            st.pyplot(fig)

    # ------------------ ANALISIS KUANTITATIF & EXPORT ------------------
    st.write("---")
    st.markdown("### 📊 Analisis Data Proyeksi & Unduh Hasil")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    col_a1.metric(label="Sudut Aktif Saat Ini", value=f"{sudut_sekarang:.1f}°")
    col_a2.metric(label="Atenuasi Maksimum (Peak 1D)", value=f"{np.max(profil_1d):.2f}")
    col_a3.metric(label="Total Integral Atenuasi (Area)", value=f"{np.sum(profil_1d):.1f}")

    # Tombol Unduh Data CSV
    csv_data = sinogram.astype(str)
    np.savetxt("sinogram_data.csv", sinogram, delimiter=",")
    with open("sinogram_data.csv", "rb") as file:
        st.download_button(
            label="💾 Unduh Data Sinogram (CSV)",
            data=file,
            file_name=f"sinogram_{jenis_phantom.split()[0].lower()}.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# HALAMAN 2 & 3
# ---------------------------------------------------------
elif menu_terpilih == "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)":
    st.subheader("Modul 2: Rekonstruksi Citra 2D (SBP vs FBP)")
    st.info("🚧 Modul ini sedang dalam tahap pengembangan.")

elif menu_terpilih == "🎨 Modul 3: Manipulasi & Visualisasi":
    st.subheader("Modul 3: Manipulasi & Visualisasi Citra CT-Scan")
    st.info("🚧 Modul ini sedang dalam tahap pengembangan.")
