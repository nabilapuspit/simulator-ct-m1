import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

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
# 2. NAVIGASI UTAMA (MODULAR NAVIGATION)
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
# HALAMAN 0: PANDUAN & TEORI DASAR (TANPA SIDEBAR)
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
        - Gunakan navigasi di atas untuk berpindah antar-modul.
        - Panel kontrol di sebelah kiri (*sidebar*) akan otomatis muncul saat Anda memasuki Modul Praktikum.
        - Amati perubahan respon grafik secara *real-time*.
        - Catat hasil pengamatan untuk mengisi Lembar Kerja Mahasiswa (LKM) pada myITS Classroom.
        """)

# ---------------------------------------------------------
# HALAMAN 1: MODUL 1 (AKUISISI DATA & SINOGRAM)
# ---------------------------------------------------------
elif menu_terpilih == "🔬 Modul 1: Akuisisi & Sinogram":
    st.subheader("Modul 1: Akuisisi Data Sinar-X & Pembentukan Sinogram")
    
    # Sidebar HANYA dipanggil & dirender di dalam Modul 1
    st.sidebar.header("⚙️ Parameter Modul 1")
    jumlah_sudut = st.sidebar.slider("Jumlah Proyeksi", min_value=10, max_value=360, value=180, step=10, key="m1_jml")
    sudut_maksimal = st.sidebar.slider("Rentang Sudut (°)", min_value=10, max_value=360, value=180, step=10, key="m1_rentang")
    #sudut_maksimal = st.sidebar.selectbox("Rentang Sudut (°)", options=[180, 360], index=0, key="m1_rentang")
    sudut_aktif = st.sidebar.slider("Sudut Aktif (°)", min_value=0, max_value=int(sudut_maksimal - 1), value=45, step=1, key="m1_aktif")

    # Fungsi Pemrosesan Modul 1
    @st.cache_data
    def get_phantom():
        img = shepp_logan_phantom()
        return rescale(img, scale=0.4, mode='reflect', channel_axis=None)

    image = get_phantom()
    center = image.shape[0] / 2

    theta = np.linspace(0.0, float(sudut_maksimal), int(jumlah_sudut), endpoint=False)
    sinogram = radon(image, theta=theta)

    idx_sudut = int((sudut_aktif / sudut_maksimal) * jumlah_sudut)
    idx_sudut = min(idx_sudut, jumlah_sudut - 1)
    sudut_sekarang = theta[idx_sudut]
    profil_1d = sinogram[:, idx_sudut]

    # Plotting 3 Panel
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.patch.set_facecolor('#0e1117')

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#161b22')

    # Panel 1
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

    # Panel 2
    ax2.set_title(f"2. Profil Proyeksi 1D ({sudut_sekarang:.1f}°)", color='#00e5ff', fontsize=11, fontweight='bold')
    ax2.plot(profil_1d, color='#00e5ff', linewidth=1.8)
    ax2.fill_between(range(len(profil_1d)), profil_1d, color='#00e5ff', alpha=0.2)
    ax2.set_xlabel("Posisi Detektor (t)", color='white', fontsize=9)
    ax2.set_ylabel("Jumlah Atenuasi", color='white', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.tick_params(colors='white', labelsize=8)

    # Panel 3
    ax3.set_title(f"3. Sinogram ({jumlah_sudut} Proyeksi, 0–{sudut_maksimal}°)", color='#00e5ff', fontsize=11, fontweight='bold')
    ax3.imshow(sinogram, cmap='bone', extent=(0, sudut_maksimal, 0, sinogram.shape[0]), aspect='auto', interpolation='nearest')
    ax3.axvline(x=sudut_sekarang, color='#ff1744', linewidth=1.8, linestyle='--', label='Posisi Pemindaian')
    ax3.set_xlabel(r"Sudut Proyeksi $\theta$ (°)", color='white', fontsize=9)
    ax3.set_ylabel("Posisi Detektor (t)", color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=8)
    ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------
# HALAMAN 2: MODUL 2 (REKONSTRUKSI 2D)
# ---------------------------------------------------------
elif menu_terpilih == "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)":
    st.subheader("Modul 2: Rekonstruksi Citra 2D (SBP vs FBP)")
    
    # Sidebar Khusus Modul 2 (Akan dipasang saat pembuatan Modul 2)
    st.sidebar.header("⚙️ Parameter Modul 2")
    st.sidebar.info("Pengaturan filter dan jumlah sudut rekonstruksi akan ada di sini.")
    
    st.info("🚧 Modul ini sedang dalam tahap pengembangan. Segera siap digunakan!")

# ---------------------------------------------------------
# HALAMAN 3: MODUL 3 (MANIPULASI & VISUALISASI)
# ---------------------------------------------------------
elif menu_terpilih == "🎨 Modul 3: Manipulasi & Visualisasi":
    st.subheader("Modul 3: Manipulasi & Visualisasi Citra CT-Scan")
    
    # Sidebar Khusus Modul 3 (Akan dipasang saat pembuatan Modul 3)
    st.sidebar.header("⚙️️ Parameter Modul 3")
    st.sidebar.info("Pengaturan Window Width & Level akan ada di sini.")
    
    st.info("🚧 Modul ini sedang dalam tahap pengembangan. Segera siap digunakan!")
