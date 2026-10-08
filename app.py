import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale, iradon
from scipy.ndimage import map_coordinates

import io
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Simulator Praktikum Pencitraan Biomedik - CT-Scan",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use("dark_background")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center; color:#5edcff;">
        Modul Pembelajaran Interaktif CT-Scan
    </h1>
    <p style="text-align:center; color:#cccccc; font-size:16px;">
        Simulasi Akuisisi Proyeksi, Atenuasi Sinar-X, dan Pembentukan Sinogram
    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MAIN MENU
# ============================================================

menu_terpilih = st.radio(
    "Pilih Modul Pembelajaran",
    [
        "Panduan & Teori Dasar",
        "Modul 1: Akuisisi & Sinogram",
        "Modul 2: Rekonstruksi 2D (SBP vs FBP)",
        "Modul 3: Manipulasi & Visualisasi"
    ],
    horizontal=True
)

# ============================================================
# PAGE 0 — PANDUAN & TEORI DASAR
# ============================================================

if menu_terpilih == "Panduan & Teori Dasar":
    st.markdown("## Panduan & Teori Dasar CT-Scan")
    st.markdown(
        """
        ### 1. Prinsip Dasar CT-Scan
        Pada CT-Scan, sinar-X melewati objek dari berbagai sudut. Ketika sinar-X melewati material, intensitas sinar mengalami atenuasi.

        Secara sederhana:
        $$
        I = I_0 e^{-\\int_L \\mu(x,y)\\,dl}
        $$

        dengan:
        - $I_0$ = intensitas sinar-X sebelum melewati objek
        - $I$ = intensitas sinar-X setelah melewati objek
        - $\\mu(x,y)$ = koefisien atenuasi linear
        - $L$ = lintasan sinar-X

        Setelah dilakukan transformasi logaritmik:
        $$
        p(t,\\theta) = -\\ln\\left(\\frac{I}{I_0}\\right) = \\int_L \\mu(x,y)\\,dl
        $$

        Nilai $p(t,\\theta)$ disebut sebagai **projection data**.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        ### 2. Beam dan Projection
        Pada satu sudut $\\theta$, detector menerima sinar-X pada berbagai posisi detector $t$.
        Setiap posisi $t$ merepresentasikan satu lintasan sinar-X.

        Jika lintasan melewati objek:
        $$
        p(t,\\theta) > 0
        $$

        Jika lintasan tidak melewati objek:
        $$
        p(t,\\theta) = 0
        $$

        Dengan demikian, projection 1D merupakan kumpulan nilai atenuasi dari seluruh lintasan sinar-X pada satu sudut.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        ### 3. Mengapa Titik Off-Center Menghasilkan Sinusoid?
        Misalkan sebuah titik pada phantom berada pada koordinat $(x_0,y_0)$. Posisi proyeksi titik tersebut pada detector mengikuti:

        $$
        \\boxed{
        t(\\theta) = x_0\\cos\\theta + y_0\\sin\\theta
        }
        $$

        Ketika $\\theta$ berubah, nilai $t$ berubah secara sinusoidal. Karena setiap projection pada setiap sudut dimasukkan ke dalam sinogram, lintasan titik tersebut akan terlihat sebagai pola sinusoidal.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        ### 4. Urutan Proses pada Simulator
        Simulator ini menggunakan urutan:

        **Phantom** $\\rightarrow$ **X-ray Beam** $\\rightarrow$ **Interaksi dengan Phantom** $\\rightarrow$ **Line Integral Attenuation** $\\rightarrow$ **1D Projection** $\\rightarrow$ **Sinogram**

        Dengan demikian, sinogram tidak hanya ditampilkan sebagai gambar akhir, tetapi dibentuk secara bertahap selama proses simulasi akuisisi.
        """
    )

# ============================================================
# PAGE 1 — MODULE 1 — AKUISISI & SINOGRAM
# ============================================================

elif menu_terpilih == "Modul 1: Akuisisi & Sinogram":

    # CONSTANTS
    N = 160
    IMAGE_CENTER = N / 2.0
    OFFCENTER_X = 25.0
    OFFCENTER_Y = 20.0
    DOT_RADIUS = 5.0

    # SIDEBAR
    st.sidebar.header("1. Geometri Pemindaian")
    jenis_phantom = st.sidebar.selectbox(
        "Pilih Objek / Phantom",
        [
            "Titik Tunggal (Off-Center Dot)",
            "Shepp-Logan (Anatomi Otak)",
            "Dua Titik (Multi-Dot)",
            "Lingkaran Konsentris"
        ]
    )
    jumlah_sudut = st.sidebar.slider("Jumlah Proyeksi (Sampling Sudut)", min_value=10, max_value=360, value=180, step=10)
    sudut_maksimal = st.sidebar.slider("Rentang Sudut Total (°)", min_value=10, max_value=360, value=180, step=10)

    st.sidebar.header("2. Kontrol Beam")
    mode_beam = st.sidebar.radio(
        "Mode Posisi Beam",
        [
            "Otomatis: Beam Mengikuti Titik",
            "Manual: Geser Beam"
        ]
    )

    if mode_beam == "Manual: Geser Beam":
        detector_t_manual = st.sidebar.slider("Posisi Detector t", min_value=-80.0, max_value=80.0, value=0.0, step=0.5)
    else:
        detector_t_manual = None

    st.sidebar.header("3. Kondisi Fisika Sinar-X")
    tambah_noise = st.sidebar.checkbox("Simulasi Derau Pengukuran")
    level_noise = 0
    if tambah_noise:
        level_noise = st.sidebar.slider("Tingkat Noise", min_value=1, max_value=10, value=3, step=1)

    st.sidebar.header("4. Kontrol Akuisisi")
    btn_start = st.sidebar.button("Mulai Pemindaian")
    sudut_aktif = st.sidebar.slider(
        "Sudut Beam θ (°)",
        min_value=0.0,
        max_value=float(max(1, sudut_maksimal - 1)),
        value=0.0,
        step=1.0
    )

    # PHANTOM GENERATION
    Y, X = np.indices((N, N))
    x_grid = X - IMAGE_CENTER + 0.5
    y_grid = Y - IMAGE_CENTER + 0.5

    def generate_phantom(phantom_type):
        image = np.zeros((N, N), dtype=float)
        if phantom_type == "Titik Tunggal (Off-Center Dot)":
            mask = (x_grid - OFFCENTER_X)**2 + (y_grid - OFFCENTER_Y)**2 <= DOT_RADIUS**2
            image[mask] = 1.0
        elif phantom_type == "Shepp-Logan (Anatomi Otak)":
            raw = shepp_logan_phantom()
            image = rescale(raw, scale=N / raw.shape[0], mode="reflect", channel_axis=None)
            image = image / np.max(image)
        elif phantom_type == "Dua Titik (Multi-Dot)":
            mask1 = (x_grid - 25)**2 + (y_grid - 20)**2 <= DOT_RADIUS**2
            mask2 = (x_grid + 20)**2 + (y_grid + 25)**2 <= DOT_RADIUS**2
            image[mask1] = 1.0
            image[mask2] = 0.7
        elif phantom_type == "Lingkaran Konsentris":
            mask1 = x_grid**2 + y_grid**2 <= 60**2
            mask2 = x_grid**2 + y_grid**2 <= 35**2
            mask3 = x_grid**2 + y_grid**2 <= 15**2
            image[mask1] = 0.3
            image[mask2] = 0.7
            image[mask3] = 1.0
        return image

    image = generate_phantom(jenis_phantom)
    detector_t = np.arange(N) - N / 2 + 0.5
    theta = np.linspace(0.0, float(sudut_maksimal), int(jumlah_sudut), endpoint=False)

    def point_projection_t(x0, y0, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        return x0 * np.cos(angle_rad) + y0 * np.sin(angle_rad)

    def beam_geometry(angle_deg, t_value):
        angle_rad = np.deg2rad(angle_deg)
        normal_x = np.cos(angle_rad)
        normal_y = np.sin(angle_rad)
        direction_x = -np.sin(angle_rad)
        direction_y = np.cos(angle_rad)
        beam_center_x = t_value * normal_x
        beam_center_y = t_value * normal_y
        return beam_center_x, beam_center_y, direction_x, direction_y

    @st.cache_data(show_spinner=False)
    def calculate_forward_projections(image, theta_values, detector_values):
        image_n = image.shape[0]
        center = image_n / 2.0
        n_detector = len(detector_values)
        n_angles = len(theta_values)
        sinogram = np.zeros((n_detector, n_angles), dtype=float)

        s_values = np.linspace(-image_n * 0.9, image_n * 0.9, 2 * image_n + 1)
        ds = s_values[1] - s_values[0]

        for j, angle_deg in enumerate(theta_values):
            angle_rad = np.deg2rad(angle_deg)
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)

            for i, t_value in enumerate(detector_values):
                x_path = t_value * cos_theta - s_values * sin_theta
                y_path = t_value * sin_theta + s_values * cos_theta
                col = x_path + center - 0.5
                row = y_path + center - 0.5

                valid = (col >= 0) & (col <= image_n - 1) & (row >= 0) & (row <= image_n - 1)
                if not np.any(valid):
                    continue

                coords = np.vstack([row[valid], col[valid]])
                attenuation_values = map_coordinates(image, coords, order=1, mode="constant", cval=0.0)
                sinogram[i, j] = np.sum(attenuation_values) * ds

        return sinogram

    with st.spinner("Menghitung data projection..."):
        sinogram_clean = calculate_forward_projections(image, theta, detector_t)

    if tambah_noise:
        rng = np.random.default_rng(42)
        noise_sigma = level_noise * 0.02 * max(float(np.max(sinogram_clean)), 1e-8)
        sinogram_display = sinogram_clean + rng.normal(0, noise_sigma, sinogram_clean.shape)
        sinogram_display = np.clip(sinogram_display, 0, None)
    else:
        sinogram_display = sinogram_clean.copy()

    # ========================================================
    # SIMPAN DATA MODUL 1 UNTUK DIGUNAKAN DI MODUL 2
    # ========================================================

    st.session_state["ct_image"] = image.copy()
    st.session_state["ct_sinogram"] = sinogram_display.copy()
    st.session_state["ct_theta"] = theta.copy()
    st.session_state["ct_detector_t"] = detector_t.copy()

    current_angle = float(sudut_aktif)

    if mode_beam == "Otomatis: Beam Mengikuti Titik":
        if jenis_phantom == "Titik Tunggal (Off-Center Dot)":
            active_t = point_projection_t(OFFCENTER_X, OFFCENTER_Y, current_angle)
        else:
            active_t = 0.0
    else:
        active_t = float(detector_t_manual)

    active_t = float(np.clip(active_t, detector_t[0], detector_t[-1]))
    active_detector_idx = int(np.argmin(np.abs(detector_t - active_t)))
    active_angle_idx = int(np.argmin(np.abs(theta - current_angle)))

    projection_profile = sinogram_display[:, active_angle_idx]
    active_attenuation = float(projection_profile[active_detector_idx])

    max_projection = max(float(np.max(sinogram_display)), 1e-8)
    hit_threshold = max_projection * 0.01
    beam_hits_object = active_attenuation > hit_threshold

    beam_center_x, beam_center_y, direction_x, direction_y = beam_geometry(current_angle, active_t)
    beam_length = N * 0.95
    beam_x1 = beam_center_x - beam_length * direction_x
    beam_y1 = beam_center_y - beam_length * direction_y
    beam_x2 = beam_center_x + beam_length * direction_x
    beam_y2 = beam_center_y + beam_length * direction_y

    if beam_hits_object:
        st.error("BEAM MENGENAI OBJEK")
    else:
        st.info("BEAM TIDAK MENGENAI OBJEK")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sudut θ", f"{current_angle:.1f}°")
    c2.metric("Detector t", f"{active_t:.2f}")
    c3.metric("Attenuation", f"{active_attenuation:.3f}")
    c4.metric("Beam", "HIT" if beam_hits_object else "MISS")

    # THREE PANELS
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#161b22")

    # PANEL 1
    ax1.set_title(f"1. Pemindaian Sinar-X ({current_angle:.1f}°)", color="#5edcff", fontsize=12, fontweight="bold")
    ax1.imshow(image, cmap="bone", origin="lower", extent=[-N / 2, N / 2, -N / 2, N / 2])
    beam_color = "#ff1744" if beam_hits_object else "#00e5ff"
    ax1.plot([beam_x1, beam_x2], [beam_y1, beam_y2], color=beam_color, linewidth=2.5, linestyle="--")
    ax1.scatter([beam_center_x], [beam_center_y], color="#ffea00", s=45, zorder=6)
    ax1.scatter([0], [0], color="#00ff88", s=25, zorder=6)

    if jenis_phantom == "Titik Tunggal (Off-Center Dot)":
        theoretical_t = point_projection_t(OFFCENTER_X, OFFCENTER_Y, current_angle)
        ax1.scatter([OFFCENTER_X], [OFFCENTER_Y], facecolors="none", edgecolors="#ffea00", s=130, linewidths=1.5, zorder=7)
        tx = theoretical_t * np.cos(np.deg2rad(current_angle))
        ty = theoretical_t * np.sin(np.deg2rad(current_angle))
        ax1.scatter([tx], [ty], color="#ffea00", s=20, zorder=7)

    ax1.set_xlim(-N / 2, N / 2)
    ax1.set_ylim(-N / 2, N / 2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.12)

    # PANEL 2
    ax2.set_title(f"2. Profil Proyeksi 1D ({current_angle:.1f}°)", color="#5edcff", fontsize=12, fontweight="bold")
    ax2.plot(detector_t, projection_profile, color="#5edcff", linewidth=2)
    ax2.fill_between(detector_t, projection_profile, color="#5edcff", alpha=0.18)
    ax2.axvline(x=active_t, color=beam_color, linestyle="--", linewidth=2.5, label=f"Beam t={active_t:.2f}")
    ax2.scatter([active_t], [active_attenuation], color="#ffea00", s=75, zorder=8, edgecolors="black", linewidths=0.8)

    if mode_beam == "Otomatis: Beam Mengikuti Titik" and jenis_phantom == "Titik Tunggal (Off-Center Dot)":
        theoretical_t = point_projection_t(OFFCENTER_X, OFFCENTER_Y, current_angle)
        ax2.axvline(x=theoretical_t, color="#ffea00", linestyle=":", linewidth=1.2, alpha=0.8)

    ax2.axhline(0, color="white", alpha=0.25, linewidth=1)
    ax2.set_xlim(detector_t[0], detector_t[-1])
    ax2.set_ylim(0, max(float(np.max(projection_profile)) * 1.15, 1.0))
    ax2.set_xlabel("Posisi Detector t")
    ax2.set_ylabel("Line Integral Attenuation")
    ax2.grid(True, linestyle=":", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)

    # PANEL 3
    ax3.set_title(f"3. Sinogram ({sudut_maksimal:.0f}°)", color="#5edcff", fontsize=12, fontweight="bold")
    sinogram_max = max(float(np.max(sinogram_display)), 1e-8)
    ax3.imshow(
        sinogram_display,
        cmap="bone",
        extent=[theta[0], theta[-1] if len(theta) > 1 else sudut_maksimal, detector_t[0], detector_t[-1]],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=sinogram_max
    )
    ax3.axvline(current_angle, color="#ff1744", linestyle="--", linewidth=1.8)
    ax3.scatter([current_angle], [active_t], color="#ffea00", s=70, zorder=8, edgecolors="black", linewidths=0.8)
    ax3.set_xlim(theta[0], theta[-1] if len(theta) > 1 else sudut_maksimal)
    ax3.set_ylim(detector_t[0], detector_t[-1])
    ax3.set_xlabel(r"Sudut Proyeksi $\theta$ (°)")
    ax3.set_ylabel("Posisi Detector t")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # EDUCATIONAL EXPLANATION
    st.markdown("---")
    if beam_hits_object:
        st.success(
            f"""
            ### Beam mengenai phantom

            Pada:
            - Sudut: **θ = {current_angle:.1f}°**
            - Posisi detector: **t = {active_t:.2f}**
            - Attenuation: **{active_attenuation:.3f}**

            Beam melewati material phantom sehingga terjadi line integral attenuation.

            $$
            p(t,\\theta) = \\int_L \\mu(x,y)\\,dl = {active_attenuation:.3f}
            $$

            Karena attenuation pada posisi beam bernilai tinggi, marker kuning pada **Profil Proyeksi 1D** berada pada bagian peak.
            Titik yang sama juga direpresentasikan oleh marker pada **sinogram**.
            """
        )
    else:
        st.info(
            f"""
            ### Beam tidak mengenai phantom

            Pada:
            - Sudut: **θ = {current_angle:.1f}°**
            - Posisi detector: **t = {active_t:.2f}**
            - Attenuation: **{active_attenuation:.3f}**

            Lintasan beam tidak melewati phantom. Oleh karena itu:

            $$
            p(t,\\theta) \\approx 0
            $$

            Perhatikan bahwa **projection profile secara keseluruhan tidak harus kosong**. Peak dapat tetap muncul pada detector position lain yang dilewati phantom.
            Yang bernilai nol adalah **measurement pada posisi beam yang sedang dipilih**.
            """
        )

    if jenis_phantom == "Titik Tunggal (Off-Center Dot)":
        theoretical_t = point_projection_t(OFFCENTER_X, OFFCENTER_Y, current_angle)
        st.markdown(
            f"""
            ### Posisi Geometrik Titik

            Titik phantom berada pada:
            $$
            (x_0,y_0) = ({OFFCENTER_X:.0f},{OFFCENTER_Y:.0f})
            $$

            Posisi proyeksinya secara geometrik adalah:
            $$
            t(\\theta) = x_0\\cos\\theta + y_0\\sin\\theta
            $$

            sehingga pada θ = **{current_angle:.1f}°**:
            $$
            t(\\theta) = {theoretical_t:.2f}
            $$

            Nilai inilah yang digunakan oleh mode **"Otomatis: Beam Mengikuti Titik"**.
            """
        )

    # ANIMATION
    if btn_start:
        st.markdown("---")
        st.subheader("Simulasi Pembentukan Sinogram")
        st.caption("Beam berputar dari sudut awal hingga sudut akhir. Projection dan sinogram diperbarui pada setiap sudut.")

        animation_placeholder = st.empty()
        status_placeholder = st.empty()

        for curr_idx in range(len(theta)):
            angle_now = float(theta[curr_idx])

            if mode_beam == "Otomatis: Beam Mengikuti Titik":
                if jenis_phantom == "Titik Tunggal (Off-Center Dot)":
                    t_now = point_projection_t(OFFCENTER_X, OFFCENTER_Y, angle_now)
                else:
                    t_now = 0.0
            else:
                t_now = float(detector_t_manual)

            t_now = float(np.clip(t_now, detector_t[0], detector_t[-1]))
            detector_idx_now = int(np.argmin(np.abs(detector_t - t_now)))
            projection_now = sinogram_display[:, curr_idx]
            attenuation_now = float(projection_now[detector_idx_now])

            hit_now = attenuation_now > hit_threshold
            color_now = "#ff1744" if hit_now else "#00e5ff"

            bx, by, dx, dy = beam_geometry(angle_now, t_now)
            L = N * 0.95
            xa, ya = bx - L * dx, by - L * dy
            xb, yb = bx + L * dx, by + L * dy

            accumulated_sinogram = np.zeros_like(sinogram_display)
            accumulated_sinogram[:, :curr_idx + 1] = sinogram_display[:, :curr_idx + 1]

            fig_anim, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.5))
            fig_anim.patch.set_facecolor("#0e1117")
            for ax in [a1, a2, a3]:
                ax.set_facecolor("#161b22")

            a1.set_title(f"1. Pemindaian Sinar-X ({angle_now:.1f}°)", color="#5edcff", fontsize=12, fontweight="bold")
            a1.imshow(image, cmap="bone", origin="lower", extent=[-N / 2, N / 2, -N / 2, N / 2])
            a1.plot([xa, xb], [ya, yb], color=color_now, linewidth=2.5, linestyle="--")
            a1.scatter([bx], [by], color="#ffea00", s=45, zorder=6)

            if jenis_phantom == "Titik Tunggal (Off-Center Dot)":
                a1.scatter([OFFCENTER_X], [OFFCENTER_Y], facecolors="none", edgecolors="#ffea00", s=130, linewidths=1.5, zorder=7)

            a1.set_xlim(-N / 2, N / 2)
            a1.set_ylim(-N / 2, N / 2)
            a1.set_axis_off()

            a2.set_title(f"2. Profil Proyeksi 1D ({angle_now:.1f}°)", color="#5edcff", fontsize=12, fontweight="bold")
            a2.plot(detector_t, projection_now, color="#5edcff", linewidth=2)
            a2.fill_between(detector_t, projection_now, color="#5edcff", alpha=0.18)
            a2.axvline(t_now, color=color_now, linestyle="--", linewidth=2.5)
            a2.scatter([t_now], [attenuation_now], color="#ffea00", s=75, zorder=8)
            a2.set_xlim(detector_t[0], detector_t[-1])
            a2.set_ylim(0, max(float(np.max(projection_now)) * 1.15, 1.0))
            a2.set_xlabel("Detector position t")
            a2.set_ylabel("Attenuation")
            a2.grid(True, linestyle=":", alpha=0.25)

            a3.set_title("3. Sinogram — Dibentuk Real-Time", color="#5edcff", fontsize=12, fontweight="bold")
            a3.imshow(
                accumulated_sinogram,
                cmap="bone",
                extent=[theta[0], theta[-1] if len(theta) > 1 else sudut_maksimal, detector_t[0], detector_t[-1]],
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=0,
                vmax=sinogram_max
            )
            a3.axvline(angle_now, color="#ff1744", linestyle="--", linewidth=1.8)
            a3.scatter([angle_now], [t_now], color="#ffea00", s=70, zorder=8)
            a3.set_xlim(theta[0], theta[-1] if len(theta) > 1 else sudut_maksimal)
            a3.set_ylim(detector_t[0], detector_t[-1])
            a3.set_xlabel("Projection angle θ (°)")
            a3.set_ylabel("Detector position t")

            plt.tight_layout()
            animation_placeholder.pyplot(fig_anim, use_container_width=True)
            plt.close(fig_anim)

            if hit_now:
                status_placeholder.error(f"BEAM HIT | θ = {angle_now:.1f}° | t = {t_now:.2f} | Attenuation = {attenuation_now:.3f}")
            else:
                status_placeholder.info(f"BEAM MISS | θ = {angle_now:.1f}° | t = {t_now:.2f} | Attenuation = {attenuation_now:.3f}")

            time.sleep(0.04)

        status_placeholder.success("Akuisisi selesai. Sinogram telah terbentuk.")

    # DOWNLOAD DATA
    st.markdown("---")
    st.subheader("Data Projection")
    csv_buffer = io.StringIO()
    header = "detector_t," + ",".join([f"theta_{angle:.2f}" for angle in theta])
    np.savetxt(csv_buffer, sinogram_display, delimiter=",", header=header, comments="")

    st.download_button(
        label="Download Projection / Sinogram CSV",
        data=csv_buffer.getvalue(),
        file_name="ct_projection_sinogram.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 2 — MODULE 2 — REKONSTRUKSI
# ============================================================

elif menu_terpilih == "Modul 2: Rekonstruksi 2D (SBP vs FBP)":

    st.title("Modul 2: Rekonstruksi 2D — SBP vs FBP")

    st.markdown(
        """
        ### Tujuan Modul

        Pada modul ini, mahasiswa akan mempelajari bagaimana citra CT
        direkonstruksi dari data proyeksi yang telah diperoleh pada Modul 1.

        Dua metode dibandingkan:

        **1. Simple Back Projection (SBP)**  
        Setiap profil proyeksi diproyeksikan kembali ke ruang citra sesuai
        dengan arah akuisisinya.

        **2. Filtered Back Projection (FBP)**  
        Data proyeksi terlebih dahulu diberi filter, kemudian dilakukan
        back projection. Filtering digunakan untuk mengurangi efek blur
        yang muncul pada SBP.
        """
    )

    # ========================================================
    # CEK DATA DARI MODUL 1
    # ========================================================

    if "ct_sinogram" not in st.session_state:

        st.warning(
            """
            Data sinogram belum tersedia.

            Silakan masuk ke **Modul 1: Akuisisi & Sinogram** terlebih dahulu,
            kemudian jalankan proses akuisisi untuk menghasilkan sinogram.
            """
        )

        st.stop()

    # Ambil data dari Modul 1
    image_original = st.session_state["ct_image"]
    sinogram = st.session_state["ct_sinogram"]
    theta = st.session_state["ct_theta"]
    detector_t = st.session_state["ct_detector_t"]

    # ========================================================
    # INFORMASI DATA
    # ========================================================

    st.subheader("1. Input dari Modul 1")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ukuran Sinogram",
            f"{sinogram.shape[0]} × {sinogram.shape[1]}"
        )

    with col2:
        st.metric(
            "Jumlah Proyeksi",
            f"{len(theta)}"
        )

    with col3:
        st.metric(
            "Rentang Sudut",
            f"{theta[0]:.1f}° – {theta[-1]:.1f}°"
        )

    # ========================================================
    # TAMPILKAN SINOGRAM
    # ========================================================

    fig_input, ax_input = plt.subplots(figsize=(8, 4.5))

    fig_input.patch.set_facecolor("#0e1117")
    ax_input.set_facecolor("#161b22")

    ax_input.imshow(
        sinogram,
        cmap="bone",
        aspect="auto",
        extent=[
            theta[0],
            theta[-1],
            detector_t[-1],
            detector_t[0]
        ],
        origin="upper"
    )

    ax_input.set_xlabel("Projection angle θ (°)")
    ax_input.set_ylabel("Detector position t")
    ax_input.set_title(
        "Sinogram yang Digunakan untuk Rekonstruksi",
        color="#5edcff",
        fontweight="bold"
    )

    plt.tight_layout()
    st.pyplot(fig_input, use_container_width=True)
    plt.close(fig_input)

    st.markdown(
        """
        **Interpretasi:**

        Setiap kolom sinogram merepresentasikan satu profil proyeksi pada
        sudut tertentu. Rekonstruksi bertujuan mengembalikan informasi
        tersebut menjadi representasi spasial 2D.
        """
    )

    st.divider()

    # ========================================================
    # EKSPERIMEN JUMLAH PROYEKSI
    # ========================================================

    st.subheader("2. Eksperimen Jumlah Proyeksi")

    st.markdown(
        """
        Jumlah proyeksi menentukan seberapa banyak informasi angular yang
        digunakan dalam proses rekonstruksi.
        """
    )

    jumlah_proyeksi = st.slider(
        "Jumlah proyeksi yang digunakan",
        min_value=2,
        max_value=len(theta),
        value=len(theta),
        step=1
    )

    # Memilih proyeksi yang tersebar merata sepanjang rentang sudut
    indeks_proyeksi = np.linspace(
        0,
        len(theta) - 1,
        jumlah_proyeksi
    ).astype(int)

    # Hilangkan kemungkinan indeks duplikat
    indeks_proyeksi = np.unique(indeks_proyeksi)

    sinogram_used = sinogram[:, indeks_proyeksi]
    theta_used = theta[indeks_proyeksi]

    col1, col2 = st.columns(2)

    with col1:

        fig_sub, ax_sub = plt.subplots(figsize=(7, 4))

        fig_sub.patch.set_facecolor("#0e1117")
        ax_sub.set_facecolor("#161b22")

        ax_sub.imshow(
            sinogram_used,
            cmap="bone",
            aspect="auto",
            extent=[
                theta_used[0],
                theta_used[-1],
                detector_t[-1],
                detector_t[0]
            ],
            origin="upper"
        )

        ax_sub.set_xlabel("Projection angle θ (°)")
        ax_sub.set_ylabel("Detector position t")
        ax_sub.set_title(
            f"Sinogram Digunakan ({len(theta_used)} proyeksi)",
            color="#5edcff",
            fontweight="bold"
        )

        plt.tight_layout()
        st.pyplot(fig_sub, use_container_width=True)
        plt.close(fig_sub)

    with col2:

        st.info(
            f"""
            **Data rekonstruksi**

            Jumlah proyeksi = **{len(theta_used)}**

            Rentang sudut = **{theta_used[0]:.1f}° – {theta_used[-1]:.1f}°**

            Semakin sedikit proyeksi yang digunakan, semakin jarang sampling
            angular yang tersedia untuk rekonstruksi.
            """
        )

    st.divider()

    # ========================================================
    # FUNGSI SIMPLE BACK PROJECTION
    # ========================================================

    def simple_back_projection(
        sinogram_data,
        theta_values,
        detector_values,
        image_size
    ):
        """
        Simple Back Projection eksplisit.

        Untuk setiap pixel (x,y):

            t = x cos(theta) + y sin(theta)

        kemudian mengambil nilai proyeksi pada posisi detector t
        dan menjumlahkannya untuk seluruh sudut.
        """

        reconstructed = np.zeros(
            (image_size, image_size),
            dtype=np.float64
        )

        # Koordinat pixel relatif terhadap pusat citra
        coords = (
            np.arange(image_size)
            - image_size / 2
            + 0.5
        )

        X, Y = np.meshgrid(
            coords,
            coords
        )

        # Jarak antar detector
        detector_spacing = float(
            np.mean(np.diff(detector_values))
        )

        for j, angle_deg in enumerate(theta_values):

            angle_rad = np.deg2rad(angle_deg)

            # Parallel-beam geometry
            T = (
                X * np.cos(angle_rad)
                + Y * np.sin(angle_rad)
            )

            # Konversi posisi detector menjadi index
            detector_index = (
                T - detector_values[0]
            ) / detector_spacing

            projection = sinogram_data[:, j]

            # Interpolasi projection pada posisi detector T
            backprojected = map_coordinates(
                projection,
                detector_index,
                order=1,
                mode="constant",
                cval=0.0
            )

            reconstructed += backprojected

        # Aproksimasi integral terhadap theta
        if len(theta_values) > 1:

            delta_theta = np.mean(
                np.diff(np.deg2rad(theta_values))
            )

            reconstructed *= delta_theta

        return reconstructed

    # ========================================================
    # FUNGSI NORMALISASI VISUAL
    # ========================================================

    def normalize_image(img):

        img = np.asarray(
            img,
            dtype=float
        )

        min_val = np.min(img)
        max_val = np.max(img)

        if max_val - min_val < 1e-12:
            return np.zeros_like(img)

        return (
            img - min_val
        ) / (
            max_val - min_val
        )

    # ========================================================
    # TOMBOL REKONSTRUKSI
    # ========================================================

    st.subheader("3. Rekonstruksi")

    st.markdown(
        """
        Klik tombol berikut untuk melakukan rekonstruksi menggunakan
        **Simple Back Projection (SBP)** dan **Filtered Back Projection (FBP)**.
        """
    )

    jalankan_rekonstruksi = st.button(
        "▶ Jalankan Rekonstruksi SBP dan FBP",
        type="primary",
        use_container_width=True
    )

    if jalankan_rekonstruksi:

        with st.spinner(
            "Menghitung Simple Back Projection..."
        ):

            reconstruction_sbp = simple_back_projection(
                sinogram_used,
                theta_used,
                detector_t,
                image_original.shape[0]
            )

        with st.spinner(
            "Menghitung Filtered Back Projection..."
        ):

            # iradon menerima:
            # baris    = posisi detector
            # kolom    = projection angle
            #
            # Filter yang digunakan:
            # Ram-Lak / ramp filter

            reconstruction_fbp = iradon(
                sinogram_used,
                theta=theta_used,
                filter_name="ramp",
                circle=False,
                output_size=image_original.shape[0]
            )

        # Simpan hasil
        st.session_state["reconstruction_sbp"] = (
            reconstruction_sbp
        )

        st.session_state["reconstruction_fbp"] = (
            reconstruction_fbp
        )

        st.session_state["reconstruction_theta"] = (
            theta_used.copy()
        )

        st.session_state["reconstruction_sinogram"] = (
            sinogram_used.copy()
        )

        st.session_state["reconstruction_num_projections"] = (
            len(theta_used)
        )

        st.success(
            "Rekonstruksi SBP dan FBP selesai."
        )

    # ========================================================
    # TAMPILKAN HASIL REKONSTRUKSI
    # ========================================================

    if "reconstruction_sbp" in st.session_state:

        reconstruction_sbp = (
            st.session_state["reconstruction_sbp"]
        )

        reconstruction_fbp = (
            st.session_state["reconstruction_fbp"]
        )

        # Normalisasi hanya untuk visualisasi
        sbp_display = normalize_image(
            reconstruction_sbp
        )

        fbp_display = normalize_image(
            reconstruction_fbp
        )

        original_display = normalize_image(
            image_original
        )

        st.divider()

        st.subheader(
            "4. Hasil Rekonstruksi: SBP vs FBP"
        )

        fig_compare, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        fig_compare.patch.set_facecolor("#0e1117")

        axes[0].imshow(
            original_display,
            cmap="bone",
            vmin=0,
            vmax=1
        )

        axes[0].set_title(
            "Original Phantom",
            color="#5edcff",
            fontweight="bold"
        )

        axes[1].imshow(
            sbp_display,
            cmap="bone",
            vmin=0,
            vmax=1
        )

        axes[1].set_title(
            "Simple Back Projection",
            color="#5edcff",
            fontweight="bold"
        )

        axes[2].imshow(
            fbp_display,
            cmap="bone",
            vmin=0,
            vmax=1
        )

        axes[2].set_title(
            "Filtered Back Projection",
            color="#5edcff",
            fontweight="bold"
        )

        for ax in axes:
            ax.axis("off")
            ax.set_facecolor("#161b22")

        plt.tight_layout()
        st.pyplot(
            fig_compare,
            use_container_width=True
        )
        plt.close(fig_compare)

        # ====================================================
        # PENJELASAN SBP
        # ====================================================

        st.markdown(
            """
            ### Apa yang terjadi pada SBP?

            Pada **Simple Back Projection**, setiap nilai pada projection
            profile "dikembalikan" sepanjang garis yang sesuai dengan posisi
            detector dan sudut akuisisinya.

            Secara sederhana:

            $$
            f_{BP}(x,y)
            \\approx
            \\sum_{\\theta}
            p(t,\\theta)
            $$

            dengan:

            $$
            t = x\\cos\\theta + y\\sin\\theta
            $$

            Karena setiap measurement disebarkan kembali sepanjang garis,
            informasi dari banyak sudut akan saling tumpang tindih.

            Akibatnya, hasil SBP cenderung mengalami **blur**.
            """
        )

        # ====================================================
        # PENJELASAN FBP
        # ====================================================

        st.markdown(
            """
            ### Apa yang terjadi pada FBP?

            Pada **Filtered Back Projection**, projection profile tidak
            langsung di-backproject.

            Data proyeksi terlebih dahulu melewati proses filtering:

            $$
            p_f(t,\\theta)
            =
            p(t,\\theta) * h(t)
            $$

            kemudian dilakukan backprojection:

            $$
            f_{FBP}(x,y)
            =
            \\int
            p_f(t,\\theta)
            \\,d\\theta
            $$

            Filter yang digunakan pada simulasi ini adalah
            **Ram-Lak (ramp filter)**.

            Filtering membantu mengompensasi efek blur dari proses
            backprojection sehingga struktur dan batas objek menjadi lebih
            tajam.
            """
        )

        st.divider()

        # ====================================================
        # PERBANDINGAN KUALITATIF
        # ====================================================

        st.subheader(
            "5. Interpretasi SBP dan FBP"
        )

        col_sbp, col_fbp = st.columns(2)

        with col_sbp:

            st.markdown(
                """
                #### Simple Back Projection

                **Kelebihan**
                - Konsep sangat sederhana.
                - Mudah menunjukkan hubungan projection → image.
                - Cocok untuk memahami prinsip backprojection.

                **Kekurangan**
                - Menghasilkan citra yang relatif blur.
                - Struktur tepi kurang tajam.
                - Belum mengompensasi efek spreading pada backprojection.
                """
            )

        with col_fbp:

            st.markdown(
                """
                #### Filtered Back Projection

                **Kelebihan**
                - Mengurangi efek blur SBP.
                - Struktur dan edge lebih jelas.
                - Secara historis merupakan algoritma penting dalam CT.

                **Kekurangan**
                - Lebih sensitif terhadap noise.
                - Memerlukan proses filtering.
                - Hasil dipengaruhi oleh jumlah dan kualitas projection.
                """
            )

        # ====================================================
        # EKSPERIMEN JUMLAH PROYEKSI
        # ====================================================

        st.divider()

        st.subheader(
            "6. Eksperimen: Pengaruh Jumlah Proyeksi"
        )

        st.markdown(
            """
            Coba ubah **Jumlah Proyeksi** di bagian atas dan jalankan
            rekonstruksi kembali.

            Perhatikan:

            1. Apakah struktur phantom masih dapat dikenali?
            2. Apakah muncul artefak?
            3. Apakah edge menjadi lebih kasar?
            4. Apa perbedaan hasil SBP dan FBP ketika jumlah proyeksi sedikit?
            """
        )

        st.warning(
            """
            **Eksperimen mahasiswa**

            Bandingkan hasil rekonstruksi menggunakan jumlah proyeksi yang
            berbeda, misalnya:

            **18 → 36 → 72 → 180 proyeksi**

            Kemudian jelaskan bagaimana angular sampling mempengaruhi kualitas
            rekonstruksi.
            """
        )

        # ====================================================
        # RINGKASAN HASIL EKSPERIMEN
        # ====================================================

        st.divider()

        st.subheader("7. Ringkasan Eksperimen")

        st.markdown(
            f"""
            Rekonstruksi terakhir menggunakan:

            - **Jumlah proyeksi:** {len(theta_used)}
            - **Rentang sudut:** {theta_used[0]:.1f}° – {theta_used[-1]:.1f}°
            - **Metode 1:** Simple Back Projection
            - **Metode 2:** Filtered Back Projection
            - **Filter FBP:** Ram-Lak / Ramp

            Gunakan hasil visualisasi untuk menjelaskan mengapa FBP dapat
            menghasilkan citra yang lebih tajam dibandingkan SBP.
            """
        )


# ============================================================
# PAGE 3 — MODULE 3 — VISUALISASI
# ============================================================

# ============================================================

elif menu_terpilih == "Modul 3: Manipulasi & Visualisasi":
    st.title("Modul 3: Manipulasi & Visualisasi")
    st.info(
        """
        Modul ini akan digunakan untuk mengeksplorasi:
        - Manipulasi sinogram
        - Filtering
        - Perubahan parameter visualisasi
        - Rekonstruksi 3D
        - Visualisasi hasil CT-Scan
        """
    )
