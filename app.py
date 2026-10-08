import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
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
        🩻 Modul Pembelajaran Interaktif CT-Scan
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
        "📖 Panduan & Teori Dasar",
        "🔬 Modul 1: Akuisisi & Sinogram",
        "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)",
        "🎨 Modul 3: Manipulasi & Visualisasi"
    ],
    horizontal=True
)


# ============================================================
# PAGE 0
# PANDUAN & TEORI DASAR
# ============================================================

if menu_terpilih == "📖 Panduan & Teori Dasar":

    st.markdown("## 📖 Panduan & Teori Dasar CT-Scan")

    st.markdown(
        """
        ### 1. Prinsip Dasar CT-Scan

        Pada CT-Scan, sinar-X melewati objek dari berbagai sudut.
        Ketika sinar-X melewati material, intensitas sinar mengalami
        atenuasi.

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
        p(t,\\theta)
        =
        -\\ln\\left(\\frac{I}{I_0}\\right)
        =
        \\int_L \\mu(x,y)\\,dl
        $$

        Nilai $p(t,\\theta)$ disebut sebagai **projection data**.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 2. Beam dan Projection

        Pada satu sudut $\\theta$, detector menerima sinar-X pada
        berbagai posisi detector $t$.

        Setiap posisi $t$ merepresentasikan satu lintasan sinar-X.

        Jika lintasan melewati objek:

        $$
        p(t,\\theta) > 0
        $$

        Jika lintasan tidak melewati objek:

        $$
        p(t,\\theta) = 0
        $$

        Dengan demikian, projection 1D merupakan kumpulan nilai
        atenuasi dari seluruh lintasan sinar-X pada satu sudut.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 3. Mengapa Titik Off-Center Menghasilkan Sinusoid?

        Misalkan sebuah titik pada phantom berada pada koordinat:

        $$
        (x_0,y_0)
        $$

        Posisi proyeksi titik tersebut pada detector mengikuti:

        $$
        \\boxed{
        t(\\theta)
        =
        x_0\\cos\\theta
        +
        y_0\\sin\\theta
        }
        $$

        Ketika $\\theta$ berubah, nilai $t$ berubah secara sinusoidal.

        Karena setiap projection pada setiap sudut dimasukkan ke dalam
        sinogram, lintasan titik tersebut akan terlihat sebagai
        pola sinusoidal.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 4. Urutan Proses pada Simulator

        Simulator ini menggunakan urutan:

        **Phantom**

        ↓

        **X-ray Beam**

        ↓

        **Interaksi dengan Phantom**

        ↓

        **Line Integral Attenuation**

        ↓

        **1D Projection**

        ↓

        **Sinogram**

        Dengan demikian, sinogram tidak hanya ditampilkan sebagai
        gambar akhir, tetapi dibentuk secara bertahap selama proses
        simulasi akuisisi.
        """
    )


# ============================================================
# PAGE 1
# MODULE 1 — AKUISISI & SINOGRAM
# ============================================================

elif menu_terpilih == "🔬 Modul 1: Akuisisi & Sinogram":

    # ========================================================
    # CONSTANTS
    # ========================================================

    N = 160
    IMAGE_CENTER = N / 2.0

    # Physical coordinate of the off-center dot
    OFFCENTER_X = 25.0
    OFFCENTER_Y = 20.0

    # Radius of dot
    DOT_RADIUS = 5.0


    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.header("⚙️ 1. Geometri Pemindaian")

    jenis_phantom = st.sidebar.selectbox(
        "Pilih Objek / Phantom",
        [
            "Titik Tunggal (Off-Center Dot)",
            "Shepp-Logan (Anatomi Otak)",
            "Dua Titik (Multi-Dot)",
            "Lingkaran Konsentris"
        ]
    )

    jumlah_sudut = st.sidebar.slider(
        "Jumlah Proyeksi (Sampling Sudut)",
        min_value=10,
        max_value=360,
        value=180,
        step=10
    )

    sudut_maksimal = st.sidebar.slider(
        "Rentang Sudut Total (°)",
        min_value=10,
        max_value=360,
        value=180,
        step=10
    )


    # ========================================================
    # BEAM MODE
    # ========================================================

    st.sidebar.header("🎯 2. Kontrol Beam")

    mode_beam = st.sidebar.radio(
        "Mode Posisi Beam",
        [
            "Otomatis: Beam Mengikuti Titik",
            "Manual: Geser Beam"
        ]
    )


    if mode_beam == "Manual: Geser Beam":

        detector_t_manual = st.sidebar.slider(
            "Posisi Detector t",
            min_value=-80.0,
            max_value=80.0,
            value=0.0,
            step=0.5
        )

    else:

        detector_t_manual = None


    # ========================================================
    # PHYSICS
    # ========================================================

    st.sidebar.header("📻 3. Kondisi Fisika Sinar-X")

    tambah_noise = st.sidebar.checkbox(
        "Simulasi Derau Pengukuran"
    )

    level_noise = 0

    if tambah_noise:

        level_noise = st.sidebar.slider(
            "Tingkat Noise",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )


    # ========================================================
    # ACQUISITION
    # ========================================================

    st.sidebar.header("🎬 4. Kontrol Akuisisi")

    btn_start = st.sidebar.button(
        "▶ Mulai Pemindaian"
    )

    sudut_aktif = st.sidebar.slider(
        "Sudut Beam θ (°)",
        min_value=0.0,
        max_value=float(max(1, sudut_maksimal - 1)),
        value=0.0,
        step=1.0
    )


    # ========================================================
    # PHANTOM GENERATION
    # ========================================================

    Y, X = np.indices(
        (N, N)
    )

    # Physical coordinates.
    #
    # x increases to the right.
    # y increases upward when displayed with origin="lower".

    x_grid = (
        X
        -
        IMAGE_CENTER
        +
        0.5
    )

    y_grid = (
        Y
        -
        IMAGE_CENTER
        +
        0.5
    )


    def generate_phantom(phantom_type):

        image = np.zeros(
            (N, N),
            dtype=float
        )

        # ----------------------------------------------------
        # OFF-CENTER DOT
        # ----------------------------------------------------

        if phantom_type == "Titik Tunggal (Off-Center Dot)":

            mask = (
                (x_grid - OFFCENTER_X) ** 2
                +
                (y_grid - OFFCENTER_Y) ** 2
                <= DOT_RADIUS ** 2
            )

            image[mask] = 1.0


        # ----------------------------------------------------
        # SHEPP-LOGAN
        # ----------------------------------------------------

        elif phantom_type == "Shepp-Logan (Anatomi Otak)":

            raw = shepp_logan_phantom()

            image = rescale(
                raw,
                scale=N / raw.shape[0],
                mode="reflect",
                channel_axis=None
            )

            image = image / np.max(image)


        # ----------------------------------------------------
        # TWO DOTS
        # ----------------------------------------------------

        elif phantom_type == "Dua Titik (Multi-Dot)":

            mask1 = (
                (x_grid - 25) ** 2
                +
                (y_grid - 20) ** 2
                <= DOT_RADIUS ** 2
            )

            mask2 = (
                (x_grid + 20) ** 2
                +
                (y_grid + 25) ** 2
                <= DOT_RADIUS ** 2
            )

            image[mask1] = 1.0
            image[mask2] = 0.7


        # ----------------------------------------------------
        # CONCENTRIC CIRCLES
        # ----------------------------------------------------

        elif phantom_type == "Lingkaran Konsentris":

            mask1 = (
                x_grid ** 2
                +
                y_grid ** 2
                <= 60 ** 2
            )

            mask2 = (
                x_grid ** 2
                +
                y_grid ** 2
                <= 35 ** 2
            )

            mask3 = (
                x_grid ** 2
                +
                y_grid ** 2
                <= 15 ** 2
            )

            image[mask1] = 0.3
            image[mask2] = 0.7
            image[mask3] = 1.0

        return image


    image = generate_phantom(
        jenis_phantom
    )


    # ========================================================
    # DETECTOR COORDINATE
    # ========================================================

    # Physical detector coordinate.
    #
    # t = 0 corresponds to the center of rotation.

    detector_t = (
        np.arange(N)
        -
        N / 2
        +
        0.5
    )


    # ========================================================
    # ANGULAR SAMPLING
    # ========================================================

    theta = np.linspace(
        0.0,
        float(sudut_maksimal),
        int(jumlah_sudut),
        endpoint=False
    )


    # ========================================================
    # GEOMETRIC PROJECTION OF A POINT
    # ========================================================

    def point_projection_t(
        x0,
        y0,
        angle_deg
    ):

        angle_rad = np.deg2rad(
            angle_deg
        )

        return (
            x0 * np.cos(angle_rad)
            +
            y0 * np.sin(angle_rad)
        )


    # ========================================================
    # BEAM GEOMETRY
    #
    # Line:
    #
    # x cos(theta) + y sin(theta) = t
    #
    # Parameterization:
    #
    # x = t cos(theta) - s sin(theta)
    # y = t sin(theta) + s cos(theta)
    # ========================================================

    def beam_geometry(
        angle_deg,
        t_value
    ):

        angle_rad = np.deg2rad(
            angle_deg
        )

        normal_x = np.cos(
            angle_rad
        )

        normal_y = np.sin(
            angle_rad
        )

        direction_x = -np.sin(
            angle_rad
        )

        direction_y = np.cos(
            angle_rad
        )

        # Closest point of beam to center
        beam_center_x = (
            t_value
            *
            normal_x
        )

        beam_center_y = (
            t_value
            *
            normal_y
        )

        return (
            beam_center_x,
            beam_center_y,
            direction_x,
            direction_y
        )


    # ========================================================
    # FORWARD PROJECTION
    #
    # This explicitly computes:
    #
    # p(t,theta) = integral mu(x,y) dl
    #
    # instead of using radon() as the primary model.
    # ========================================================

    @st.cache_data(
        show_spinner=False
    )
    def calculate_forward_projections(
        image,
        theta_values,
        detector_values
    ):

        image_n = image.shape[0]

        center = image_n / 2.0

        n_detector = len(
            detector_values
        )

        n_angles = len(
            theta_values
        )

        sinogram = np.zeros(
            (
                n_detector,
                n_angles
            ),
            dtype=float
        )

        # Sampling along each ray.
        #
        # The object occupies approximately:
        # -80 ... +80
        #
        # We sample beyond the object so that
        # the complete line integral is captured.

        s_values = np.linspace(
            -image_n * 0.9,
            image_n * 0.9,
            2 * image_n + 1
        )

        ds = (
            s_values[1]
            -
            s_values[0]
        )


        # ----------------------------------------------------
        # LOOP OVER ANGLES
        # ----------------------------------------------------

        for j, angle_deg in enumerate(
            theta_values
        ):

            angle_rad = np.deg2rad(
                angle_deg
            )

            cos_theta = np.cos(
                angle_rad
            )

            sin_theta = np.sin(
                angle_rad
            )


            # ------------------------------------------------
            # LOOP OVER DETECTOR CHANNELS
            # ------------------------------------------------

            for i, t_value in enumerate(
                detector_values
            ):

                # Ray equation
                x_path = (
                    t_value * cos_theta
                    -
                    s_values * sin_theta
                )

                y_path = (
                    t_value * sin_theta
                    +
                    s_values * cos_theta
                )


                # Convert physical coordinates
                # to image array coordinates.

                col = (
                    x_path
                    +
                    center
                    -
                    0.5
                )

                row = (
                    y_path
                    +
                    center
                    -
                    0.5
                )


                valid = (
                    (col >= 0)
                    &
                    (col <= image_n - 1)
                    &
                    (row >= 0)
                    &
                    (row <= image_n - 1)
                )


                if not np.any(valid):

                    continue


                coords = np.vstack(
                    [
                        row[valid],
                        col[valid]
                    ]
                )


                # Bilinear interpolation
                # of attenuation coefficient.

                attenuation_values = map_coordinates(
                    image,
                    coords,
                    order=1,
                    mode="constant",
                    cval=0.0
                )


                # Line integral
                #
                # p = integral(mu dl)

                sinogram[
                    i,
                    j
                ] = (
                    np.sum(
                        attenuation_values
                    )
                    *
                    ds
                )


        return sinogram


    # ========================================================
    # CALCULATE PROJECTION DATA
    # ========================================================

    with st.spinner(
        "Menghitung data projection..."
    ):

        sinogram_clean = (
            calculate_forward_projections(
                image,
                theta,
                detector_t
            )
        )


    # ========================================================
    # OPTIONAL NOISE
    # ========================================================

    if tambah_noise:

        rng = np.random.default_rng(
            42
        )

        noise_sigma = (
            level_noise
            *
            0.02
            *
            max(
                float(
                    np.max(
                        sinogram_clean
                    )
                ),
                1e-8
            )
        )

        sinogram_display = (
            sinogram_clean
            +
            rng.normal(
                0,
                noise_sigma,
                sinogram_clean.shape
            )
        )

        sinogram_display = np.clip(
            sinogram_display,
            0,
            None
        )

    else:

        sinogram_display = (
            sinogram_clean.copy()
        )


    # ========================================================
    # CURRENT ANGLE
    # ========================================================

    current_angle = float(
        sudut_aktif
    )


    # ========================================================
    # DETERMINE ACTIVE BEAM t
    # ========================================================

    if (
        mode_beam
        ==
        "Otomatis: Beam Mengikuti Titik"
    ):

        # For the off-center dot,
        # calculate geometric projection.

        if (
            jenis_phantom
            ==
            "Titik Tunggal (Off-Center Dot)"
        ):

            active_t = point_projection_t(
                OFFCENTER_X,
                OFFCENTER_Y,
                current_angle
            )

        else:

            # For other phantoms,
            # use central beam by default.

            active_t = 0.0

    else:

        active_t = float(
            detector_t_manual
        )


    # ========================================================
    # CLAMP ACTIVE t
    # ========================================================

    active_t = float(
        np.clip(
            active_t,
            detector_t[0],
            detector_t[-1]
        )
    )


    # ========================================================
    # FIND NEAREST DETECTOR CHANNEL
    # ========================================================

    active_detector_idx = int(
        np.argmin(
            np.abs(
                detector_t
                -
                active_t
            )
        )
    )


    # ========================================================
    # FIND NEAREST ANGLE INDEX
    # ========================================================

    active_angle_idx = int(
        np.argmin(
            np.abs(
                theta
                -
                current_angle
            )
        )
    )


    # ========================================================
    # CURRENT PROJECTION
    # ========================================================

    projection_profile = (
        sinogram_display[
            :,
            active_angle_idx
        ]
    )


    # ========================================================
    # ACTIVE ATTENUATION
    # ========================================================

    active_attenuation = float(
        projection_profile[
            active_detector_idx
        ]
    )


    # ========================================================
    # DETERMINE HIT / MISS
    # ========================================================

    max_projection = max(
        float(
            np.max(
                sinogram_display
            )
        ),
        1e-8
    )

    hit_threshold = (
        max_projection
        *
        0.01
    )

    beam_hits_object = (
        active_attenuation
        >
        hit_threshold
    )


    # ========================================================
    # STATUS
    # ========================================================

    if beam_hits_object:

        beam_status = (
            "🔴 BEAM MENGENAI OBJEK"
        )

    else:

        beam_status = (
            "🔵 BEAM TIDAK MENGENAI OBJEK"
        )


    # ========================================================
    # CALCULATE BEAM GEOMETRY
    # ========================================================

    (
        beam_center_x,
        beam_center_y,
        direction_x,
        direction_y
    ) = beam_geometry(
        current_angle,
        active_t
    )


    beam_length = (
        N
        *
        0.95
    )

    beam_x1 = (
        beam_center_x
        -
        beam_length
        *
        direction_x
    )

    beam_y1 = (
        beam_center_y
        -
        beam_length
        *
        direction_y
    )

    beam_x2 = (
        beam_center_x
        +
        beam_length
        *
        direction_x
    )

    beam_y2 = (
        beam_center_y
        +
        beam_length
        *
        direction_y
    )


    # ========================================================
    # STATUS DISPLAY
    # ========================================================

    if beam_hits_object:

        st.error(
            "🔴 BEAM MENGENAI OBJEK"
        )

    else:

        st.info(
            "🔵 BEAM TIDAK MENGENAI OBJEK"
        )


    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Sudut θ",
            f"{current_angle:.1f}°"
        )

    with c2:

        st.metric(
            "Detector t",
            f"{active_t:.2f}"
        )

    with c3:

        st.metric(
            "Attenuation",
            f"{active_attenuation:.3f}"
        )

    with c4:

        st.metric(
            "Beam",
            "HIT"
            if beam_hits_object
            else "MISS"
        )


    # ========================================================
    # THREE PANELS
    # ========================================================

    fig, (
        ax1,
        ax2,
        ax3
    ) = plt.subplots(
        1,
        3,
        figsize=(18, 5.5)
    )

    fig.patch.set_facecolor(
        "#0e1117"
    )

    for ax in [
        ax1,
        ax2,
        ax3
    ]:

        ax.set_facecolor(
            "#161b22"
        )


    # ========================================================
    # PANEL 1
    # PHANTOM + ACTIVE BEAM
    # ========================================================

    ax1.set_title(
        f"1. Pemindaian Sinar-X ({current_angle:.1f}°)",
        color="#5edcff",
        fontsize=12,
        fontweight="bold"
    )

    ax1.imshow(
        image,
        cmap="bone",
        origin="lower",
        extent=[
            -N / 2,
            N / 2,
            -N / 2,
            N / 2
        ]
    )


    # Beam

    beam_color = (
        "#ff1744"
        if beam_hits_object
        else "#00e5ff"
    )

    ax1.plot(
        [
            beam_x1,
            beam_x2
        ],
        [
            beam_y1,
            beam_y2
        ],
        color=beam_color,
        linewidth=2.5,
        linestyle="--"
    )


    # Beam center marker

    ax1.scatter(
        [beam_center_x],
        [beam_center_y],
        color="#ffea00",
        s=45,
        zorder=6
    )


    # Rotation center

    ax1.scatter(
        [0],
        [0],
        color="#00ff88",
        s=25,
        zorder=6
    )


    # If off-center dot is used,
    # show its theoretical projection coordinate.

    if (
        jenis_phantom
        ==
        "Titik Tunggal (Off-Center Dot)"
    ):

        theoretical_t = point_projection_t(
            OFFCENTER_X,
            OFFCENTER_Y,
            current_angle
        )

        ax1.scatter(
            [OFFCENTER_X],
            [OFFCENTER_Y],
            facecolors="none",
            edgecolors="#ffea00",
            s=130,
            linewidths=1.5,
            zorder=7
        )

        # Draw a small normal line from center
        # to theoretical detector coordinate.

        tx = (
            theoretical_t
            *
            np.cos(
                np.deg2rad(
                    current_angle
                )
            )
        )

        ty = (
            theoretical_t
            *
            np.sin(
                np.deg2rad(
                    current_angle
                )
            )
        )

        ax1.scatter(
            [tx],
            [ty],
            color="#ffea00",
            s=20,
            zorder=7
        )


    ax1.set_xlim(
        -N / 2,
        N / 2
    )

    ax1.set_ylim(
        -N / 2,
        N / 2
    )

    ax1.set_xlabel(
        "x"
    )

    ax1.set_ylabel(
        "y"
    )

    ax1.grid(
        True,
        alpha=0.12
    )


    # ========================================================
    # PANEL 2
    # PROJECTION PROFILE
    # ========================================================

    ax2.set_title(
        f"2. Profil Proyeksi 1D ({current_angle:.1f}°)",
        color="#5edcff",
        fontsize=12,
        fontweight="bold"
    )

    ax2.plot(
        detector_t,
        projection_profile,
        color="#5edcff",
        linewidth=2
    )

    ax2.fill_between(
        detector_t,
        projection_profile,
        color="#5edcff",
        alpha=0.18
    )


    # Active beam detector position

    ax2.axvline(
        x=active_t,
        color=beam_color,
        linestyle="--",
        linewidth=2.5,
        label=f"Beam t={active_t:.2f}"
    )


    # Current measurement

    ax2.scatter(
        [active_t],
        [active_attenuation],
        color="#ffea00",
        s=75,
        zorder=8,
        edgecolors="black",
        linewidths=0.8
    )


    # If automatic point-following mode,
    # show theoretical projection position.

    if (
        mode_beam
        ==
        "Otomatis: Beam Mengikuti Titik"
        and
        jenis_phantom
        ==
        "Titik Tunggal (Off-Center Dot)"
    ):

        theoretical_t = point_projection_t(
            OFFCENTER_X,
            OFFCENTER_Y,
            current_angle
        )

        ax2.axvline(
            x=theoretical_t,
            color="#ffea00",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8
        )


    ax2.axhline(
        0,
        color="white",
        alpha=0.25,
        linewidth=1
    )


    ax2.set_xlim(
        detector_t[0],
        detector_t[-1]
    )

    ax2.set_ylim(
        0,
        max(
            float(
                np.max(
                    projection_profile
                )
            )
            *
            1.15,
            1.0
        )
    )

    ax2.set_xlabel(
        "Posisi Detector t"
    )

    ax2.set_ylabel(
        "Line Integral Attenuation"
    )

    ax2.grid(
        True,
        linestyle=":",
        alpha=0.25
    )

    ax2.legend(
        loc="upper right",
        fontsize=8
    )


    # ========================================================
    # PANEL 3
    # SINOGRAM
    # ========================================================

    ax3.set_title(
        f"3. Sinogram ({sudut_maksimal:.0f}°)",
        color="#5edcff",
        fontsize=12,
        fontweight="bold"
    )


    sinogram_max = max(
        float(
            np.max(
                sinogram_display
            )
        ),
        1e-8
    )


    ax3.imshow(
        sinogram_display,
        cmap="bone",
        extent=[
            theta[0],
            theta[-1]
            if len(theta) > 1
            else sudut_maksimal,
            detector_t[0],
            detector_t[-1]
        ],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=sinogram_max
    )


    # Current angle

    ax3.axvline(
        current_angle,
        color="#ff1744",
        linestyle="--",
        linewidth=1.8
    )


    # Current beam point in sinogram

    ax3.scatter(
        [current_angle],
        [active_t],
        color="#ffea00",
        s=70,
        zorder=8,
        edgecolors="black",
        linewidths=0.8
    )


    ax3.set_xlim(
        theta[0],
        theta[-1]
        if len(theta) > 1
        else sudut_maksimal
    )

    ax3.set_ylim(
        detector_t[0],
        detector_t[-1]
    )

    ax3.set_xlabel(
        r"Sudut Proyeksi $\theta$ (°)"
    )

    ax3.set_ylabel(
        "Posisi Detector t"
    )


    # ========================================================
    # SHOW FIGURE
    # ========================================================

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # EDUCATIONAL EXPLANATION
    # ========================================================

    st.markdown("---")

    if beam_hits_object:

        st.success(
            f"""
            ### 🔴 Beam mengenai phantom

            Pada:

            - Sudut: **θ = {current_angle:.1f}°**
            - Posisi detector: **t = {active_t:.2f}**
            - Attenuation: **{active_attenuation:.3f}**

            Beam melewati material phantom sehingga terjadi
            line integral attenuation.

            $$
            p(t,\\theta)
            =
            \\int_L \\mu(x,y)\\,dl
            =
            {active_attenuation:.3f}
            $$

            Karena attenuation pada posisi beam bernilai tinggi,
            marker kuning pada **Profil Proyeksi 1D** berada pada
            bagian peak.

            Titik yang sama juga direpresentasikan oleh marker pada
            **sinogram**.
            """
        )

    else:

        st.info(
            f"""
            ### 🔵 Beam tidak mengenai phantom

            Pada:

            - Sudut: **θ = {current_angle:.1f}°**
            - Posisi detector: **t = {active_t:.2f}**
            - Attenuation: **{active_attenuation:.3f}**

            Lintasan beam tidak melewati phantom.

            Oleh karena itu:

            $$
            p(t,\\theta) \\approx 0
            $$

            Perhatikan bahwa **projection profile secara keseluruhan
            tidak harus kosong**. Peak dapat tetap muncul pada detector
            position lain yang dilewati phantom.

            Yang bernilai nol adalah **measurement pada posisi beam
            yang sedang dipilih**.
            """
        )


    # ========================================================
    # SPECIAL EXPLANATION FOR POINT PHANTOM
    # ========================================================

    if (
        jenis_phantom
        ==
        "Titik Tunggal (Off-Center Dot)"
    ):

        theoretical_t = point_projection_t(
            OFFCENTER_X,
            OFFCENTER_Y,
            current_angle
        )

        st.markdown(
            f"""
            ### 🎯 Posisi Geometrik Titik

            Titik phantom berada pada:

            $$
            (x_0,y_0)
            =
            ({OFFCENTER_X:.0f},{OFFCENTER_Y:.0f})
            $$

            Posisi proyeksinya secara geometrik adalah:

            $$
            t(\\theta)
            =
            x_0\\cos\\theta
            +
            y_0\\sin\\theta
            $$

            sehingga pada θ = **{current_angle:.1f}°**:

            $$
            t(\\theta)
            =
            {theoretical_t:.2f}
            $$

            Nilai inilah yang digunakan oleh mode
            **"Otomatis: Beam Mengikuti Titik"**.
            """
        )


    # ========================================================
    # ANIMATION
    # ========================================================

    if btn_start:

        st.markdown("---")

        st.subheader(
            "🎬 Simulasi Pembentukan Sinogram"
        )

        st.caption(
            "Beam berputar dari sudut awal hingga sudut akhir. "
            "Projection dan sinogram diperbarui pada setiap sudut."
        )


        animation_placeholder = st.empty()

        status_placeholder = st.empty()


        # ----------------------------------------------------
        # ANIMATION LOOP
        # ----------------------------------------------------

        for curr_idx in range(
            len(theta)
        ):

            angle_now = float(
                theta[curr_idx]
            )


            # ------------------------------------------------
            # ACTIVE BEAM POSITION
            # ------------------------------------------------

            if (
                mode_beam
                ==
                "Otomatis: Beam Mengikuti Titik"
            ):

                if (
                    jenis_phantom
                    ==
                    "Titik Tunggal (Off-Center Dot)"
                ):

                    t_now = point_projection_t(
                        OFFCENTER_X,
                        OFFCENTER_Y,
                        angle_now
                    )

                else:

                    t_now = 0.0

            else:

                t_now = float(
                    detector_t_manual
                )


            t_now = float(
                np.clip(
                    t_now,
                    detector_t[0],
                    detector_t[-1]
                )
            )


            # ------------------------------------------------
            # CURRENT DETECTOR INDEX
            # ------------------------------------------------

            detector_idx_now = int(
                np.argmin(
                    np.abs(
                        detector_t
                        -
                        t_now
                    )
                )
            )


            # ------------------------------------------------
            # CURRENT PROJECTION
            # ------------------------------------------------

            projection_now = (
                sinogram_display[
                    :,
                    curr_idx
                ]
            )


            # ------------------------------------------------
            # CURRENT ATTENUATION
            # ------------------------------------------------

            attenuation_now = float(
                projection_now[
                    detector_idx_now
                ]
            )


            # ------------------------------------------------
            # HIT / MISS
            # ------------------------------------------------

            hit_now = (
                attenuation_now
                >
                hit_threshold
            )


            if hit_now:

                color_now = "#ff1744"

            else:

                color_now = "#00e5ff"


            # ------------------------------------------------
            # BEAM GEOMETRY
            # ------------------------------------------------

            (
                bx,
                by,
                dx,
                dy
            ) = beam_geometry(
                angle_now,
                t_now
            )


            L = N * 0.95


            xa = (
                bx
                -
                L * dx
            )

            ya = (
                by
                -
                L * dy
            )

            xb = (
                bx
                +
                L * dx
            )

            yb = (
                by
                +
                L * dy
            )


            # ------------------------------------------------
            # ACCUMULATED SINOGRAM
            # ------------------------------------------------

            accumulated_sinogram = np.zeros_like(
                sinogram_display
            )


            accumulated_sinogram[
                :,
                :curr_idx + 1
            ] = (
                sinogram_display[
                    :,
                    :curr_idx + 1
                ]
            )


            # ------------------------------------------------
            # FIGURE
            # ------------------------------------------------

            fig_anim, (
                a1,
                a2,
                a3
            ) = plt.subplots(
                1,
                3,
                figsize=(18, 5.5)
            )


            fig_anim.patch.set_facecolor(
                "#0e1117"
            )


            for ax in [
                a1,
                a2,
                a3
            ]:

                ax.set_facecolor(
                    "#161b22"
                )


            # =================================================
            # ANIMATION PANEL 1
            # =================================================

            a1.set_title(
                f"1. Pemindaian Sinar-X ({angle_now:.1f}°)",
                color="#5edcff",
                fontsize=12,
                fontweight="bold"
            )


            a1.imshow(
                image,
                cmap="bone",
                origin="lower",
                extent=[
                    -N / 2,
                    N / 2,
                    -N / 2,
                    N / 2
                ]
            )


            a1.plot(
                [
                    xa,
                    xb
                ],
                [
                    ya,
                    yb
                ],
                color=color_now,
                linewidth=2.5,
                linestyle="--"
            )


            # Beam center

            a1.scatter(
                [bx],
                [by],
                color="#ffea00",
                s=45,
                zorder=6
            )


            # Phantom point

            if (
                jenis_phantom
                ==
                "Titik Tunggal (Off-Center Dot)"
            ):

                a1.scatter(
                    [OFFCENTER_X],
                    [OFFCENTER_Y],
                    facecolors="none",
                    edgecolors="#ffea00",
                    s=130,
                    linewidths=1.5,
                    zorder=7
                )


            a1.set_xlim(
                -N / 2,
                N / 2
            )

            a1.set_ylim(
                -N / 2,
                N / 2
            )

            a1.set_axis_off()


            # =================================================
            # ANIMATION PANEL 2
            # =================================================

            a2.set_title(
                f"2. Profil Proyeksi 1D ({angle_now:.1f}°)",
                color="#5edcff",
                fontsize=12,
                fontweight="bold"
            )


            a2.plot(
                detector_t,
                projection_now,
                color="#5edcff",
                linewidth=2
            )


            a2.fill_between(
                detector_t,
                projection_now,
                color="#5edcff",
                alpha=0.18
            )


            # Active beam

            a2.axvline(
                t_now,
                color=color_now,
                linestyle="--",
                linewidth=2.5
            )


            # Active attenuation

            a2.scatter(
                [t_now],
                [attenuation_now],
                color="#ffea00",
                s=75,
                zorder=8
            )


            a2.set_xlim(
                detector_t[0],
                detector_t[-1]
            )

            a2.set_ylim(
                0,
                max(
                    float(
                        np.max(
                            projection_now
                        )
                    )
                    * 1.15,
                    1.0
                )
            )


            a2.set_xlabel(
                "Detector position t"
            )

            a2.set_ylabel(
                "Attenuation"
            )

            a2.grid(
                True,
                linestyle=":",
                alpha=0.25
            )


            # =================================================
            # ANIMATION PANEL 3
            # =================================================

            a3.set_title(
                "3. Sinogram — Dibentuk Real-Time",
                color="#5edcff",
                fontsize=12,
                fontweight="bold"
            )


            a3.imshow(
                accumulated_sinogram,
                cmap="bone",
                extent=[
                    theta[0],
                    theta[-1]
                    if len(theta) > 1
                    else sudut_maksimal,
                    detector_t[0],
                    detector_t[-1]
                ],
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=0,
                vmax=sinogram_max
            )


            # Current angle

            a3.axvline(
                angle_now,
                color="#ff1744",
                linestyle="--",
                linewidth=1.8
            )


            # Current measurement

            a3.scatter(
                [angle_now],
                [t_now],
                color="#ffea00",
                s=70,
                zorder=8
            )


            a3.set_xlim(
                theta[0],
                theta[-1]
                if len(theta) > 1
                else sudut_maksimal
            )

            a3.set_ylim(
                detector_t[0],
                detector_t[-1]
            )


            a3.set_xlabel(
                "Projection angle θ (°)"
            )

            a3.set_ylabel(
                "Detector position t"
            )


            # ------------------------------------------------
            # RENDER
            # ------------------------------------------------

            plt.tight_layout()


            animation_placeholder.pyplot(
                fig_anim,
                use_container_width=True
            )


            plt.close(
                fig_anim
            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if hit_now:

                status_placeholder.error(
                    f"""
                    🔴 **BEAM HIT**
                    
                    θ = {angle_now:.1f}°
                    |
                    t = {t_now:.2f}
                    |
                    Attenuation = {attenuation_now:.3f}
                    """
                )

            else:

                status_placeholder.info(
                    f"""
                    🔵 **BEAM MISS**
                    
                    θ = {angle_now:.1f}°
                    |
                    t = {t_now:.2f}
                    |
                    Attenuation = {attenuation_now:.3f}
                    """
                )


            # Animation speed

            time.sleep(
                0.04
            )


        status_placeholder.success(
            "✅ Akuisisi selesai. Sinogram telah terbentuk."
        )


    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Data Projection"
    )


    csv_buffer = io.StringIO()


    header = (
        "detector_t,"
        +
        ",".join(
            [
                f"theta_{angle:.2f}"
                for angle in theta
            ]
        )
    )


    np.savetxt(
        csv_buffer,
        sinogram_display,
        delimiter=",",
        header=header,
        comments=""
    )


    st.download_button(
        label="⬇️ Download Projection / Sinogram CSV",
        data=csv_buffer.getvalue(),
        file_name="ct_projection_sinogram.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 2
# MODULE 2 — REKONSTRUKSI
# ============================================================

elif menu_terpilih == "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)":

    st.title(
        "🧩 Modul 2: Rekonstruksi 2D"
    )

    st.info(
        """
        Modul rekonstruksi akan menggunakan sinogram yang
        dihasilkan pada Modul 1.

        Metode yang akan dibandingkan:

        1. Simple Back Projection (SBP)
        2. Filtered Back Projection (FBP)
        """
    )


# ============================================================
# PAGE 3
# MODULE 3 — VISUALISASI
# ============================================================

elif menu_terpilih == "🎨 Modul 3: Manipulasi & Visualisasi":

    st.title(
        "🎨 Modul 3: Manipulasi & Visualisasi"
    )

    st.info(
        """
        Modul ini akan digunakan untuk mengeksplorasi:

        - manipulasi sinogram
        - filtering
        - perubahan parameter visualisasi
        - rekonstruksi 3D
        - visualisasi hasil CT-Scan
        """
    )
