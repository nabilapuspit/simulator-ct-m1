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
    <h1 style='text-align:center; color:#00e5ff;'>
        🩻 Modul Pembelajaran Interaktif CT-Scan
    </h1>
    <p style='text-align:center; color:#cccccc; font-size:16px;'>
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
# PAGE 0 — THEORY
# ============================================================

if menu_terpilih == "📖 Panduan & Teori Dasar":

    st.markdown("## 📖 Panduan & Teori Dasar CT-Scan")

    st.markdown(
        """
        ### 1. Prinsip Dasar CT-Scan

        Pada CT-Scan, sinar-X melewati objek dari berbagai sudut.
        Ketika melewati material, intensitas sinar-X mengalami atenuasi.

        Secara sederhana:

        $$
        I = I_0 e^{-\\int_L \\mu(x,y)dl}
        $$

        dengan:

        - $I_0$ = intensitas sinar-X sebelum melewati objek
        - $I$ = intensitas setelah melewati objek
        - $\\mu$ = koefisien atenuasi linear
        - $L$ = lintasan sinar-X

        Setelah dilakukan transformasi logaritmik:

        $$
        p(t,\\theta)
        =
        -\\ln\\left(\\frac{I}{I_0}\\right)
        =
        \\int_L \\mu(x,y)dl
        $$

        Nilai $p(t,\\theta)$ disebut sebagai **projection data**.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 2. Hubungan Beam dengan Projection

        Pada setiap sudut $\\theta$, detector mengukur banyak sinar-X
        pada posisi detector $t$ yang berbeda.

        Jika beam melewati objek:

        $$
        p(t,\\theta) > 0
        $$

        Jika beam tidak melewati objek:

        $$
        p(t,\\theta) = 0
        $$

        Dengan demikian, **profil projection 1D merupakan kumpulan
        nilai atenuasi dari seluruh detector channel pada satu sudut**.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 3. Pembentukan Sinogram

        Projection pada berbagai sudut kemudian disusun:

        $$
        p(t,\\theta_1),
        p(t,\\theta_2),
        p(t,\\theta_3), \\ldots
        $$

        sehingga diperoleh:

        $$
        S(t,\\theta)
        $$

        yang disebut sebagai **sinogram**.

        Untuk sebuah titik yang berada di luar pusat rotasi,
        lintasan proyeksinya mengikuti pola sinusoidal:

        $$
        t(\\theta)
        =
        x_0\\cos\\theta+y_0\\sin\\theta
        $$

        Oleh karena itu, sebuah titik pada phantom akan menghasilkan
        kurva sinusoidal pada sinogram.
        """
    )


# ============================================================
# PAGE 1 — MODULE 1
# ============================================================

elif menu_terpilih == "🔬 Modul 1: Akuisisi & Sinogram":

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.header("⚙️ 1. Geometri Pemindaian")

    jenis_phantom = st.sidebar.selectbox(
        "Pilih Objek / Phantom",
        options=[
            "Titik Tunggal (Off-Center Dot)",
            "Shepp-Logan (Anatomi Otak)",
            "Dua Titik (Multi-Dot)",
            "Lingkaran Konsentris"
        ],
        key="k_phantom"
    )

    jumlah_sudut = st.sidebar.slider(
        "Jumlah Proyeksi (Sampling Sudut)",
        min_value=10,
        max_value=360,
        value=180,
        step=10,
        key="k_jml_sudut"
    )

    sudut_maksimal = st.sidebar.slider(
        "Rentang Sudut Total (°)",
        min_value=10,
        max_value=360,
        value=180,
        step=10,
        key="k_max_sudut"
    )

    st.sidebar.header("📻 2. Kondisi Fisika Sinar-X")

    tambah_noise = st.sidebar.checkbox(
        "Simulasi Derau Foton",
        key="k_noise_chk"
    )

    level_noise = 0

    if tambah_noise:

        level_noise = st.sidebar.slider(
            "Tingkat Noise",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="k_noise_lvl"
        )

    st.sidebar.header("🎬 3. Kontrol Akuisisi")

    btn_start = st.sidebar.button(
        "▶ Mulai Pemindaian",
        key="k_btn_start"
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("🎯 Kontrol Beam")

    sudut_aktif = st.sidebar.slider(
        "Sudut Beam θ (°)",
        min_value=0,
        max_value=max(1, int(sudut_maksimal - 1)),
        value=0,
        step=1,
        key="k_angle_manual"
    )

    # Detector coordinate is now a PHYSICAL coordinate
    # rather than an array index.
    t_limit = 80

    detektor_t_manual = st.sidebar.slider(
        "Posisi Detector t",
        min_value=-t_limit,
        max_value=t_limit,
        value=0,
        step=1,
        key="k_t_manual"
    )

    st.sidebar.markdown("---")

    st.sidebar.info(
        """
        **Interpretasi beam**

        🔴 Beam mengenai objek  
        → attenuation > 0

        🔵 Beam tidak mengenai objek  
        → attenuation ≈ 0

        Posisi beam, marker projection,
        dan titik sinogram menggunakan
        koordinat detector yang sama.
        """
    )


    # ========================================================
    # PHANTOM GENERATION
    # ========================================================

    N = 160
    center = N / 2.0

    # Coordinate system:
    #
    # x → right
    # y → upward
    #
    # Center of image = (0, 0)

    Y, X = np.indices((N, N))

    x_grid = X - center + 0.5
    y_grid = Y - center + 0.5


    def generate_phantom(tipe):

        img = np.zeros((N, N), dtype=float)

        if tipe == "Titik Tunggal (Off-Center Dot)":

            # Off-center point
            x0 = 25
            y0 = 20

            radius = 5

            mask = (
                (x_grid - x0) ** 2
                +
                (y_grid - y0) ** 2
                <= radius ** 2
            )

            img[mask] = 1.0


        elif tipe == "Shepp-Logan (Anatomi Otak)":

            raw_img = shepp_logan_phantom()

            img = rescale(
                raw_img,
                scale=N / raw_img.shape[0],
                mode="reflect",
                channel_axis=None
            )

            # Normalize attenuation coefficient
            img = img / np.max(img)


        elif tipe == "Dua Titik (Multi-Dot)":

            x1, y1 = 25, 20
            x2, y2 = -20, -25

            radius = 5

            mask1 = (
                (x_grid - x1) ** 2
                +
                (y_grid - y1) ** 2
                <= radius ** 2
            )

            mask2 = (
                (x_grid - x2) ** 2
                +
                (y_grid - y2) ** 2
                <= radius ** 2
            )

            img[mask1] = 1.0
            img[mask2] = 0.7


        elif tipe == "Lingkaran Konsentris":

            mask1 = x_grid ** 2 + y_grid ** 2 <= 60 ** 2
            mask2 = x_grid ** 2 + y_grid ** 2 <= 35 ** 2
            mask3 = x_grid ** 2 + y_grid ** 2 <= 15 ** 2

            img[mask1] = 0.3
            img[mask2] = 0.7
            img[mask3] = 1.0

        return img


    image = generate_phantom(jenis_phantom)


    # ========================================================
    # DETECTOR GEOMETRY
    # ========================================================

    # Detector channel centers.
    #
    # Instead of:
    #     t = 0,1,2,...159
    #
    # we use:
    #     t = -79.5 ... +79.5
    #
    # This makes t a physical coordinate centered
    # at the axis of rotation.

    detector_t = (
        np.arange(N) - N / 2 + 0.5
    )


    # ========================================================
    # ANGULAR SAMPLING
    # ========================================================

    theta = np.linspace(
        0,
        float(sudut_maksimal),
        int(jumlah_sudut),
        endpoint=False
    )


    # ========================================================
    # FORWARD PROJECTION
    #
    # IMPORTANT:
    #
    # We do NOT use radon() as the primary acquisition model.
    #
    # Instead, every projection is generated by explicitly
    # integrating attenuation along each X-ray path.
    #
    # Line equation:
    #
    # x cos(theta) + y sin(theta) = t
    #
    # Therefore:
    #
    # x = t cos(theta) - s sin(theta)
    # y = t sin(theta) + s cos(theta)
    #
    # This is exactly the same geometry used to draw
    # the active beam.
    # ========================================================

    @st.cache_data(show_spinner=False)
    def calculate_forward_projections(
        image,
        theta_values,
        detector_values
    ):

        n_angles = len(theta_values)
        n_detector = len(detector_values)

        sinogram = np.zeros(
            (n_detector, n_angles),
            dtype=float
        )

        # Sampling length along each beam.
        #
        # The image spans approximately -80 ... +80.
        # Sampling beyond this range guarantees that the
        # complete object is captured.
        s_values = np.linspace(
            -N * 0.8,
            N * 0.8,
            2 * N + 1
        )

        # ds = distance between neighboring samples
        ds = s_values[1] - s_values[0]

        for j, angle in enumerate(theta_values):

            rad = np.deg2rad(angle)

            cos_theta = np.cos(rad)
            sin_theta = np.sin(rad)

            for i, t in enumerate(detector_values):

                # Equation of ray:
                #
                # x = t cosθ - s sinθ
                # y = t sinθ + s cosθ

                x_path = (
                    t * cos_theta
                    -
                    s_values * sin_theta
                )

                y_path = (
                    t * sin_theta
                    +
                    s_values * cos_theta
                )

                # Convert physical coordinate to image array
                # coordinate.
                #
                # image row 0 corresponds to y = -79.5
                # image column 0 corresponds to x = -79.5

                col = x_path + center - 0.5
                row = y_path + center - 0.5

                valid = (
                    (col >= 0)
                    &
                    (col <= N - 1)
                    &
                    (row >= 0)
                    &
                    (row <= N - 1)
                )

                if not np.any(valid):
                    continue

                coords = np.vstack(
                    [
                        row[valid],
                        col[valid]
                    ]
                )

                attenuation_samples = map_coordinates(
                    image,
                    coords,
                    order=1,
                    mode="constant",
                    cval=0.0
                )

                # Line integral:
                #
                # p(t,theta) = ∫ μ(x,y) dl

                sinogram[i, j] = (
                    np.sum(attenuation_samples)
                    * ds
                )

        return sinogram


    # ========================================================
    # CALCULATE SINOGRAM
    # ========================================================

    with st.spinner("Menghitung projection data..."):

        sinogram_clean = calculate_forward_projections(
            image,
            theta,
            detector_t
        )


    # ========================================================
    # OPTIONAL NOISE
    # ========================================================

    if tambah_noise:

        # Simple Gaussian approximation for visualization.
        # The underlying physical model remains the
        # attenuation projection above.

        rng = np.random.default_rng(42)

        noise_sigma = (
            level_noise
            * 0.03
            * max(
                float(np.max(sinogram_clean)),
                1e-6
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

        sinogram_display = sinogram_clean.copy()


    max_attenuation = max(
        float(np.max(sinogram_display) * 1.15),
        1.0
    )


    # ========================================================
    # HELPER FUNCTIONS
    # ========================================================

    def get_detector_index(t_value):

        return int(
            np.argmin(
                np.abs(detector_t - t_value)
            )
        )


    def calculate_beam_geometry(
        angle,
        t_value
    ):

        rad = np.deg2rad(angle)

        # Normal vector of detector
        #
        # n = (cosθ, sinθ)

        normal_x = np.cos(rad)
        normal_y = np.sin(rad)

        # Direction vector along the beam
        #
        # d = (-sinθ, cosθ)

        direction_x = -np.sin(rad)
        direction_y = np.cos(rad)

        # Closest point of the beam to
        # the center of rotation.
        #
        # x cosθ + y sinθ = t

        x_center = t_value * normal_x
        y_center = t_value * normal_y

        return (
            x_center,
            y_center,
            direction_x,
            direction_y
        )


    def calculate_active_measurement(
        angle,
        t_value
    ):

        detector_idx = get_detector_index(
            t_value
        )

        angle_idx = int(
            np.argmin(
                np.abs(theta - angle)
            )
        )

        measured_attenuation = float(
            sinogram_display[
                detector_idx,
                angle_idx
            ]
        )

        return (
            detector_idx,
            angle_idx,
            measured_attenuation
        )


    # ========================================================
    # CURRENT STATE
    # ========================================================

    current_angle = float(
        sudut_aktif
    )

    (
        detector_idx,
        angle_idx,
        active_attenuation
    ) = calculate_active_measurement(
        current_angle,
        detektor_t_manual
    )


    # ========================================================
    # HIT / MISS
    # ========================================================

    # Dynamic threshold.
    #
    # For a beam to be considered interacting with the object,
    # its measured line integral must be greater than a small
    # fraction of the maximum attenuation.

    hit_threshold = (
        max(
            float(np.max(sinogram_display)),
            1e-8
        )
        * 0.01
    )

    beam_hits_object = (
        active_attenuation > hit_threshold
    )

    if beam_hits_object:

        line_color = "#ff1744"
        beam_status = "🔴 BEAM MENGENAI OBJEK"
        status_color = "#ff1744"

    else:

        line_color = "#00e5ff"
        beam_status = "🔵 BEAM TIDAK MENGENAI OBJEK"
        status_color = "#00e5ff"


    # ========================================================
    # ACTIVE PROJECTION PROFILE
    # ========================================================

    projection_profile = sinogram_display[
        :,
        angle_idx
    ]


    # ========================================================
    # BEAM GEOMETRY
    # ========================================================

    (
        beam_x_center,
        beam_y_center,
        direction_x,
        direction_y
    ) = calculate_beam_geometry(
        current_angle,
        detektor_t_manual
    )

    beam_length = N * 0.9

    x1 = (
        beam_x_center
        -
        beam_length * direction_x
    )

    y1 = (
        beam_y_center
        -
        beam_length * direction_y
    )

    x2 = (
        beam_x_center
        +
        beam_length * direction_x
    )

    y2 = (
        beam_y_center
        +
        beam_length * direction_y
    )


    # ========================================================
    # DISPLAY CURRENT MEASUREMENT
    # ========================================================

    st.markdown(
        f"""
        <div style="
            padding:10px;
            border-radius:8px;
            border:2px solid {status_color};
            background-color:#161b22;
            margin-bottom:12px;
        ">
            <h3 style="
                color:{status_color};
                margin:0;
            ">
                {beam_status}
            </h3>

            <p style="margin:5px 0 0 0;">
                Sudut θ = <b>{current_angle:.1f}°</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Detector t = <b>{detektor_t_manual:.1f}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Attenuation =
                <b>{active_attenuation:.3f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CREATE THREE PANELS
    # ========================================================

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(18, 5.2)
    )

    fig.patch.set_facecolor(
        "#0e1117"
    )

    for ax in [ax1, ax2, ax3]:

        ax.set_facecolor(
            "#161b22"
        )


    # ========================================================
    # PANEL 1
    # X-RAY BEAM / PHANTOM
    # ========================================================

    ax1.set_title(
        f"1. Pemindaian Sinar-X ({current_angle:.1f}°)",
        color="#00e5ff",
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
    ax1.plot(
        [x1, x2],
        [y1, y2],
        color=line_color,
        linewidth=2.5,
        linestyle="--",
        label="X-ray beam"
    )

    # Source marker
    #
    # In this simplified parallel-beam representation,
    # the yellow marker represents the active source/ray
    # position.

    ax1.scatter(
        [x2],
        [y2],
        color="#ffea00",
        s=70,
        zorder=5,
        label="Sumber X-ray"
    )

    # Detector-side marker
    ax1.scatter(
        [x1],
        [y1],
        color="#ffffff",
        s=35,
        zorder=5
    )

    # Rotation center
    ax1.scatter(
        [0],
        [0],
        color="#00ff88",
        s=25,
        zorder=5
    )

    ax1.axhline(
        0,
        color="white",
        alpha=0.15,
        linewidth=0.8
    )

    ax1.axvline(
        0,
        color="white",
        alpha=0.15,
        linewidth=0.8
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

    ax1.legend(
        loc="upper right",
        fontsize=8
    )


    # ========================================================
    # PANEL 2
    # 1D PROJECTION
    # ========================================================

    ax2.set_title(
        f"2. Profil Proyeksi 1D ({current_angle:.1f}°)",
        color="#00e5ff",
        fontsize=12,
        fontweight="bold"
    )

    ax2.plot(
        detector_t,
        projection_profile,
        color="#00e5ff",
        linewidth=2
    )

    ax2.fill_between(
        detector_t,
        projection_profile,
        color="#00e5ff",
        alpha=0.18
    )

    # Active detector position

    ax2.axvline(
        x=detektor_t_manual,
        color=line_color,
        linestyle="--",
        linewidth=2,
        label=(
            f"Beam t={detektor_t_manual:.1f}"
        )
    )

    # Active measurement point

    ax2.scatter(
        [detector_t_manual],
        [active_attenuation],
        color="#ffea00",
        s=70,
        zorder=5,
        edgecolors="black",
        linewidths=0.8
    )

    # Horizontal zero reference

    ax2.axhline(
        0,
        color="white",
        alpha=0.3,
        linewidth=1
    )

    ax2.set_xlim(
        detector_t[0],
        detector_t[-1]
    )

    ax2.set_ylim(
        0,
        max_attenuation
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
        f"3. Sinogram Akumulatif (0–{sudut_maksimal}°)",
        color="#00e5ff",
        fontsize=12,
        fontweight="bold"
    )

    # Extent:
    #
    # x = theta
    # y = detector t

    ax3.imshow(
        sinogram_display,
        cmap="bone",
        extent=[
            theta[0],
            theta[-1] if len(theta) > 1 else sudut_maksimal,
            detector_t[0],
            detector_t[-1]
        ],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=max(
            float(np.max(sinogram_display)),
            1e-8
        )
    )

    # Current scanning angle

    ax3.axvline(
        x=current_angle,
        color="#ff1744",
        linewidth=1.8,
        linestyle="--",
        label=(
            f"θ={current_angle:.1f}°"
        )
    )

    # Current detector position

    ax3.scatter(
        [current_angle],
        [detektor_t_manual],
        color="#ffea00",
        s=65,
        zorder=5,
        edgecolors="black",
        linewidths=0.8
    )

    ax3.set_xlim(
        theta[0],
        theta[-1] if len(theta) > 1
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

    ax3.legend(
        loc="upper right",
        fontsize=8
    )


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # NUMERICAL INFORMATION
    # ========================================================

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Sudut Beam",
            f"{current_angle:.1f}°"
        )

    with col2:

        st.metric(
            "Detector Position",
            f"{detektor_t_manual:.1f}"
        )

    with col3:

        st.metric(
            "Attenuation",
            f"{active_attenuation:.3f}"
        )

    with col4:

        if beam_hits_object:

            st.metric(
                "Status",
                "HIT"
            )

        else:

            st.metric(
                "Status",
                "MISS"
            )


    # ========================================================
    # EXPLANATION
    # ========================================================

    if beam_hits_object:

        st.success(
            f"""
            **Beam mengenai phantom.**

            Pada sudut θ = {current_angle:.1f}° dan detector
            position t = {detektor_t_manual:.1f}, sinar-X melewati
            material phantom.

            Akibatnya terdapat line integral attenuation:

            $$
            p(t,\\theta)
            =
            \\int_L \\mu(x,y)dl
            =
            {active_attenuation:.3f}
            $$

            Titik kuning pada panel 2 menunjukkan measurement
            yang sedang dihasilkan oleh beam aktif.
            """
        )

    else:

        st.info(
            f"""
            **Beam tidak mengenai phantom.**

            Pada sudut θ = {current_angle:.1f}° dan detector
            position t = {detektor_t_manual:.1f}, lintasan sinar-X
            tidak melewati material phantom.

            Oleh karena itu:

            $$
            p(t,\\theta) \\approx 0
            $$

            Tidak terdapat peak pada posisi beam tersebut.
            """
        )


    # ========================================================
    # OPTIONAL: START SCAN ANIMATION
    # ========================================================

    if btn_start:

        st.markdown("---")

        st.subheader(
            "🎬 Simulasi Akuisisi Berurutan"
        )

        animation_placeholder = st.empty()

        status_placeholder = st.empty()

        # Start from 0°
        #
        # One frame corresponds to one projection angle.

        for curr_idx in range(
            len(theta)
        ):

            angle_now = float(
                theta[curr_idx]
            )

            # Current projection
            current_projection = (
                sinogram_display[
                    :,
                    curr_idx
                ]
            )

            # Measurement at selected detector t

            active_idx = get_detector_index(
                detektor_t_manual
            )

            active_value = float(
                current_projection[
                    active_idx
                ]
            )

            is_hit = (
                active_value > hit_threshold
            )

            if is_hit:

                current_color = "#ff1744"

            else:

                current_color = "#00e5ff"


            # ------------------------------------------------
            # Beam geometry
            # ------------------------------------------------

            (
                bx,
                by,
                dx,
                dy
            ) = calculate_beam_geometry(
                angle_now,
                detektor_t_manual
            )

            L = N * 0.9

            xa = bx - L * dx
            ya = by - L * dy

            xb = bx + L * dx
            yb = by + L * dy


            # ------------------------------------------------
            # Accumulated sinogram
            # ------------------------------------------------

            accumulated = np.zeros_like(
                sinogram_display
            )

            accumulated[
                :,
                :curr_idx + 1
            ] = sinogram_display[
                :,
                :curr_idx + 1
            ]


            # ------------------------------------------------
            # Create figure
            # ------------------------------------------------

            fig_anim, (
                a1,
                a2,
                a3
            ) = plt.subplots(
                1,
                3,
                figsize=(18, 5.2)
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
                color="#00e5ff",
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
                [xa, xb],
                [ya, yb],
                color=current_color,
                linewidth=2.5,
                linestyle="--"
            )

            a1.scatter(
                [xb],
                [yb],
                color="#ffea00",
                s=70,
                zorder=5
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
                color="#00e5ff",
                fontsize=12,
                fontweight="bold"
            )

            a2.plot(
                detector_t,
                current_projection,
                color="#00e5ff",
                linewidth=2
            )

            a2.fill_between(
                detector_t,
                current_projection,
                color="#00e5ff",
                alpha=0.18
            )

            a2.axvline(
                detektor_t_manual,
                color=current_color,
                linestyle="--",
                linewidth=2
            )

            a2.scatter(
                [detektor_t_manual],
                [active_value],
                color="#ffea00",
                s=70,
                zorder=5
            )

            a2.set_xlim(
                detector_t[0],
                detector_t[-1]
            )

            a2.set_ylim(
                0,
                max_attenuation
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
                "3. Sinogram — Akuisisi Real-Time",
                color="#00e5ff",
                fontsize=12,
                fontweight="bold"
            )

            a3.imshow(
                accumulated,
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
                vmax=max(
                    float(np.max(sinogram_display)),
                    1e-8
                )
            )

            a3.axvline(
                angle_now,
                color="#ff1744",
                linestyle="--",
                linewidth=1.8
            )

            a3.scatter(
                [angle_now],
                [detektor_t_manual],
                color="#ffea00",
                s=65,
                zorder=5
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


            plt.tight_layout()

            animation_placeholder.pyplot(
                fig_anim,
                use_container_width=True
            )

            plt.close(fig_anim)


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if is_hit:

                status_placeholder.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        padding:8px;
                        color:#ff1744;
                        font-weight:bold;
                    ">
                        🔴 θ = {angle_now:.1f}°
                        &nbsp; | &nbsp;
                        BEAM HIT
                        &nbsp; | &nbsp;
                        Attenuation = {active_value:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                status_placeholder.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        padding:8px;
                        color:#00e5ff;
                        font-weight:bold;
                    ">
                        🔵 θ = {angle_now:.1f}°
                        &nbsp; | &nbsp;
                        BEAM MISS
                        &nbsp; | &nbsp;
                        Attenuation = {active_value:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            time.sleep(0.04)


        status_placeholder.success(
            "✅ Akuisisi selesai."
        )


    # ========================================================
    # DOWNLOAD SINOGRAM DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Data Akuisisi"
    )

    csv_buffer = io.StringIO()

    header = (
        "detector_t,"
        +
        ",".join(
            [
                f"theta_{a:.2f}"
                for a in theta
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
# PAGE 2 — MODULE 2
# ============================================================

elif menu_terpilih == "🧩 Modul 2: Rekonstruksi 2D (SBP vs FBP)":

    st.title(
        "🧩 Modul 2: Rekonstruksi 2D"
    )

    st.info(
        """
        Modul rekonstruksi akan dikembangkan menggunakan
        sinogram hasil akuisisi pada Modul 1.

        Metode yang akan dibandingkan:

        1. Simple Back Projection (SBP)
        2. Filtered Back Projection (FBP)
        """
    )


# ============================================================
# PAGE 3 — MODULE 3
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
        - dan visualisasi hasil CT-Scan
        """
    )
