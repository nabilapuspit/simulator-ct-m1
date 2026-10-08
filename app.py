import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Virtual CT-Scan Practicum",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩻 Virtual CT-Scan Practicum")

st.markdown(
    """
    **Virtual CT-Scan Simulator**

    Simulasi sederhana prinsip **parallel-beam CT**:
    
    **Phantom → X-ray Beam → Attenuation → Projection → Sinogram**
    """
)


# ============================================================
# SIDEBAR MENU
# ============================================================

st.sidebar.title("📚 Menu")

menu = st.sidebar.radio(
    "Pilih Modul:",
    [
        "📖 Theory",
        "🔬 Module 1 — CT Acquisition",
        "🔄 Module 2 — SBP vs FBP",
        "🎨 Module 3 — Manipulation & Visualization"
    ]
)


# ============================================================
# COMMON FUNCTIONS
# ============================================================

@st.cache_data
def create_shepp_logan(size=128):

    phantom = shepp_logan_phantom()

    phantom = resize(
        phantom,
        (size, size),
        anti_aliasing=True
    )

    return phantom


def create_dot_phantom(
    size=128,
    x0=25,
    y0=20,
    radius=5,
    attenuation=1.0
):

    y, x = np.mgrid[
        -size / 2:size / 2,
        -size / 2:size / 2
    ]

    phantom = np.zeros((size, size))

    mask = (
        (x - x0) ** 2 +
        (y - y0) ** 2
    ) <= radius ** 2

    phantom[mask] = attenuation

    return phantom


def calculate_dot_projection(
    detector_positions,
    theta_deg,
    x0,
    y0,
    radius,
    attenuation
):

    """
    Analytical projection of a circular object.

    Parallel beam geometry:

        t = x cos(theta) + y sin(theta)

    The X-ray beam intersects the circular object when:

        |t - t0| < radius

    where:

        t0 = x0 cos(theta) + y0 sin(theta)

    Projection value is proportional to
    the chord length through the circle.
    """

    theta = np.deg2rad(theta_deg)

    t0 = (
        x0 * np.cos(theta)
        +
        y0 * np.sin(theta)
    )

    distance = np.abs(
        detector_positions - t0
    )

    projection = np.zeros_like(
        detector_positions,
        dtype=float
    )

    inside = distance < radius

    projection[inside] = (
        2
        *
        np.sqrt(
            radius ** 2
            -
            distance[inside] ** 2
        )
        *
        attenuation
    )

    return projection, t0


def create_sinogram_dot(
    detector_positions,
    theta_array,
    x0,
    y0,
    radius,
    attenuation,
    noise_level=0
):

    sinogram = []

    for theta in theta_array:

        projection, _ = calculate_dot_projection(
            detector_positions,
            theta,
            x0,
            y0,
            radius,
            attenuation
        )

        if noise_level > 0:

            noise = np.random.normal(
                0,
                noise_level,
                projection.shape
            )

            projection = projection + noise
            projection = np.clip(
                projection,
                0,
                None
            )

        sinogram.append(projection)

    return np.array(sinogram)


def draw_beam(
    ax,
    theta_deg,
    detector_t,
    beam_length=150
):

    """
    Draw a parallel X-ray beam.

    Geometry:

        x cos(theta) + y sin(theta) = t

    """

    theta = np.deg2rad(theta_deg)

    normal = np.array([
        np.cos(theta),
        np.sin(theta)
    ])

    direction = np.array([
        -np.sin(theta),
        np.cos(theta)
    ])

    center = detector_t * normal

    p1 = (
        center
        -
        beam_length * direction
    )

    p2 = (
        center
        +
        beam_length * direction
    )

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="red",
        linewidth=2.5,
        label="X-ray Beam"
    )


def setup_image_axis(ax, size=128):

    ax.set_xlim(
        -size / 2,
        size / 2
    )

    ax.set_ylim(
        -size / 2,
        size / 2
    )

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.grid(
        alpha=0.2
    )


# ============================================================
# THEORY
# ============================================================

if menu == "📖 Theory":

    st.header("📖 CT-Scan Theory")

    st.subheader("1. Basic CT Acquisition")

    st.markdown(
        """
        Pada CT, objek dipindai dari berbagai arah menggunakan sinar-X.
        Untuk simulasi ini digunakan pendekatan **parallel-beam geometry**.
        
        Alur dasar simulasi:
        
        **Phantom**
        ↓
        **X-ray Beam**
        ↓
        **Attenuation**
        ↓
        **Projection**
        ↓
        **Sinogram**
        ↓
        **Reconstruction**
        """
    )

    st.divider()

    st.subheader("2. Projection")

    st.latex(
        r"""
        p(t,\theta)
        =
        \int_L \mu(x,y)\,dl
        """
    )

    st.markdown(
        """
        Projection merupakan integral koefisien atenuasi sepanjang
        lintasan sinar-X.
        """
    )

    st.divider()

    st.subheader("3. Beer–Lambert Law")

    st.latex(
        r"""
        I = I_0 e^{-p}
        """
    )

    st.markdown(
        """
        atau:
        """
    )

    st.latex(
        r"""
        p = -\ln\left(\frac{I}{I_0}\right)
        """
    )

    st.divider()

    st.subheader("4. Posisi Struktur pada Projection")

    st.latex(
        r"""
        t = x\cos(\theta)+y\sin(\theta)
        """
    )

    st.markdown(
        """
        Untuk sebuah objek kecil pada posisi `(x₀, y₀)`, posisi peak
        pada projection akan berubah ketika sudut pemindaian berubah.
        """
    )

    st.latex(
        r"""
        t(\theta)
        =
        x_0\cos(\theta)
        +
        y_0\sin(\theta)
        """
    )

    st.divider()

    st.subheader("5. Sinogram")

    st.markdown(
        """
        Projection dari berbagai sudut kemudian disusun menjadi
        **sinogram**.
        
        Sumbu:
        
        - Horizontal → posisi detector `t`
        - Vertikal → sudut `θ`
        
        Sebuah objek yang berada di luar pusat rotasi akan menghasilkan
        pola sinusoidal pada sinogram.
        """
    )

    st.latex(
        r"""
        t(\theta)
        =
        x_0\cos(\theta)
        +
        y_0\sin(\theta)
        """
    )


# ============================================================
# MODULE 1
# ============================================================

elif menu == "🔬 Module 1 — CT Acquisition":

    st.header("🔬 Module 1 — CT Acquisition")

    st.markdown(
        """
        Modul ini memperlihatkan hubungan langsung antara:

        **Phantom → Beam → Projection → Sinogram**
        """
    )

    # --------------------------------------------------------
    # DEFAULT PARAMETERS
    # --------------------------------------------------------

    IMAGE_SIZE = 128

    detector_positions = np.linspace(
        -64,
        64,
        257
    )

    # ========================================================
    # TOP SECTION — ANIMATION
    # ========================================================

    st.subheader("🎬 A. Animasi Pemindaian CT")

    st.info(
        """
        Animasi menggunakan interval sudut **10°**:
        
        0° → 10° → 20° → ... → 170°
        
        Total = **18 projection views**.
        """
    )

    animation_col1, animation_col2 = st.columns(
        [1, 4]
    )

    with animation_col1:

        start_animation = st.button(
            "▶️ Mulai Scan",
            type="primary",
            use_container_width=True
        )

        animation_speed = st.slider(
            "Kecepatan animasi",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05
        )

    with animation_col2:

        st.markdown(
            """
            **Interpretasi animasi**
            
            Setiap frame menunjukkan satu projection pada satu sudut.
            Beam merah ditempatkan pada detector position yang melewati
            struktur objek.
            
            Peak pada projection dan titik pada sinogram harus berada
            pada posisi yang sama.
            """
        )

    # --------------------------------------------------------
    # FIXED ANIMATION PHANTOM
    # --------------------------------------------------------

    animation_x0 = 25
    animation_y0 = 20
    animation_radius = 6
    animation_attenuation = 1.0

    animation_theta = np.arange(
        0,
        180,
        10
    )

    animation_phantom = create_dot_phantom(
        size=IMAGE_SIZE,
        x0=animation_x0,
        y0=animation_y0,
        radius=animation_radius,
        attenuation=animation_attenuation
    )

    animation_sinogram = np.zeros(
        (
            len(animation_theta),
            len(detector_positions)
        )
    )

    # --------------------------------------------------------
    # PLACEHOLDER
    # --------------------------------------------------------

    animation_placeholder = st.empty()

    if start_animation:

        for frame_idx, theta in enumerate(
            animation_theta
        ):

            projection, active_t = (
                calculate_dot_projection(
                    detector_positions,
                    theta,
                    animation_x0,
                    animation_y0,
                    animation_radius,
                    animation_attenuation
                )
            )

            animation_sinogram[
                frame_idx,
                :
            ] = projection

            # ------------------------------------------------
            # CREATE FIGURE
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5)
            )

            ax1, ax2, ax3 = axes

            # =================================================
            # PANEL 1 — PHANTOM + BEAM
            # =================================================

            ax1.imshow(
                animation_phantom,
                cmap="gray",
                origin="lower",
                extent=[
                    -64,
                    64,
                    -64,
                    64
                ],
                vmin=0,
                vmax=1
            )

            draw_beam(
                ax1,
                theta,
                active_t
            )

            ax1.scatter(
                animation_x0,
                animation_y0,
                color="yellow",
                edgecolor="black",
                s=50,
                zorder=5
            )

            ax1.set_title(
                f"Phantom + X-ray Beam\nθ = {theta}°"
            )

            setup_image_axis(
                ax1,
                IMAGE_SIZE
            )

            # =================================================
            # PANEL 2 — PROJECTION
            # =================================================

            ax2.plot(
                detector_positions,
                projection,
                linewidth=2
            )

            ax2.axvline(
                active_t,
                linestyle="--",
                linewidth=2
            )

            ax2.scatter(
                active_t,
                np.max(projection),
                s=70,
                zorder=5
            )

            ax2.set_xlabel(
                "Detector position t"
            )

            ax2.set_ylabel(
                "Projection p(t, θ)"
            )

            ax2.set_title(
                "Projection Profile"
            )

            ax2.grid(
                alpha=0.25
            )

            # =================================================
            # PANEL 3 — SINOGRAM
            # =================================================

            masked_sinogram = np.ma.masked_where(
                animation_sinogram == 0,
                animation_sinogram
            )

            ax3.imshow(
                masked_sinogram,
                cmap="gray",
                origin="lower",
                aspect="auto",
                extent=[
                    detector_positions[0],
                    detector_positions[-1],
                    0,
                    170
                ]
            )

            ax3.scatter(
                active_t,
                theta,
                color="red",
                s=50
            )

            ax3.set_xlabel(
                "Detector position t"
            )

            ax3.set_ylabel(
                "Projection angle θ"
            )

            ax3.set_title(
                "Sinogram"
            )

            ax3.set_ylim(
                0,
                170
            )

            # =================================================
            # STATUS
            # =================================================

            fig.suptitle(
                (
                    f"CT Acquisition — "
                    f"View {frame_idx + 1}/{len(animation_theta)}"
                    f" | θ = {theta}°"
                    f" | Active detector t = {active_t:.2f}"
                ),
                fontsize=14
            )

            plt.tight_layout()

            animation_placeholder.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

            time.sleep(
                animation_speed
            )

        st.success(
            "✅ Pemindaian selesai: 18 projection views dari 0°–170°."
        )

    else:

        # ----------------------------------------------------
        # INITIAL STATIC FRAME
        # ----------------------------------------------------

        theta = 0

        projection, active_t = (
            calculate_dot_projection(
                detector_positions,
                theta,
                animation_x0,
                animation_y0,
                animation_radius,
                animation_attenuation
            )
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5)
        )

        ax1, ax2, ax3 = axes

        # Phantom

        ax1.imshow(
            animation_phantom,
            cmap="gray",
            origin="lower",
            extent=[
                -64,
                64,
                -64,
                64
            ],
            vmin=0,
            vmax=1
        )

        draw_beam(
            ax1,
            theta,
            active_t
        )

        ax1.scatter(
            animation_x0,
            animation_y0,
            color="yellow",
            edgecolor="black",
            s=50
        )

        ax1.set_title(
            "Phantom + X-ray Beam"
        )

        setup_image_axis(
            ax1,
            IMAGE_SIZE
        )

        # Projection

        ax2.plot(
            detector_positions,
            projection,
            linewidth=2
        )

        ax2.axvline(
            active_t,
            linestyle="--",
            linewidth=2
        )

        ax2.set_title(
            "Projection Profile"
        )

        ax2.set_xlabel(
            "Detector position t"
        )

        ax2.set_ylabel(
            "Projection p(t, θ)"
        )

        ax2.grid(
            alpha=0.25
        )

        # Empty sinogram

        ax3.set_title(
            "Sinogram — tekan 'Mulai Scan'"
        )

        ax3.set_xlabel(
            "Detector position t"
        )

        ax3.set_ylabel(
            "Projection angle θ"
        )

        ax3.set_xlim(
            -64,
            64
        )

        ax3.set_ylim(
            0,
            170
        )

        plt.tight_layout()

        animation_placeholder.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    # ========================================================
    # DIVIDER
    # ========================================================

    st.divider()

    # ========================================================
    # LOWER SECTION — INTERACTIVE EXPLORATION
    # ========================================================

    st.subheader(
        "🎛️ B. Eksplorasi Interaktif"
    )

    st.markdown(
        """
        Setelah memahami animasi di atas, mahasiswa dapat mengubah
        parameter pemindaian secara manual dan mengamati bagaimana
        perubahan tersebut memengaruhi **beam, projection, dan sinogram**.
        """
    )

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    control_col1, control_col2, control_col3 = st.columns(
        3
    )

    with control_col1:

        st.markdown(
            "### 🟢 Object"
        )

        x0 = st.slider(
            "Posisi X objek",
            -40,
            40,
            25,
            1
        )

        y0 = st.slider(
            "Posisi Y objek",
            -40,
            40,
            20,
            1
        )

        radius = st.slider(
            "Radius objek",
            2,
            15,
            6,
            1
        )

        attenuation = st.slider(
            "Koefisien atenuasi μ",
            0.1,
            2.0,
            1.0,
            0.1
        )

    with control_col2:

        st.markdown(
            "### 🔵 Acquisition"
        )

        angle_options = list(
            np.arange(
                0,
                180,
                10
            )
        )

        selected_theta = st.select_slider(
            "Sudut pemindaian θ",
            options=angle_options,
            value=0
        )

        noise_level = st.slider(
            "Noise level",
            0.0,
            0.5,
            0.0,
            0.01
        )

    with control_col3:

        st.markdown(
            "### 🔴 Detector"

        )

        detector_t = st.slider(
            "Posisi detector t",
            -64.0,
            64.0,
            0.0,
            0.5
        )

        show_beam = st.checkbox(
            "Tampilkan X-ray beam",
            value=True
        )

        show_object_center = st.checkbox(
            "Tampilkan posisi objek",
            value=True
        )

    # ========================================================
    # INTERACTIVE CALCULATION
    # ========================================================

    interactive_phantom = create_dot_phantom(
        size=IMAGE_SIZE,
        x0=x0,
        y0=y0,
        radius=radius,
        attenuation=attenuation
    )

    interactive_projection, physical_t = (
        calculate_dot_projection(
            detector_positions,
            selected_theta,
            x0,
            y0,
            radius,
            attenuation
        )
    )

    # Add noise

    if noise_level > 0:

        np.random.seed(42)

        interactive_projection = (
            interactive_projection
            +
            np.random.normal(
                0,
                noise_level,
                interactive_projection.shape
            )
        )

        interactive_projection = np.clip(
            interactive_projection,
            0,
            None
        )

    # ========================================================
    # MANUAL BEAM PROJECTION
    # ========================================================

    manual_distance = np.abs(
        detector_t - physical_t
    )

    if manual_distance < radius:

        manual_projection = (
            2
            *
            np.sqrt(
                radius ** 2
                -
                manual_distance ** 2
            )
            *
            attenuation
        )

    else:

        manual_projection = 0.0

    # ========================================================
    # MANUAL SINOGRAM
    # ========================================================

    manual_theta_array = np.arange(
        0,
        180,
        10
    )

    manual_sinogram = create_sinogram_dot(
        detector_positions,
        manual_theta_array,
        x0,
        y0,
        radius,
        attenuation,
        noise_level
    )

    # ========================================================
    # DISPLAY INTERACTIVE VIEW
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    ax1, ax2, ax3 = axes

    # --------------------------------------------------------
    # PHANTOM
    # --------------------------------------------------------

    ax1.imshow(
        interactive_phantom,
        cmap="gray",
        origin="lower",
        extent=[
            -64,
            64,
            -64,
            64
        ],
        vmin=0,
        vmax=max(
            attenuation,
            1
        )
    )

    if show_beam:

        draw_beam(
            ax1,
            selected_theta,
            detector_t
        )

    if show_object_center:

        ax1.scatter(
            x0,
            y0,
            color="yellow",
            edgecolor="black",
            s=60,
            zorder=5
        )

    ax1.set_title(
        f"Phantom + Beam\nθ = {selected_theta}°"
    )

    setup_image_axis(
        ax1,
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # PROJECTION
    # --------------------------------------------------------

    ax2.plot(
        detector_positions,
        interactive_projection,
        linewidth=2
    )

    ax2.axvline(
        detector_t,
        linestyle="--",
        linewidth=2
    )

    if np.max(interactive_projection) > 0:

        ax2.scatter(
            physical_t,
            np.max(interactive_projection),
            s=70,
            zorder=5
        )

    ax2.set_title(
        "Projection Profile"
    )

    ax2.set_xlabel(
        "Detector position t"
    )

    ax2.set_ylabel(
        "Projection p(t, θ)"
    )

    ax2.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # SINOGRAM
    # --------------------------------------------------------

    ax3.imshow(
        manual_sinogram,
        cmap="gray",
        origin="lower",
        aspect="auto",
        extent=[
            detector_positions[0],
            detector_positions[-1],
            0,
            170
        ]
    )

    ax3.scatter(
        physical_t,
        selected_theta,
        color="red",
        s=70
    )

    ax3.axhline(
        selected_theta,
        linestyle="--",
        linewidth=1
    )

    ax3.axvline(
        physical_t,
        linestyle="--",
        linewidth=1
    )

    ax3.set_title(
        "Sinogram"
    )

    ax3.set_xlabel(
        "Detector position t"
    )

    ax3.set_ylabel(
        "Projection angle θ"
    )

    ax3.set_ylim(
        0,
        170
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    # ========================================================
    # NUMERICAL RESULT
    # ========================================================

    st.divider()

    result_col1, result_col2, result_col3 = st.columns(
        3
    )

    with result_col1:

        st.metric(
            "Sudut θ",
            f"{selected_theta}°"
        )

    with result_col2:

        st.metric(
            "Posisi struktur t",
            f"{physical_t:.2f}"
        )

    with result_col3:

        st.metric(
            "Measurement beam",
            f"{manual_projection:.3f}"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    if manual_projection > 0:

        st.success(
            f"""
            **Beam mengenai objek.**
            
            Pada θ = {selected_theta}°, detector position
            t = {detector_t:.2f} melewati objek.
            
            Nilai projection:
            
            **p = {manual_projection:.3f}**
            """
        )

    else:

        st.info(
            f"""
            **Beam tidak mengenai objek.**
            
            Pada θ = {selected_theta}°, detector position
            t = {detector_t:.2f} berada di luar objek.
            
            Karena tidak ada material yang dilewati beam,
            measurement idealnya:
            
            **p = 0**
            """
        )

    st.caption(
        """
        Catatan: simulasi ini menggunakan model ideal parallel-beam
        untuk tujuan pembelajaran. Sistem CT klinis menggunakan geometri
        dan mekanisme akuisisi yang lebih kompleks.
        """
    )


# ============================================================
# MODULE 2
# ============================================================

elif menu == "🔄 Module 2 — SBP vs FBP":

    st.header(
        "🔄 Module 2 — Simple Back Projection vs Filtered Back Projection"
    )

    st.markdown(
        """
        Modul ini menunjukkan perbedaan antara:

        - **SBP — Simple Back Projection**
        - **FBP — Filtered Back Projection**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        image_size = st.slider(
            "Ukuran phantom",
            64,
            256,
            128,
            16
        )

    with col2:

        number_of_projections = st.slider(
            "Jumlah projection views",
            18,
            180,
            60,
            18
        )

    # --------------------------------------------------------
    # PHANTOM
    # --------------------------------------------------------

    phantom = create_shepp_logan(
        image_size
    )

    theta = np.linspace(
        0,
        180,
        number_of_projections,
        endpoint=False
    )

    # --------------------------------------------------------
    # RADON
    # --------------------------------------------------------

    sinogram = radon(
        phantom,
        theta=theta,
        circle=True
    )

    # --------------------------------------------------------
    # SBP
    # --------------------------------------------------------

    reconstruction_sbp = iradon(
        sinogram,
        theta=theta,
        filter_name=None,
        circle=True
    )

    # --------------------------------------------------------
    # FBP
    # --------------------------------------------------------

    reconstruction_fbp = iradon(
        sinogram,
        theta=theta,
        filter_name="ramp",
        circle=True
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(
        phantom,
        cmap="gray"
    )

    axes[0].set_title(
        "Original Phantom"
    )

    axes[1].imshow(
        sinogram,
        cmap="gray",
        aspect="auto"
    )

    axes[1].set_title(
        f"Sinogram\n{number_of_projections} views"
    )

    axes[2].imshow(
        reconstruction_sbp,
        cmap="gray"
    )

    axes[2].set_title(
        "Simple Back Projection"
    )

    axes[3].imshow(
        reconstruction_fbp,
        cmap="gray"
    )

    axes[3].set_title(
        "Filtered Back Projection"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.info(
        """
        **Observasi:**
        
        SBP menghasilkan citra yang lebih blur karena projection
        langsung diback-project tanpa filtering.
        
        FBP menggunakan filtering sebelum back-projection sehingga
        struktur dan edge dapat dipertahankan lebih baik.
        """
    )


# ============================================================
# MODULE 3
# ============================================================

elif menu == "🎨 Module 3 — Manipulation & Visualization":

    st.header(
        "🎨 Module 3 — Manipulation & Visualization"
    )

    st.markdown(
        """
        Modul ini digunakan untuk mengeksplorasi pengaruh
        **angular sampling** terhadap kualitas sinogram dan hasil
        rekonstruksi.
        """
    )

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        number_of_views = st.select_slider(
            "Jumlah projection views",
            options=[
                18,
                30,
                45,
                60,
                90,
                120,
                180
            ],
            value=18
        )

    with col2:

        noise_sigma = st.slider(
            "Noise",
            0.0,
            0.5,
            0.0,
            0.01
        )

    # --------------------------------------------------------
    # PHANTOM
    # --------------------------------------------------------

    phantom = create_shepp_logan(
        128
    )

    theta = np.linspace(
        0,
        180,
        number_of_views,
        endpoint=False
    )

    # --------------------------------------------------------
    # SINOGRAM
    # --------------------------------------------------------

    sinogram = radon(
        phantom,
        theta=theta,
        circle=True
    )

    # --------------------------------------------------------
    # ADD NOISE
    # --------------------------------------------------------

    if noise_sigma > 0:

        np.random.seed(42)

        sinogram = (
            sinogram
            +
            np.random.normal(
                0,
                noise_sigma,
                sinogram.shape
            )
        )

    # --------------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------------

    reconstruction = iradon(
        sinogram,
        theta=theta,
        filter_name="ramp",
        circle=True
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    axes[0].imshow(
        phantom,
        cmap="gray"
    )

    axes[0].set_title(
        "Original Phantom"
    )

    axes[1].imshow(
        sinogram,
        cmap="gray",
        aspect="auto"
    )

    axes[1].set_title(
        f"Sinogram — {number_of_views} views"
    )

    axes[2].imshow(
        reconstruction,
        cmap="gray"
    )

    axes[2].set_title(
        "FBP Reconstruction"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "💡 What should you observe?"
    )

    if number_of_views <= 30:

        st.warning(
            """
            Jumlah projection relatif sedikit.
            Perhatikan kemungkinan munculnya **streak artifacts**
            akibat angular undersampling.
            """
        )

    elif number_of_views <= 60:

        st.info(
            """
            Projection mulai lebih rapat sehingga kualitas
            rekonstruksi meningkat dibandingkan jumlah view yang sangat
            sedikit.
            """
        )

    else:

        st.success(
            """
            Projection lebih rapat sehingga sampling angular menjadi
            lebih baik dan hasil rekonstruksi umumnya lebih stabil.
            """
        )

    if noise_sigma > 0:

        st.warning(
            """
            Noise ditambahkan ke projection data.
            Perhatikan bagaimana noise pada sinogram dapat memengaruhi
            hasil rekonstruksi.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Virtual CT-Scan Practicum | Biomedical Imaging | Teknik Biomedik ITS
    """
)
