import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

plt.style.use('dark_background')

# Load Phantom
image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)
center = image.shape[0] / 2

def simulasi_ct(jumlah_sudut, sudut_maksimal, sudut_aktif):
    sudut_maksimal = float(sudut_maksimal)
    jumlah_sudut = int(jumlah_sudut)
    
    # Hitung Sinogram
    theta = np.linspace(0.0, sudut_maksimal, jumlah_sudut, endpoint=False)
    sinogram = radon(image, theta=theta)
    
    # Indeks sudut aktif
    sudut_aktif_efektif = min(sudut_aktif, sudut_maksimal - 1)
    idx_sudut = int((sudut_aktif_efektif / sudut_maksimal) * jumlah_sudut)
    idx_sudut = min(idx_sudut, jumlah_sudut - 1)
    sudut_sekarang = theta[idx_sudut]
    profil_1d = sinogram[:, idx_sudut]
    
    # Plotting 3 Panel
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0e1117')
    for ax in [ax1, ax2, ax3]: 
        ax.set_facecolor('#161b22')
    
    # Panel 1: Sinar-X
    ax1.set_title(f"1. Pemindaian Sinar-X ({sudut_sekarang:.1f}°)", color='#00e5ff', fontweight='bold', fontsize=11)
    ax1.imshow(image, cmap='bone')
    rad = np.deg2rad(sudut_sekarang + 90)
    length = center * 0.95
    x_line = [center - length * np.cos(rad), center + length * np.cos(rad)]
    y_line = [center - length * np.sin(rad), center + length * np.sin(rad)]
    ax1.plot(x_line, y_line, color='#ff1744', linewidth=2, linestyle='--')
    ax1.scatter([x_line[0]], [y_line[0]], color='#ffea00', s=70, zorder=5)
    ax1.set_axis_off()
    
    # Panel 2: Profil 1D
    ax2.set_title(f"2. Profil Proyeksi 1D ({sudut_sekarang:.1f}°)", color='#00e5ff', fontweight='bold', fontsize=11)
    ax2.plot(profil_1d, color='#00e5ff', linewidth=1.8)
    ax2.fill_between(range(len(profil_1d)), profil_1d, color='#00e5ff', alpha=0.2)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.tick_params(colors='white')
    
    # Panel 3: Sinogram
    ax3.set_title(f"3. Sinogram ({jumlah_sudut} Proyeksi)", color='#00e5ff', fontweight='bold', fontsize=11)
    ax3.imshow(sinogram, cmap='bone', extent=(0, sudut_maksimal, 0, sinogram.shape[0]), aspect='auto', interpolation='nearest')
    ax3.axvline(x=sudut_sekarang, color='#ff1744', linewidth=1.8, linestyle='--')
    ax3.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

# Antarmuka Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# <center>SIMULATOR AKUISISI CT-SCAN & SINOGRAM</center>")
    
    with gr.Row():
        with gr.Column(scale=1):
            slider_jml = gr.Slider(minimum=10, maximum=360, value=180, step=10, label="Jumlah Proyeksi")
            radio_rentang = gr.Radio(choices=[180, 360], value=180, label="Rentang Sudut (°)")
            slider_sudut = gr.Slider(minimum=0, maximum=359, value=45, step=1, label="Sudut Aktif (°)")
        
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Visualisasi Simulasi")
    
    # Event update otomatis saat slider digeser
    inputs = [slider_jml, radio_rentang, slider_sudut]
    for inp in inputs:
        inp.change(fn=simulasi_ct, inputs=inputs, outputs=plot_output)
        
    demo.load(fn=simulasi_ct, inputs=inputs, outputs=plot_output)

demo.launch()