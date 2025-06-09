# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

"""
import keyboard  # pip install keyboard

from capturing import VirtualCamera
from overlays import (
    initialize_hist_figure,
    plot_overlay_to_image,
    plot_strings_to_image,
    update_histogram
)
from basics import (
    histogram_figure_numba,
    calculate_mean,
    calculate_mode,
    calculate_std,
    calculate_min,
    calculate_max,
    calculate_entropy,
    linear_transform,
    histogram_equalization
)


def custom_processing(img_source_generator):
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    for sequence in img_source_generator:
        # --- 1. Linear transformation ---
        sequence = linear_transform(sequence)

        # --- 2. Histogram equalization ---
        sequence = histogram_equalization(sequence)

        # --- 3. Compute RGB histogram ---
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        sequence = plot_overlay_to_image(sequence, fig)

        # --- 4. Compute statistics ---
        mean_r, mean_g, mean_b = calculate_mean(sequence)
        mode_r, mode_g, mode_b = calculate_mode(sequence)
        std_r, std_g, std_b = calculate_std(sequence)
        min_r, min_g, min_b = calculate_min(sequence)
        max_r, max_g, max_b = calculate_max(sequence)
        ent_r, ent_g, ent_b = calculate_entropy(sequence)

        # --- 5. Prepare text overlay ---
        display_text_arr = [
            f"Mean: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}",
            f"Mode: R={mode_r}, G={mode_g}, B={mode_b}",
            f"Std:  R={std_r:.1f}, G={std_g:.1f}, B={std_b:.1f}",
            f"Min:  R={min_r}, G={min_g}, B={min_b}",
            f"Max:  R={max_r}, G={max_g}, B={max_b}",
            f"Entropy: R={ent_r:.2f}, G={ent_g:.2f}, B={ent_b:.2f}"
        ]
        sequence = plot_strings_to_image(sequence, display_text_arr)

        # --- Optional: Keyboard interactions ---
        if keyboard.is_pressed('h'):
            print('h pressed')

        # --- Final output ---
        yield sequence


def main():
    width = 1280
    height = 720
    fps = 30

    vc = VirtualCamera(fps, width, height)

    vc.virtual_cam_interaction(
        custom_processing(
            vc.capture_cv_video(0, bgr_to_rgb=True)
        )
    )


if __name__ == "__main__":
    main()
