# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
import math

from numba import njit  # conda install numba


@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''

    height = np_img.shape[0]
    width = np_img.shape[1]

    # initialise empty histogram
    r_hist = [0.0] * 256
    g_hist = [0.0] * 256
    b_hist = [0.0] * 256

    # counting pixel values
    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]
            r_hist[r] += 1
            g_hist[g] += 1
            b_hist[b] += 1

    # normalize for a range [0,3[
    total_pixels = height * width
    for i in range(256):
        r_hist[i] = (r_hist[i] / total_pixels) * 3.0
        g_hist[i] = (g_hist[i] / total_pixels) * 3.0
        b_hist[i] = (b_hist[i] / total_pixels) * 3.0

    # return r_bars, g_bars, b_bars
    return r_hist, g_hist, b_hist


####

### All other basic functions

def calculate_mean(np_img):
    """
    Returns the mean value for each color channel (R, G, B) of the image.
    """
    height = np_img.shape[0]
    width = np_img.shape[1]

    total_pixels = height * width

    # initialise sum for each color channel
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0

    # calculating total sum for each color channel
    for y in range(height):
        for x in range(width):
            pixel = np_img[y, x]
            sum_r += pixel[0]
            sum_g += pixel[1]
            sum_b += pixel[2]

    # mean formula
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    return mean_r, mean_g, mean_b


def calculate_mode(np_img):
    """
    Returns the mode (most common value) for each channel in the image.
    """
    height = np_img.shape[0]
    width = np_img.shape[1]

    # initialise empty histogram
    r_hist = [0.0] * 256
    g_hist = [0.0] * 256
    b_hist = [0.0] * 256

    # counting pixel values
    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]
            r_hist[r] += 1
            g_hist[g] += 1
            b_hist[b] += 1

    # find the index (intensity) with the maximum count
    mode_r = 0
    mode_g = 0
    mode_b = 0

    max_r = r_hist[0]
    max_g = g_hist[0]
    max_b = b_hist[0]

    '''
    comparing y_hist[i] with max_y
    e.g. max_r =45 and r_hist[i=45] = 70, 
    max_r will be i= 45 because it is now the intensity with highest count
    '''
    for i in range(1, 256):
        if r_hist[i] > max_r:
            max_r = r_hist[i]
            mode_r = i
        if g_hist[i] > max_g:
            max_g = g_hist[i]
            mode_g = i
        if b_hist[i] > max_b:
            max_b = b_hist[i]
            mode_b = i

    return mode_r, mode_g, mode_b


def calculate_std(np_img):
    """
    Returns the standard deviation of pixel values for each channel (R, G, B).
    """

    mean_r, mean_g, mean_b = calculate_mean(np_img)
    height = np_img.shape[0]
    width = np_img.shape[1]
    pixel_count = height * width

    sqrt_diff_r = 0
    sqrt_diff_g = 0
    sqrt_diff_b = 0

    # calculating squared differences
    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]

            sqrt_diff_r += (r - mean_r) ** 2
            sqrt_diff_g += (g - mean_g) ** 2
            sqrt_diff_b += (b - mean_b) ** 2

    # compute standard deviation
    sd_r = math.sqrt(sqrt_diff_r / pixel_count)
    sd_g = math.sqrt(sqrt_diff_g / pixel_count)
    sd_b = math.sqrt(sqrt_diff_b / pixel_count)

    return sd_r, sd_g, sd_b


def calculate_max(np_img):
    """
    Returns the maximum pixel value per channel (R, G, B).
    """
    height = np_img.shape[0]
    width = np_img.shape[1]

    # initialize with min possible value (since we want to find maximum)
    max_r = 0
    max_g = 0
    max_b = 0

    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]

            if r > max_r:
                max_r = r
            if g > max_g:
                max_g = g
            if b > max_b:
                max_b = b

    return max_r, max_g, max_b


def calculate_min(np_img):
    """
    Returns the minimum pixel value per channel (R, G, B).
    """
    height = np_img.shape[0]
    width = np_img.shape[1]

    # initialize with max possible value (since we want to find minimum)
    min_r = 255
    min_g = 255
    min_b = 255

    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]

            if r < min_r:
                min_r = r
            if g < min_g:
                min_g = g
            if b < min_b:
                min_b = b

    return min_r, min_g, min_b



def calculate_entropy(np_img):
    """
    Calculates the Shannon entropy of the image per channel.
    """
    height = np_img.shape[0]
    width = np_img.shape[1]
    pixel_count = height * width

    # initialise empty histogram
    r_hist = [0.0] * 256
    g_hist = [0.0] * 256
    b_hist = [0.0] * 256


    # creating hists to get intensity counts
    for y in range(height):
        for x in range(width):
            r = np_img[y, x, 0]
            g = np_img[y, x, 1]
            b = np_img[y, x, 2]
            r_hist[r] += 1
            g_hist[g] += 1
            b_hist[b] += 1


    ent_r = 0.0
    ent_g = 0.0
    ent_b = 0.0

    # computing probability at each intensity
    for i in range(256):
        if r_hist[i] > 0:
            p = r_hist[i] / pixel_count
            ent_r -= p * math.log2(p)
        if g_hist[i] > 0:
            p = g_hist[i] / pixel_count
            ent_g -= p * math.log2(p)
        if b_hist[i] > 0:
            p = b_hist[i] / pixel_count
            ent_b -= p * math.log2(p)

    return ent_r, ent_g, ent_b
####
