import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import ndimage

### HELPER FUNCTIONS

def load_frames(video_dir, num_frames=None):
    frame_paths = sorted(glob(os.path.join(video_dir, '*.jpg')))
    if num_frames:
        frame_paths = frame_paths[:num_frames]
    frames = []
    for path in frame_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #convert to grayscale
        if img is not None:
            frames.append(img.astype(np.float32))
    return np.array(frames)


def create_1d_gaussian(sigma, truncate=3.0): 
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def derivative_of_gaussian_1d(sigma, truncate=3.0):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = -x / (sigma ** 2) * np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.abs(kernel).sum()
    return kernel


def create_2d_gaussian(sigma, truncate=3.0):
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - radius
            y = j - radius
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def apply_temporal_filter(frames, kernel):
    filtered = np.zeros_like(frames)
    n_frames = len(frames)
    k_len = len(kernel)
    k_half = k_len // 2
    
    for t in range(n_frames):
        result = np.zeros_like(frames[0])
        for i, k_val in enumerate(kernel): # here we apply the kernel (convolution) to the frames
            t_idx = t + i - k_half
            if 0 <= t_idx < n_frames:
                result += k_val * frames[t_idx]
        filtered[t] = result
    return filtered


def apply_spatial_filter(frame, kernel): #snoothing
    if len(kernel.shape) == 1:
        filtered = ndimage.convolve1d(frame, kernel, axis=0)
        filtered = ndimage.convolve1d(filtered, kernel, axis=1)
    else:
        filtered = ndimage.convolve(frame, kernel)
    return filtered


def threshold_motion(derivative, threshold):
    return (np.abs(derivative) > threshold).astype(np.uint8) * 255 # if above threshold, set to 255, else 0


def overlay_mask_on_frame(frame, mask, color=[0, 255, 0]):
    if len(frame.shape) == 2:
        frame_color = cv2.cvtColor((frame).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        frame_color = frame.copy()
    
    mask_binary = (mask > 0).astype(np.uint8)
    overlay = frame_color.copy()
    overlay[mask_binary == 1] = color
    result = cv2.addWeighted(frame_color, 0.7, overlay, 0.3, 0)
    return result


# estimate by removing the largest values (noise) and taking the std of the remaining
def estimate_noise_std(temporal_derivatives, percentile=50):
    # with a percentile of 50, we remove the largest 50% of values (noise above median)
    background_pixels = temporal_derivatives[np.abs(temporal_derivatives) < np.percentile(np.abs(temporal_derivatives), percentile)]
    return np.std(background_pixels)


#basic motion detection using simple temporal derivative
def analyze_basic_pipeline(frames, output_dir='output', frame_idx=25):
    #1. frames converted to grayscale when loaded in
    simple_derivative = np.array([-0.5, 0, 0.5])
    #2. apply the 1d differential operator
    temporal_deriv = apply_temporal_filter(frames, simple_derivative)
    
    #3. threshold the derivatives to create a mask
    threshold_val = 30 # arbitrary
    motion_mask = threshold_motion(temporal_deriv, threshold_val)
    
    #4. combine with original frame
    original_frame = frames[frame_idx]
    derivative_frame = temporal_deriv[frame_idx]
    mask_frame = motion_mask[frame_idx]
    overlay_frame = overlay_mask_on_frame(original_frame, mask_frame)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(original_frame, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.abs(derivative_frame), cmap='hot')
    axes[0, 1].set_title('Temporal Derivative (Absolute)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask_frame, cmap='gray')
    axes[1, 0].set_title(f'Binary Motion Mask (threshold={threshold_val})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_frame)
    axes[1, 1].set_title('Motion Overlay on Original')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q1_basic_pipeline.png'), dpi=150, bbox_inches='tight')
    #save figure
    plt.close()


# variatoin 1: compare simple derivative vs derivative of gaussian filters
def analyze_temporal_filters(frames, output_dir='output', frame_idx=25):
    
    simple_derivative = np.array([-0.5, 0, 0.5])
    t_sigmas = [1.0, 2.0, 3.0]
    
    temporal_derivs = {}
    temporal_derivs['simple'] = apply_temporal_filter(frames, simple_derivative)
    
    # apply dog with our specified sigmas
    for t_sigma in t_sigmas:
        dog_kernel = derivative_of_gaussian_1d(t_sigma)
        temporal_derivs[f'dog_sigma_{t_sigma}'] = apply_temporal_filter(frames, dog_kernel)
    
    threshold_val = 30
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    axes[0, 0].imshow(frames[frame_idx], cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.abs(temporal_derivs['simple'][frame_idx]), cmap='hot')
    axes[0, 1].set_title('Simple Derivative')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(threshold_motion(temporal_derivs['simple'][frame_idx], threshold_val), cmap='gray')
    axes[1, 1].set_title('Simple (Thresholded)')
    axes[1, 1].axis('off')
    
    for idx, t_sigma in enumerate(t_sigmas):
        key = f'dog_sigma_{t_sigma}'
        col = idx + 2
        
        axes[0, col].imshow(np.abs(temporal_derivs[key][frame_idx]), cmap='hot')
        axes[0, col].set_title(f'DoG σ={t_sigma}')
        axes[0, col].axis('off')
        
        axes[1, col].imshow(threshold_motion(temporal_derivs[key][frame_idx], threshold_val), cmap='gray')
        axes[1, col].set_title(f'DoG σ={t_sigma} (Thresh)')
        axes[1, col].axis('off')
    
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q2_temporal_filters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # comparison notes:
    # simple derivative: sharp, noisy, detects all intensity changes
    # derivative of gaussian: smoother, reduces noise with larger sigma
    # larger sigma values suppress high-frequency noise but may blur motion


# variation 2: compare spatial filters: box filters vs gaussian smoothing
def analyze_spatial_smoothing(frames, output_dir='output', frame_idx=25):
    
    simple_derivative = np.array([-0.5, 0, 0.5])
    box_3x3 = np.ones((3, 3)) / 9
    box_5x5 = np.ones((5, 5)) / 25
    s_sigmas = [0.5, 1.0, 2.0]
    
    results = {}
    
    results['no_spatial'] = apply_temporal_filter(frames, simple_derivative)
    
    frames_box3 = np.array([apply_spatial_filter(f, box_3x3) for f in frames])
    results['box_3x3'] = apply_temporal_filter(frames_box3, simple_derivative)
    
    frames_box5 = np.array([apply_spatial_filter(f, box_5x5) for f in frames])
    results['box_5x5'] = apply_temporal_filter(frames_box5, simple_derivative)
    
    # specify sigmas for gaussian filters
    for s_sigma in s_sigmas:
        gaussian_2d = create_2d_gaussian(s_sigma)
        frames_gauss = np.array([apply_spatial_filter(f, gaussian_2d) for f in frames])
        results[f'gauss_sigma_{s_sigma}'] = apply_temporal_filter(frames_gauss, simple_derivative)
    
    threshold_val = 30
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # plotting stuff
    axes[0, 0].imshow(frames[frame_idx], cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.abs(results['no_spatial'][frame_idx]), cmap='hot')
    axes[0, 1].set_title('No Spatial Smoothing')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(threshold_motion(results['no_spatial'][frame_idx], threshold_val), cmap='gray')
    axes[0, 2].set_title('No Spatial (Thresh)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(np.abs(results['box_3x3'][frame_idx]), cmap='hot')
    axes[1, 0].set_title('3x3 Box Filter')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(threshold_motion(results['box_3x3'][frame_idx], threshold_val), cmap='gray')
    axes[1, 1].set_title('3x3 Box (Thresh)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(results['box_5x5'][frame_idx]), cmap='hot')
    axes[1, 2].set_title('5x5 Box Filter')
    axes[1, 2].axis('off')
    
    for idx, s_sigma in enumerate(s_sigmas):
        key = f'gauss_sigma_{s_sigma}'
        row = 2
        col = idx
        
        axes[row, col].imshow(np.abs(results[key][frame_idx]), cmap='hot')
        axes[row, col].set_title(f'Gaussian σ={s_sigma}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q3_spatial_smoothing_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    for idx, s_sigma in enumerate(s_sigmas):
        key = f'gauss_sigma_{s_sigma}'
        col = idx
        
        axes2[0, col].imshow(np.abs(results[key][frame_idx]), cmap='hot')
        axes2[0, col].set_title(f'Gaussian σ={s_sigma}')
        axes2[0, col].axis('off')
        
        axes2[1, col].imshow(threshold_motion(results[key][frame_idx], threshold_val), cmap='gray')
        axes2[1, col].set_title(f'Gaussian σ={s_sigma} (Thresh)')
        axes2[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q3_spatial_smoothing_gaussian.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    #spatial smoothing comparison:
    # no spatial smoothing: noisy, many false positives from sensor noise
    # 3x3 box filter: reduces noise slightly, preserves edges moderately
    # 5x5 box filter: more noise reduction, some edge blurring
    # gaussian filters: best noise reduction with edge preservation
    # optimal: gaussian with sigma =1.0 balances noise reduction and motion detail


# Q4 ie variation 3: adaptive thresholding based on noise estimation
def analyze_adaptive_threshold(frames, output_dir='output', frame_idx=25):
    # idea: first we smooth frames to reduce noise,
    # then we estimate noise std by keeping the smallest values below some threshold, which we deem mostly background (non motion)
    # what we are left with is used to estimate noise std
    # then we threshold the temporal derivative based on this noise std
    s_sigma = 1.0
    gaussian_2d = create_2d_gaussian(s_sigma)
    frames_smoothed = np.array([apply_spatial_filter(f, gaussian_2d) for f in frames])
    
    simple_derivative = np.array([-0.5, 0, 0.5])
    temporal_deriv = apply_temporal_filter(frames_smoothed, simple_derivative)
    
    noise_std = estimate_noise_std(temporal_deriv.flatten())
    
    threshold_multipliers = [1.0, 2.0, 3.0, 4.0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(frames[frame_idx], cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(np.abs(temporal_deriv[frame_idx]), cmap='hot')
    axes[1, 0].set_title('Temporal Derivative')
    axes[1, 0].axis('off')
    
    for idx, mult in enumerate(threshold_multipliers):
        threshold_val = mult * noise_std
        motion_mask = threshold_motion(temporal_deriv[frame_idx], threshold_val)
        
        col = idx + 1
        if col < 4:
            axes[0, col].imshow(motion_mask, cmap='gray')
            axes[0, col].set_title(f'Threshold = {mult}σ ({threshold_val:.1f})')
            axes[0, col].axis('off')
            
            overlay = overlay_mask_on_frame(frames[frame_idx], motion_mask)
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f'{mult}σ Overlay')
            axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q4_adaptive_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    #threshold strategy comparison:
    #sigma = 1: very sensitive, many false positives from noise
    #sigma = 2: balanced, captures most motion with some noise
    #sigma = 3: conservative, good motion detection with minimal false positives
    #sigma = 4: very conservative, may miss subtle motion
    # best seems to be sigma = 3 threshold for robust motion detection


#generate video showing motion detection over all frames
def generate_motion_video(frames, output_dir='output', method='adaptive', fps=10):
    #setup motion detection based on method
    if method == 'adaptive':
        #use adaptive thresholding with spatial smoothing
        s_sigma = 1.0
        gaussian_2d = create_2d_gaussian(s_sigma)
        frames_smoothed = np.array([apply_spatial_filter(f, gaussian_2d) for f in frames])
        simple_derivative = np.array([-0.5, 0, 0.5])
        temporal_deriv = apply_temporal_filter(frames_smoothed, simple_derivative)
        noise_std = estimate_noise_std(temporal_deriv.flatten())
        threshold_val = 3.0 * noise_std
    else:
        #basic method with fixed threshold
        simple_derivative = np.array([-0.5, 0, 0.5])
        temporal_deriv = apply_temporal_filter(frames, simple_derivative)
        threshold_val = 30
    
    motion_mask = threshold_motion(temporal_deriv, threshold_val)
    
    #get frame dimensions
    height, width = frames[0].shape
    
    #setup video writer
    output_path = os.path.join(output_dir, f'motion_video_{method}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    #generate frames with motion overlay
    for i in range(len(frames)):
        overlay = overlay_mask_on_frame(frames[i], motion_mask[i])
        #convert RGB to BGR for opencv
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        video_writer.write(overlay_bgr)
    
    video_writer.release()
    return output_path


def main():
    # configuration - tweak this as you see fit
    video_dir = 'videos/EnterExitCrossingPaths2cor/EnterExitCrossingPaths2cor'
    output_dir = 'output'
    frame_idx = 40  #which frame to display in static images
    generate_videos = True  #set to False to skip video generation
    
    os.makedirs(output_dir, exist_ok=True)
    
    #load video frames
    frames = load_frames(video_dir, num_frames=50)
    
    #q1
    analyze_basic_pipeline(frames, output_dir, frame_idx)
    #q2 variation 1
    analyze_temporal_filters(frames, output_dir, frame_idx)
    #q2 variation 2
    analyze_spatial_smoothing(frames, output_dir, frame_idx)
    #q2 variation 3
    analyze_adaptive_threshold(frames, output_dir, frame_idx)
    
    #generate motion videos if requested
    if generate_videos:
        video_path_basic = generate_motion_video(frames, output_dir, method='basic', fps=10)
        video_path_adaptive = generate_motion_video(frames, output_dir, method='adaptive', fps=10)

if __name__ == '__main__':
    main()
