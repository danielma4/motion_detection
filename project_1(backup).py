import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Loading and preprocessing the images
def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error loading image: {image_path}")
        return None, None
    image = cv2.resize(image, (1280, 720))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def compute_temporal_derivative(frame1, frame2):
    """
    Compute temporal derivative between consecutive frames
    """
    diff = cv2.absdiff(frame1, frame2)
    return diff

def create_motion_mask(diff, threshold_value=30):
    """
    Threshold the temporal derivative to create binary mask
    """
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    mask = (thresh > 0).astype(np.uint8)
    return thresh, mask

def apply_mask_to_frame(original_frame, mask):
    """
    Combine mask with original frame to highlight moving objects
    """
    mask_3channel = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    masked_frame = cv2.bitwise_and(original_frame, mask_3channel)
    return masked_frame

def get_image_files_from_folder(folder_path):
    """
    Get all image files from a specific folder using pathlib
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return []
    
    # Get all image files (*.jpg, *.jpeg, *.png)
    image_files = []
    
    # Use rglob to search recursively if needed, or glob for just the folder
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(list(folder_path.glob(pattern)))
    
    # Sort the files
    image_files.sort()
    
    print(f"\nFound {len(image_files)} images in {folder_path.name}/")
    for img in image_files:
        print(f"  - {img.name}")
    
    return image_files

def get_all_images_from_subdirectories(base_dir, subdirs):
    """
    Get images from multiple subdirectories
    Returns a dictionary: {subdirectory_name: [list of image paths]}
    """
    base_path = Path(base_dir)
    all_images = {}
    
    for subdir in subdirs:
        subdir_path = base_path / subdir
        images = get_image_files_from_folder(subdir_path)
        if images:
            all_images[subdir] = images
    
    return all_images

def process_consecutive_frames(image_files, frame_idx1=0, frame_idx2=1, threshold_value=30, folder_name=""):
    """
    Process two consecutive frames from a list of image files
    """
    if len(image_files) < 2:
        print("Error: Need at least 2 images for motion detection!")
        return
    
    if frame_idx2 >= len(image_files):
        print(f"Error: Frame index {frame_idx2} out of range. Only {len(image_files)} images available.")
        return
    
    # Load the two frames
    print(f"\nProcessing frames from {folder_name}:")
    print(f"  Frame 1: {image_files[frame_idx1].name}")
    print(f"  Frame 2: {image_files[frame_idx2].name}")
    
    original_image1, gray_image1 = preprocess_image(image_files[frame_idx1])
    original_image2, gray_image2 = preprocess_image(image_files[frame_idx2])
    
    if gray_image1 is None or gray_image2 is None:
        return
    
    # Step 2: Compute temporal derivative
    temporal_derivative = compute_temporal_derivative(gray_image1, gray_image2)
    
    # Step 3: Create motion mask
    thresh, motion_mask = create_motion_mask(temporal_derivative, threshold_value)
    
    # Step 4: Apply mask to original frame
    result = apply_mask_to_frame(original_image2, motion_mask)
    
    # Plotting results
    plt.figure(figsize=(20, 10))
    
    plt.suptitle(f"Motion Detection - {folder_name}", fontsize=16, y=0.98)
    
    plt.subplot(2, 3, 1)
    plt.title(f"Frame {frame_idx1}: {image_files[frame_idx1].name}")
    plt.imshow(cv2.cvtColor(original_image1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title(f"Frame {frame_idx2}: {image_files[frame_idx2].name}")
    plt.imshow(cv2.cvtColor(original_image2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Temporal Derivative (Difference)")
    plt.imshow(temporal_derivative, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Thresholded Difference')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('Motion Mask (0 and 1)')
    plt.imshow(motion_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title('Moving Objects Only')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_all_consecutive_frames_in_folder(image_files, threshold_value=30, folder_name=""):
    """
    Process all consecutive frame pairs in a folder
    """
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {folder_name}!")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)-1} frame pairs from {folder_name}...")
    print(f"{'='*60}\n")
    
    # Process each consecutive pair
    for i in range(len(image_files) - 1):
        print(f"\n--- Processing pair {i+1}/{len(image_files)-1} ---")
        process_consecutive_frames(image_files, i, i+1, threshold_value, folder_name)

def process_all_subdirectories(base_dir, subdirs, threshold_value=30, process_all_pairs=False):
    """
    Process images from all specified subdirectories
    """
    # Get all images organized by subdirectory
    all_images = get_all_images_from_subdirectories(base_dir, subdirs)
    
    print(f"\n{'#'*60}")
    print(f"MOTION DETECTION ACROSS MULTIPLE FOLDERS")
    print(f"{'#'*60}")
    
    # Process each subdirectory
    for subdir_name, image_files in all_images.items():
        print(f"\n\n{'*'*60}")
        print(f"FOLDER: {subdir_name}")
        print(f"{'*'*60}")
        
        if process_all_pairs:
            # Process all consecutive pairs in this folder
            process_all_consecutive_frames_in_folder(image_files, threshold_value, subdir_name)
        else:
            # Process only first pair (frames 0 and 1)
            if len(image_files) >= 2:
                process_consecutive_frames(image_files, 0, 1, threshold_value, subdir_name)



# base directory and subdirectories
INPUT_DIR = Path(r"C:\Users\Lover\OneDrive\Desktop\motion_detection\videos")

# List of subdirectories to process
subdirs = [
    "EnterExitCrossingPaths2cor/EnterExitCrossingPaths2cor",
    "Office/Office",
    "RedChair/RedChair"
]

# User selection
print("\n" + "="*60)
print("MOTION DETECTION ANALYSIS")
print("="*60)
print("\nAvailable folders:")
for idx, subdir in enumerate(subdirs, 1):
    folder_name = subdir.split('/')[0]
    print(f"  {idx}. {folder_name}")

while True:
    try:
        choice = int(input("\nSelect a folder (1-3): "))
        if 1 <= choice <= len(subdirs):
            selected_path = subdirs[choice - 1]
            selected_name = selected_path.split('/')[0]
            break
        else:
            print(f"Please enter a number between 1 and {len(subdirs)}")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Load images from selected folder
specific_folder = INPUT_DIR / selected_path
images = get_image_files_from_folder(specific_folder)

if images:
    print(f"\nProcessing ALL consecutive frame pairs from {selected_name}...")
    process_all_consecutive_frames_in_folder(images, threshold_value=30, folder_name=selected_name)