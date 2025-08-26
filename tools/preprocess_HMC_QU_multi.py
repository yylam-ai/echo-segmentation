import os
import sys
import argparse
import pandas as pd
import imageio
import cv2
import numpy as np

"""
Preprocessing script for ultrasound video data focusing on keypoint extraction.

This script loops over a dataset to create the following files:
- Individual frames as PNG images.
- NPZ files containing keypoint annotations for each frame.
- Full video cycles as NUMPY arrays.
- NPZ files containing keypoints for annotated frames within a cycle.

It is adapted from a script that handled both masks and keypoints, but this version
is streamlined to handle ONLY keypoints.

**ASSUMPTION**: The keypoint coordinates in 'VolumeTracings.csv' are already
normalized or scaled for the target image size (e.g., 224x224), not for the
original video dimensions. This script will resize video frames to the target
size and save the keypoints as-is.

Expected Input CSV format:
- FileList.csv: Must contain 'ECHO' (filename), and 'split' (TRAIN/VAL/TEST).
                Can optionally contain 'EF', 'ESV', 'EDV'.
- VolumeTracings.csv: Must contain 'ECHO' (filename), 'frame', and a series of
                      x,y coordinate columns (e.g., 'x1', 'y1', 'x2', 'y2', ...).

Output structure:
- preprocessed_kpts/
    - frames/
        - FILENAME_FRAMENUM.png
    - annotations/
        - FILENAME_FRAMENUM.npz (contains 'kpts' array)
    - cycle/
        - frames/
            - FILENAME.npy (full video)
        - annotations/
            - FILENAME.npz (contains 'kpts' for annotated frames)
    - filenames/
        - {view}_train_filenames.txt
        - ...
"""

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ultrasound videos for keypoint extraction.")
    parser.add_argument('-v', '--view', type=str, default='A4C')
    parser.add_argument('--annotation_suffix', type=str, default='48')
    parser.add_argument('-i', '--input_dir', type=str, default='complete_HMC_QU',
                        help="Path to the directory containing 'FileList.csv', 'VolumeTracings.csv', and 'Videos/' folder.")
    parser.add_argument('-o', '--output_dir', type=str,
                        help="Path to the directory where preprocessed files will be saved. Defaults to '[input_dir]/preprocessed_kpts/'.")
    parser.add_argument('--img_size', type=int, default=224, help="Target width and height for resized frames.")
    parser.add_argument('--save_kpts', action='store_true', default=True,
                        help="Save keypoint .npz files (default: True).")
    parser.add_argument('--no-save_kpts', dest='save_kpts', action='store_false',
                        help="Do not save keypoint .npz files.")
    parser.add_argument('--save_imgs', action='store_true', default=True,
                        help="Save individual frame .png images (default: True).")
    parser.add_argument('--no-save_imgs', dest='save_imgs', action='store_false',
                        help="Do not save individual frame .png images.")
    
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.view)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'preprocessed_kpts')

    return args


def loadvideo(filename: str) -> np.ndarray:
    """
    Loads a video from a file.
    Args:
        filename (str): Full path to the video file.
    Returns:
        A np.ndarray with dimensions (frames, height, width, channels=3).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Video file not found: {filename}")
    
    try:
        capture = cv2.VideoCapture(filename)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                # It's common for frame count to be off by one, so we'll just break
                v = v[:count]
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count, ...] = frame
        
        capture.release()
        return v
    except Exception as e:
        raise ValueError(f"An error occurred while reading video {filename}: {e}")


def preprocess_data(input_path, output_path, save_kpts, save_imgs, view, img_size, annotation_suffix):
    """Main preprocessing function."""
    
    # Define target image size
    TARGET_H = img_size
    TARGET_W = img_size
    
    # Validate input paths
    volume_tracings_path = os.path.join(input_path, f'VolumeTracings_{annotation_suffix}.csv')
    file_list_path = os.path.join(input_path, 'FileList.csv')
    
    print("volume_tracings_path:", volume_tracings_path)
    if not os.path.exists(volume_tracings_path):
        raise FileNotFoundError(f"Required file not found: {volume_tracings_path}")
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"Required file not found: {file_list_path}")

    # Load metadata
    tracings_df = pd.read_csv(volume_tracings_path)
    file_list_df = pd.read_csv(file_list_path)

    # Standardize column names for robustness
    tracings_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    file_list_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    
    # Check for essential columns
    if 'echo' not in file_list_df.columns or 'split' not in file_list_df.columns:
        raise ValueError("'FileList.csv' must contain 'ECHO' and 'split' columns.")
    if 'echo' not in tracings_df.columns or 'frame' not in tracings_df.columns:
        raise ValueError("'VolumeTracings.csv' must contain 'ECHO' and 'frame' columns.")

    fnames = file_list_df["echo"].tolist()
    fnames = [f"{fn}.avi" if '.' not in fn else fn for fn in fnames]

    # Get train/val/test splits
    x_train = set(file_list_df[file_list_df["split"] == 'TRAIN']["echo"].tolist())
    x_val = set(file_list_df[file_list_df["split"] == 'VAL']["echo"].tolist())
    x_test = set(file_list_df[file_list_df["split"] == 'TEST']["echo"].tolist())

    # Prepare lists for output filenames
    output_list_train, output_list_val, output_list_test, output_list_invalid = [], [], [], []
    output_list_train_cycle, output_list_val_cycle, output_list_test_cycle = [], [], []

    # Create output directories
    frames_folder = os.path.join(output_path, "frames")
    anno_folder = os.path.join(output_path, f"annotations_{annotation_suffix}")
    frames_cycle_folder = os.path.join(output_path, "cycle/frames")
    anno_cycle_folder = os.path.join(output_path, f"cycle/annotations_{annotation_suffix}")
    file_dir = os.path.join(output_path, 'filenames')

    for folder in [frames_folder, anno_folder, frames_cycle_folder, anno_cycle_folder, file_dir]:
        os.makedirs(folder, exist_ok=True)

    for i, fname in enumerate(fnames):
        name = os.path.splitext(fname)[0]
        print(f"Processing [{i+1}/{len(fnames)}]: {name}")

        video_path = os.path.join(input_path, "Videos", fname)
        if not os.path.exists(video_path):
            print(f"  -> Video not found, skipping: {video_path}")
            output_list_invalid.append(name)
            continue

        # Get patient-specific data
        patient_tracings = tracings_df[tracings_df.echo == name]
        if patient_tracings.empty:
            print(f"  -> No tracings found for {name}, skipping.")
            output_list_invalid.append(name)
            continue
        
        # Try to get optional metadata
        try:
            patient_meta = file_list_df[file_list_df.echo == name].iloc[0]
            ef = patient_meta.get('ef')
        except (IndexError, KeyError):
            ef = None

        video = loadvideo(video_path)
        
        if video.shape[0] == 0:
            print(f"  -> Video is empty, skipping: {video_path}")
            output_list_invalid.append(name)
            continue
        
        # NOTE: Keypoint scaling is removed. We assume the keypoints in the CSV
        # are already in the coordinate space of the target image size (TARGET_W, TARGET_H).
        
        pts_cycle = []
        annotated_frames = []

        # Loop through unique annotated frames for this video
        for frame_num in sorted(patient_tracings.frame.unique()):
            frame_num = int(frame_num)
            frame_df = patient_tracings[patient_tracings.frame == frame_num]
            
            # Extract all coordinate columns (anything not 'echo' or 'frame')
            coord_cols = [c for c in frame_df.columns if c not in ['echo', 'frame']]
            coords = frame_df[coord_cols].values.flatten()
            
            # Ensure we have pairs of coordinates
            if len(coords) % 2 != 0:
                print(f"  -> Warning: Odd number of coordinates for {name} at frame {frame_num}. Skipping frame.")
                output_list_invalid.append(name)
                continue
            
            # Reshape into (N, 2) keypoints array
            keypoints = coords.reshape(-1, 2)
            
            # --- FIX: Swap x and y coordinates ---
            # The user reported that x and y are inverted. This line swaps the two columns.
            # If the CSV is ordered (x1, y1, x2, y2, ...), this changes the keypoints
            # from shape (N, [x, y]) to (N, [y, x]), which aligns with (row, col) indexing.
            keypoints = keypoints[:, [1, 0]]
            
            # Add to lists for train/val/test splits
            output_filename = f"{name}_{frame_num}.png"
            if name in x_train:
                output_list_train.append(output_filename)
            elif name in x_val:
                output_list_val.append(output_filename)
            elif name in x_test:
                output_list_test.append(output_filename)

            # Save individual frame and annotations
            if frame_num < len(video):
                frame_img = video[frame_num]
                
                # Resize the frame to the target size
                resized_frame = cv2.resize(frame_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                
                if save_imgs:
                    # Save the resized frame
                    imageio.imsave(os.path.join(frames_folder, f"{name}_{frame_num}.png"), resized_frame)
                if save_kpts:
                    # Save the keypoints without scaling, as they are assumed to match the resized frame
                    save_dict = {'kpts': keypoints}
                    if ef is not None:
                        save_dict['ef'] = ef
                    np.savez(os.path.join(anno_folder, f"{name}_{frame_num}"), **save_dict)
                
                pts_cycle.append(keypoints)
                annotated_frames.append(frame_num)
            else:
                print(f"  -> Warning: Frame number {frame_num} is out of bounds for video {name} with {len(video)} frames.")
                output_list_invalid.append(name)

        # Save full cycle data if annotations were found
        if pts_cycle:
            # Create a resized version of the full video
            num_frames = video.shape[0]
            resized_video = np.zeros((num_frames, TARGET_H, TARGET_W, 3), dtype=np.uint8)
            for f_idx in range(num_frames):
                resized_video[f_idx] = cv2.resize(video[f_idx], (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            
            # Save the resized video
            np.save(os.path.join(frames_cycle_folder, f"{name}.npy"), resized_video)
            
            save_dict = {'kpts': np.array(pts_cycle, dtype=object), 'fnum': np.array(annotated_frames)}
            if ef is not None:
                save_dict['ef'] = ef
            np.savez(os.path.join(anno_cycle_folder, f"{name}.npz"), **save_dict)

            cycle_filename = f"{name}.png" # Keep .png extension for consistency with original script
            if name in x_train:
                output_list_train_cycle.append(cycle_filename)
            if name in x_val:
                output_list_val_cycle.append(cycle_filename)
            if name in x_test:
                output_list_test_cycle.append(cycle_filename)

    # Write filename lists to disk
    print("\nWriting filename lists...")
    unique_invalid = set(output_list_invalid)
    
    def write_list_to_file(filepath, data_list, is_cycle=False):
        with open(filepath, 'w') as f:
            for item in sorted(list(set(data_list))):
                # For single frames, check if the base name is invalid
                base_name = item.split('_')[0] if not is_cycle else os.path.splitext(item)[0]
                if base_name not in unique_invalid:
                    f.write(item + '\n')

    write_list_to_file(os.path.join(file_dir, f'{view}_train_filenames.txt'), output_list_train)
    write_list_to_file(os.path.join(file_dir, f'{view}_val_filenames.txt'), output_list_val)
    write_list_to_file(os.path.join(file_dir, f'{view}_test_filenames.txt'), output_list_test)
    
    write_list_to_file(os.path.join(file_dir, f'{view}_cycle_train_filenames.txt'), output_list_train_cycle, is_cycle=True)
    write_list_to_file(os.path.join(file_dir, f'{view}_cycle_val_filenames.txt'), output_list_val_cycle, is_cycle=True)
    write_list_to_file(os.path.join(file_dir, f'{view}_cycle_test_filenames.txt'), output_list_test_cycle, is_cycle=True)

    with open(os.path.join(file_dir, f'{view}_invalid_filenames.txt'), 'w') as f:
        for name in sorted(list(unique_invalid)):
            f.write(name + '\n')
            
    print("Preprocessing complete.")


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise ValueError('Input directory does not exist.')
    
    os.makedirs(args.output_dir, exist_ok=True)

    preprocess_data(input_path=args.input_dir,
                    output_path=args.output_dir,
                    save_kpts=args.save_kpts,
                    save_imgs=args.save_imgs,
                    view=args.view,
                    img_size=args.img_size,
                    annotation_suffix=args.annotation_suffix)