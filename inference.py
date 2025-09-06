import os
import torch
import argparse
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from scipy.special import softmax
from fvcore.common.config import CfgNode

from engine.loops import validate
from engine.checkpoints import load_trained_model
from evaluation.EchonetEvaluator import EchonetEvaluator
from datasets import datas, load_dataset
from utils.utils_files import to_numpy
from config.defaults import cfg_costum_setup, default_argument_parser,overwrite_eval_cfg

import pickle
import os
import matplotlib.pyplot as plt
import cv2  # Using OpenCV to read images, as it's used in the target script
from collections import defaultdict

view = 'A2C'
dataset_name = f'HMC_QU_{view}_48_multi_kp_snake_inference'
output_pickle_path = f'complete_HMC_QU/{view}/inference/48_multi_kp_snake/inference_output.pkl'
base_image_dir = f'complete_HMC_QU/{view}/preprocessed_kpts/frames'

#change this
weights_path = 'experiments/HMC_QU/A2C/logs/HMC_QU_A2C_48_multi_kp_snake/CNNGCN/mobilenet2/905125824/weights_HMC_QU_A2C_48_multi_kp_snake_CNNGCN_best_kptsErr.pth'


def get_subject_key_from_filename(filename):
    """
    Derives a subject key (e.g., 'ES00043_CH2_1.npy') from a frame filename
    (e.g., 'ES00043_CH2_1_5.png').
    It assumes the frame number is the last part after an underscore.
    """
    # Remove the file extension (.png)
    base_name = os.path.splitext(filename)[0]
    # Split by the last underscore to separate the frame number
    subject_part = base_name.rsplit('_', 1)[0]
    # Return the subject part with the desired .npy extension
    return f"{subject_part}.npy"


def create_inference_pickle(source_file, image_dir, output_file):
    """
    Loads data from the source pickle, processes it, and saves it in a new
    format compatible with the second script.
    """
    # --- 1. Load the source data ---
    try:
        with open(source_file, "rb") as f:
            source_data = pickle.load(f)
        print(f"Successfully loaded data from {source_file}")
        print(f"Found data for {len(source_data)} total frames.")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{source_file}'")
        return
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return

    # --- 2. Process and group the data ---
    # Use defaultdict to easily create lists for new subjects
    grouped_data = defaultdict(lambda: {'kpts_pred': [], 'imgs': []})

    print("Processing and grouping data...")
    for i, (key, value) in enumerate(source_data.items()):
        
        filename = value.get('data_path_from_root')
        if not filename:
            print(f"Warning: 'data_path_from_root' not found for key {key}. Skipping.")
            continue
            
        image_path = os.path.join(image_dir, filename)

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at '{image_path}'. Skipping.")
            continue

        # Load the image using OpenCV
        try:
            # cv2.imread loads the image as a NumPy array in BGR format
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image could not be read.")
            height, width, _ = img.shape
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            continue

        # Get predicted keypoints (normalized)
        predicted_kpts_normalized = value['keypoints_prediction']

        # Scale keypoints to image pixel dimensions
        pred_x = predicted_kpts_normalized[:, 0] * width
        pred_y = predicted_kpts_normalized[:, 1] * height
        
        # Combine back into a (N, 2) array
        predicted_kpts_pixels = np.vstack((pred_x, pred_y)).T

        # Get the subject key for grouping
        subject_key = get_subject_key_from_filename(filename)

        # Append the image and keypoints to the correct subject group
        grouped_data[subject_key]['imgs'].append(img)
        grouped_data[subject_key]['kpts_pred'].append(predicted_kpts_pixels)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(source_data)} frames...")

    print(f"Processing complete. Found {len(grouped_data)} unique subjects.")

    # --- 3. Save the new data structure to the output pickle file ---
    # Convert defaultdict back to a regular dict for saving
    output_data = dict(grouped_data)
    
    try:
        with open(output_file, "wb") as f:
            pickle.dump(output_data, f)
        print(f"Successfully created and saved the new inference file to '{output_file}'")
    except Exception as e:
        print(f"An error occurred while saving the new pickle file: {e}")


def sliding():
    mode = 'sliding_window'

########################################
########################################
# Main
########################################
########################################
def eval_trained_model(model: torch.nn.Module, cfg: CfgNode, ds: datas,
                       basedir: str, basename: str, device: torch.device, batch_size: int, num_workers: int, num_examples_to_plot: int):

    # Load model:
    model = model.to(device)
    model_name = cfg.MODEL.NAME

    # Get dataloaders
    testloader = torch.utils.data.DataLoader(ds.testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True
                                              )

    # # run test:
    test_losses, test_outputs, test_inputs = validate(mode='test',
                                                      epoch=1,
                                                      loader=testloader,
                                                      model=model,
                                                      device=device,
                                                      criterion=None)

    test_loss = test_losses["main"].avg
    dataset_info = dataset_name
    out_directory = os.path.join(basedir, "{}_eval_on_{}/".format(basename, dataset_info))

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    with open(os.path.join(out_directory,"eval_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file
    # frames_info_file = pd.read_csv(ds.testset.echonet_frame_info_csvfile, index_col=0)
    # frames_info_file = frames_info_file[frames_info_file.Split == "TEST"]

    evaluator = EchonetEvaluator(dataset=ds.testset, tasks=["ef"], output_dir=out_directory)
    evaluator.process(test_inputs, test_outputs)
    evaluator.evaluate()
    evaluator.plot(num_examples_to_plot=min(num_examples_to_plot, len(test_outputs)))

    print(" ** test loss: {}".format(test_loss))
    # compute_stats(total_filenames, total_output_guiding, total_gt_guiding, textfilename=textfilename)

def eval_sliding_window(model:torch.nn.Module, cfg: CfgNode, ds: datas, device: torch.device, basedir: str, basename: str,):
    g = ds.testset
    window_size = 16
    frame_step = 2
    predictions = dict()

    # !!!
    model.eval()

    for case_index in range(len(g)):
        prediction = dict()
        case_data = g.get_img_and_kpts(index=case_index)
        all_frames = case_data["img"]
        num_frames_in_case = all_frames.shape[-1]
        frame_size = all_frames.shape[:2]
        data_path_from_root = case_data["img_path"].replace(g.img_folder, "")
        prediction["ef_prediction"], prediction["sd_prediction"], prediction["keypoints_prediction"] = [], [], []
        prediction["data_path_from_root"] = data_path_from_root
        prediction["ef"] = case_data["ef"]
        prediction["sd"] = np.asarray([case_data["index_frame1"], case_data["index_frame2"]])

        #for ii in range(num_frames_in_case - window_size * frame_step):
        for ii in range(0, num_frames_in_case - window_size * frame_step, window_size * frame_step):
        #for ii in list(np.random.randint(num_frames_in_case - window_size * frame_step, size=10)):
            indices = list(range(ii, ii + window_size * frame_step, frame_step))
            #indices = list(range(16))
            img = all_frames[:, :, :, indices]
            # image norm:
            resized_img = torch.zeros([img.shape[2], g.input_size, g.input_size, img.shape[3]])
            for idx in range(img.shape[3]):
                img_slice = img[:, :, :, idx]
                img_slice = Image.fromarray(np.uint8(img_slice))
                img_slice = g.basic_transform(img_slice)
                resized_img[:, :, :, idx] = img_slice
            img = resized_img

            #img = [g.basic_transform(Image.fromarray(np.uint8(img[:, :, :, k]))) for k in range(window_size)]
            #img = torch.stack(img)
            #img = torch.reshape(img, (1, 3, window_size, frame_size[0], frame_size[1]))
            img = img.unsqueeze(dim=0)
            img = img.to(device)
            ef_pred, kpts_pred, sd_pred = model(img)
            to_numpy(img)
            prediction["ef_prediction"].append(g.denormalize_ef(to_numpy(ef_pred)[0][0]))
            # fix code duplication, taken from EchoNetEvaluator
            sd_pred = np.argmax(softmax(to_numpy(sd_pred)[0]), axis=0)  # convert to logits format, same as gt
            prediction["sd_prediction"].append(g.denormalize_sd(sd_pred))
            prediction["keypoints_prediction"].append(to_numpy(kpts_pred)[0])
            #

        prediction["ef_mean_prediction"] = np.mean(np.array(prediction["ef_prediction"]))
        prediction["mEFerr"] = np.abs(prediction["ef_mean_prediction"] - prediction["ef"])
        if case_index % 10 == 0:
            print("done running sliding window ({} iterations) for case {} [{}/{}].  Pred ef={}, ef={},   mEFerr={}".
                  format(num_frames_in_case - window_size, data_path_from_root, case_index, len(g),
                         prediction["ef_mean_prediction"], prediction["ef"], prediction["mEFerr"]))
        predictions[data_path_from_root] = prediction

    print("done eval")
    all_ef_predictions = np.asarray([prediction[1]["ef_mean_prediction"] for prediction in predictions.items()])
    all_ef = np.asarray([prediction[1]["ef"] for prediction in predictions.items()])
    total_mEFerr = np.mean(np.abs(all_ef - all_ef_predictions))
    print("total EF error for test set: {}".format(total_mEFerr))
    dataset_info = dataset_name
    out_directory = os.path.join(basedir, "{}_eval_on_{}/".format(basename, dataset_info))
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    with open(os.path.join(out_directory,"eval_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file

    evaluator = EchonetEvaluator(dataset=ds.testset, tasks=["ef"], output_dir=out_directory)
    fig = evaluator._plot_ef_scatters(ef=all_ef, ef_prediction=all_ef_predictions)
    file_path = os.path.join(out_directory, "scatter_plot_echonet_sliding_window_OVERLAP.png")
    fig.savefig(file_path)
    file_path = os.path.join(out_directory, "echonet_sliding_window_predictions_OVERLAP.npz")
    np.savez(file_path, predictions=predictions)
    print("predictions were saved to {}".format(file_path))



if __name__ == '__main__':
    args = default_argument_parser()
    cfg_eval = cfg_costum_setup(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_model, _ = load_trained_model(weights_filename=weights_path)
    cfg = overwrite_eval_cfg(cfg_model,cfg_eval)
    
    model = model.to(device)
    basedir = os.path.dirname(weights_path)
    basename = os.path.splitext(os.path.basename(weights_path))[0]

    if cfg.EVAL.MODE == 'normal':
        ds = load_dataset(ds_name=dataset_name, input_transform=None, input_size=cfg.EVAL.INPUT_SIZE, num_frames=cfg.NUM_FRAMES)
        eval_trained_model(model=model, cfg=cfg, ds=ds,
                           basedir=basedir,
                           basename=basename,
                           device=device,
                           batch_size=cfg.EVAL.BATCH_SIZE,
                           num_workers=cfg.EVAL.NUM_WORKERS,
                           num_examples_to_plot=cfg.EVAL.EXAMPLES_TO_PLOT
                           )
        out_directory = os.path.join(basedir, "{}_eval_on_{}/".format(basename, dataset_name))
        source_pickle_path = f'{out_directory}/echonet_predictions.pkl'

        create_inference_pickle(source_pickle_path, base_image_dir, output_pickle_path)


    elif cfg.EVAL.MODE == 'sliding_window':
        ds = load_dataset(ds_name="sliding_window", input_transform=None, input_size=cfg.EVAL.INPUT_SIZE, num_frames=cfg.NUM_FRAMES)
        #ds = load_dataset(ds_name="echonet_random", input_transform=None, input_size=train_params.input_size, num_frames=16)
        eval_sliding_window(model=model,cfg=cfg, ds=ds, device=device,
                            basedir=basedir,
                            basename=basename,
                            )


