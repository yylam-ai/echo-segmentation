import os
import numpy as np
import pickle
from utils.utils_plot import plot_inference_movie
import torch
from typing import List, Dict, Tuple
import torchvision.transforms as transforms
from PIL import Image


from engine.checkpoints import load_trained_model
from utils.utils_data import load_sequence_as_npy, load_image_as_npy, transform_image_sequence_to_tensor
from config.defaults import cfg_costum_setup, default_argument_parser,overwrite_eval_cfg


########################################### model ##############################################
def load_model_from_weights(weight_file:str = None) -> torch.nn.Module:
    model, _, _ = load_trained_model(weight_file, load_dataset_from_checkpoint=False)
    model.eval()
    return model

def run_inference(model:torch.nn.Module, device:torch.device, output_directory:str, data:np.ndarray = None, name:str ='sample') -> Dict:

    data = [data]
    if model.output_type == 'seq2ef':
        outputs = seq2ef(model, data, device)
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]

    elif model.output_type == 'img2kpts':
        outputs = img2kpts(model, data, device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()
        outputs['imgs'] = outputs['imgs'].cpu().detach().numpy()

    elif model.output_type == 'seq2ef&kpts':
        outputs = seq2ef_kpts(model,data,device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()[0]
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]

    elif model.output_type == 'seq2ef&kpts&sd':
        outputs = seq2ef_kpts_sd(model, data, device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()[0]
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]
        outputs['sd_pred'] = outputs['sd_pred'].cpu().detach().numpy()[0]
    else:
        raise NotImplementedError("Forward method to model type {} is not supported..".format(model.output_type))

    anim = plot_inference_movie(outputs['imgs'],outputs['kpts_pred'],input_size=512,metric_name='Name',value = name)
    output_filname = name+".gif"
    out_directory = output_directory
    gifname = os.path.join(out_directory, output_filname)
    anim.save(gifname, writer='imagemagick', fps=10)
    return outputs


def seq2ef(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    ef_pred = torch.squeeze(model(imgs), 1)
    outputs = {"ef_pred": ef_pred, "imgs": imgs}

    return outputs

def img2kpts(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    # print('images shape:', imgs.shape)
    model.to(device)
    kpts_pred = model(imgs)
    # print("kpts_pred shape:", kpts_pred.shape)
    outputs = {"kpts_pred": kpts_pred, "imgs": imgs}

    return outputs

def seq2ef_kpts(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs= data[0].to(device)
    ef_pred, kpts_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))
    outputs = {"kpts_pred": kpts_pred, "ef_pred": ef_pred, "imgs": imgs}

    return outputs

def seq2ef_kpts_sd(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    ef_pred, kpts_pred, sd_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))
    outputs = {"kpts_pred": kpts_pred, "ef_pred": ef_pred, "sd_pred": sd_pred, "imgs": imgs}

    return outputs

########################################### load ##############################################

def get_filenames_from_folder(image_folder:str) -> List:
    image_list = []
    for (dirpath,dirnames,filenames) in os.walk(image_folder):
        image_list =[dirpath+name for name in filenames]
    return image_list

if __name__ == '__main__':
    args = default_argument_parser()
    cfg_eval = cfg_costum_setup(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    weights = cfg_eval.INF.WEIGHTS
    mode = cfg_eval.INF.MODE
    output_directory =cfg_eval.INF.OUTPUT
    input = cfg_eval.INF.INPUT
    input_dir = cfg_eval.INF.INPUT_DIR
    model = load_model_from_weights(weights)

    if input_dir:
        inference_output = {}
        file_list = os.listdir(input_dir)
        input_size = 224

        # 2. Create the same transformation pipeline as used in eval.py
        #    This typically includes resizing and converting to a tensor.
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # Add transforms.Normalize if your model was trained with it.
        ])

        for i, f in enumerate(file_list):
            print(f"Inferencing: {i+1}/{len(file_list)}")
            npz_file = os.path.join(input_dir, f)
            
            # Assuming the .npz file contains a single array of images (N, H, W, C)
            # If the key is named, use data['arr_0'] or the correct key.
            original_imgs_np = np.load(npz_file)

            # 3. Apply the transformation to each frame individually
            transformed_frames = []
            for frame_idx in range(original_imgs_np.shape[0]):
                # Convert numpy frame to PIL Image
                frame_np = original_imgs_np[frame_idx]
                # Handle single-channel (grayscale) vs 3-channel images
                if frame_np.shape[2] == 1:
                    frame_np = frame_np.squeeze(axis=2) # (H, W, 1) -> (H, W)
                
                frame_pil = Image.fromarray(frame_np)
                
                # Apply the transform
                transformed_frame = transform(frame_pil)
                transformed_frames.append(transformed_frame)
            
            # Stack frames into a single tensor and add a batch dimension
            data_tensor = torch.stack(transformed_frames).unsqueeze(0) # Shape: (1, N, C, H, W)
            # The model likely expects (N, C, H, W), so let's adjust if needed.
            # Based on your `img2kpts` function, it seems to process a batch of images, not a sequence.
            # Let's reshape to (NumFrames, C, H, W)
            data_tensor = torch.stack(transformed_frames).to(device) # Shape: (N, C, H, W)

            inference_output[f] = {}

            # Run inference on the correctly preprocessed tensor
            outputs = img2kpts(model, [data_tensor], device)
            kpts_norm = outputs['kpts_pred'].cpu().detach().numpy()
            
            # The 'imgs' from the output are the transformed tensors, not the originals
            imgs_tensor_transformed = outputs['imgs'].cpu().detach().numpy()

            # 4. Scale normalized coords by the MODEL'S INPUT SIZE, not the original image size
            kpts_scaled = kpts_norm.copy()
            kpts_scaled[..., 0] *= input_size   # x
            kpts_scaled[..., 1] *= input_size   # y
            
            # Store the original images and the scaled keypoints
            inference_output[f]['imgs'] = original_imgs_np # Store original for visualization
            inference_output[f]['kpts_pred'] = kpts_scaled

        # --- END: MODIFIED CODE ---

        os.makedirs(output_directory, exist_ok=True)

        with open(f"{output_directory}/inference_output.pkl", "wb") as f:
            pickle.dump(inference_output, f)
    else:
        if 'single' in mode:
            file = input
            if not os.path.exists(file) & os.path.isfile(file):
                raise FileNotFoundError(file)

            if mode == 'single_image':
                image = load_image_as_npy(file)
                tensor = transform_image_sequence_to_tensor(image,device)

            elif mode == 'single_sequence':
                sequence = load_sequence_as_npy(file)
                tensor = transform_image_sequence_to_tensor(sequence,device)
            else:
                raise NotImplementedError("Mode {} is not supported..".format(mode))

            run_inference(model,device,output_directory,tensor, file.split('/')[-1][:-4])

        elif 'folder' in mode:
            if not os.path.isdir(input):
                raise 'Path is not a directory'

            image_files = get_filenames_from_folder(input)
            for file in image_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(file)
                if mode == 'folder_image':
                    image = load_image_as_npy(file)
                    tensor = transform_image_sequence_to_tensor(image,device)

                elif mode == 'folder_sequence':
                    sequence = load_sequence_as_npy(file)
                    if len(sequence[0]) > 100:
                        sequence = sequence[:100]
                    tensor = transform_image_sequence_to_tensor(sequence,device)
                else:
                    raise NotImplementedError("Mode {} is not supported..".format(mode))

                print(file.split('/')[-1][:-4])
                run_inference(model,device,output_directory,tensor, file.split('/')[-1][:-4])

        else:
            raise NotImplementedError("Mode {} is not supported..".format(mode))
