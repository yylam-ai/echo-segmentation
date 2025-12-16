# EchonetEvaluator.py

import numpy as np
import copy
import cv2
import random
import logging
import os
import shutil
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from typing import Dict, List, Tuple
import pickle

from datasets import datas
from .BaseEvaluator import DatasetEvaluator
from utils.utils_plot import plot_grid, draw_kpts, plot_kpts_pred_and_gt
from utils.utils_stat import match_two_kpts_set

class EchonetEvaluator(DatasetEvaluator):
    """
    Evaluate EchoNet segmentation predictions for a single iteration of the cardiac navigation model
    """

    def __init__(
        self,
        dataset: datas,
        tasks: List = ["kpts", "ef"],
        output_dir: str = "./visu",
        verbose: bool = True
    ):
        """
        Args:
            dataset (dataset object): Note: used to be dataset_name: name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "single_iter", "multi_iter".
                By default, will infer this automatically from predictions.
            output_dir (str): optional, an output directory to dump results.
        """

        self._dataset = dataset
        self._verbose = verbose
        self._tasks = tasks
        self._output_dir = output_dir
        if self._verbose:
            self.set_logger(logname=os.path.join(output_dir, "eval_log.log"))
            self._logger = logging.getLogger(__name__)

        self._cpu_device = torch.device("cpu")
        self._do_evaluation = True  # todo: add option to evaluate without gt

    def reset(self):
        self._predictions = dict()

    def set_logger(self, logname):
        print("Evaluation log file is set to {}".format(logname))
        logging.basicConfig(filename=logname,
                            filemode='w', #'a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)    #level=logging.DEBUG)    # level=logging.INFO)


    def process(self, inputs: Dict, outputs: Dict) -> None:
        """
        Args:
            inputs: the inputs to a EF and Kpts model. It is a list of dicts. Each dict corresponds to an image and
                contains keys like "keypoints", "ef".
            outputs: the outputs of a EF and Kpts model. It is a list of dicts with keys
                such as "ef_prediction" or "keypoints_prediction" that contains the proposed ef measure or keypoints coordinates.
        """
        some_val_output_item = next(iter(outputs.items()))[1]
        tasks = []
        if some_val_output_item["keypoints_prediction"] is not None:
            tasks.append("kpts")
        if some_val_output_item["ef_prediction"] is not None:
            tasks.append("ef")
        if some_val_output_item["sd_prediction"] is not None:
            tasks.append("sd")
        self._tasks = tasks

        self._predictions = dict()
        for ii, data_path in enumerate(outputs):
            prediction = dict()

            # get predictions:
            if some_val_output_item["ef_prediction"] is not None:
                prediction["ef_prediction"] = self._dataset.denormalize_ef(outputs[data_path]["ef_prediction"])
                prediction["ef"] = self._dataset.denormalize_ef(inputs[data_path]["ef"])

            if some_val_output_item["keypoints_prediction"] is not None:
                prediction["keypoints_prediction"] = outputs[data_path]["keypoints_prediction"]
                prediction["keypoints"] = inputs[data_path]["keypoints"]

            if some_val_output_item["sd_prediction"] is not None:
                prediction["sd_prediction"] = outputs[data_path]["sd_prediction"]
                prediction["sd_prediction"] = np.argmax(softmax(prediction["sd_prediction"]), axis=0)   # convert to logits format, same as gt
                prediction["sd_prediction"] = self._dataset.denormalize_sd(prediction["sd_prediction"])

                prediction["sd"] = self._dataset.denormalize_sd(inputs[data_path]["sd"])


            # get case name:
            prediction["data_path_from_root"] = data_path.replace(self._dataset.img_folder, "")

            self._predictions[data_path] = prediction


    def evaluate(self, tasks: List = None):
        if tasks is not None:
            self._tasks = tasks

        predictions = self._predictions

        if len(predictions) == 0 and self._verbose:
            self._logger.warning("[EchonetEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir is not None:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "echonet_predictions.pkl")
            with open(file_path, 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not self._do_evaluation and self._verbose:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._verbose:
            self._logger.info("Evaluating predictions ...")
        self._results = OrderedDict()
        tasks = self._tasks
        for task in sorted(tasks):
            if self._verbose:
                self._logger.info("Preparing results in the EchoNet format for task {} ...".format(task))
            if task == "ef":
                res = self._eval_ejection_fraction_predictions(predictions)
            if task == "kpts":
                res = self._eval_keypoints_predictions(predictions)
            if task == "sd":
                res = self._eval_diastolic_systolic_predictions(predictions)

            self._results[task] = res
        
        # --- NEW: Save results to a text file ---
        self._save_results_to_txt()
        # -----------------------------------------

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def plot(self, num_examples_to_plot: int) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 16))
        plot_directory = os.path.join(self._output_dir, "plots")
        if os.path.exists(plot_directory):
            shutil.rmtree(plot_directory)
        os.makedirs(plot_directory)
        self._logger.info("plotting {} prediction examples to {}".format(num_examples_to_plot, plot_directory))
        for data_path in random.sample(list(self._predictions), num_examples_to_plot):
            prediction = self._predictions[data_path]
            fig.clf()
            if "ef" in self._tasks:
                keypoints_prediction = prediction["keypoints_prediction"] if "kpts" in self._tasks else None
                sd_prediction = prediction["sd_prediction"] if "sd" in self._tasks else None
                ax1 = fig.add_subplot(1, 1, 1)
                self._plot_EF_prediction(ax=ax1, data_path_from_root=prediction["data_path_from_root"],
                                         ef_prediction=prediction["ef_prediction"],
                                         keypoints_prediction=keypoints_prediction,
                                         sd_prediction=sd_prediction)
            else:
                fig = self._plot_kpts_single_frame(fig, data_path_from_root=prediction["data_path_from_root"],
                                                   keypoints_prediction=prediction["keypoints_prediction"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, plot_filename))

    def set_tasks(self, tasks: List) -> None:
        self._tasks = tasks

    def get_tasks(self) -> List:
        return self._tasks

    def _eval_ejection_fraction_predictions(self, predictions: Dict) -> Dict:
        if self._verbose:
            self._logger.info("Eval stats for Ejection Fraction")

        ef_prediction = np.stack([output[1]["ef_prediction"] for output in predictions.items()])
        ef = np.stack([output[1]["ef"] for output in predictions.items()])
        mEfERR = np.mean(abs(ef - ef_prediction))
        if self._verbose:
            self._logger.info("Mean ef error is {}".format(mEfERR))
        if self._output_dir is not None:
            fig = self._plot_ef_scatters(ef=ef, ef_prediction=ef_prediction)
            fig.suptitle("EF scatters for EchoNet test set, size={}".format(len(predictions)))
            fig.savefig(os.path.join(self._output_dir, "ef_scatters.jpg".format()))

        return {"mEfERR": mEfERR}

    def _compute_oks(self, gt_kpts, pred_kpts, area):
        """
        Compute Object Keypoint Similarity (OKS).
        OKS = exp(-d^2 / (2 * s^2 * k^2))
        where:
        - d is the Euclidean distance between the predicted keypoint and the corresponding ground truth
        - s is the object scale (sqrt of the object segment area)
        - k is a per-keypoint constant that controls falloff
        """
        # Using a constant k for all keypoints, a common practice when per-keypoint sigmas are not available.
        k = 0.08  # This value can be tuned based on dataset characteristics.
        
        distances = np.linalg.norm(gt_kpts - pred_kpts, axis=1)
        
        # OKS calculation
        oks = np.exp(-distances**2 / (2 * area * k**2))
        return np.mean(oks)

    def _eval_keypoints_predictions(self, predictions: Dict) -> Dict:
        """
        Evaluate keypoints predictions with mKptsERR, mAP (OKS-based), and PCKh@0.5.
        """
        if self._verbose:
            self._logger.info("Eval stats for keypoints")

        dist_pred_gt_kpts = []
        all_oks = []
        all_pck = []
        
        num_kpts = self._dataset.num_kpts
        num_annotated_frames = 2 if "ef" in self._tasks else 1

        for prediction in predictions.values():
            gt_kpts_all_frames = prediction["keypoints"].reshape(num_kpts, 2, num_annotated_frames)
            pred_kpts_all_frames = prediction["keypoints_prediction"].reshape(num_kpts, 2, num_annotated_frames)

            for i in range(num_annotated_frames):
                gt_kpts = gt_kpts_all_frames[:, :, i]
                pred_kpts = pred_kpts_all_frames[:, :, i]

                # Original metric
                dist_pred_gt_kpts.append(100 * match_two_kpts_set(gt_kpts, pred_kpts))

                # --- New Metrics Calculation ---
                # 1. Calculate scale for PCK and OKS
                x_min, y_min = gt_kpts.min(axis=0)
                x_max, y_max = gt_kpts.max(axis=0)
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                
                # Handle cases with no valid bounding box
                if bbox_w == 0 or bbox_h == 0:
                    # Cannot compute scale-dependent metrics, skip this frame for OKS/PCK
                    continue

                scale = np.sqrt(bbox_w * bbox_h)
                area = bbox_w * bbox_h

                # 2. PCKh@0.5
                threshold = 0.5 * scale
                distances = np.linalg.norm(gt_kpts - pred_kpts, axis=1)
                correct_kpts = (distances < threshold).sum()
                pck_for_frame = correct_kpts / num_kpts
                all_pck.append(pck_for_frame)

                # 3. OKS for mAP
                oks_for_frame = self._compute_oks(gt_kpts, pred_kpts, area)
                all_oks.append(oks_for_frame)

        # --- Aggregate Metrics ---
        mKptsERR = np.mean(np.stack(dist_pred_gt_kpts)) if dist_pred_gt_kpts else 0.0
        pckh_05 = np.mean(all_pck) if all_pck else 0.0

        # Calculate mAP@[.5:.95]
        oks_thresholds = np.linspace(0.5, 0.95, 10)
        aps = []
        for thres in oks_thresholds:
            # AP at a given threshold is the fraction of samples with OKS > threshold
            ap = np.mean([1 if oks >= thres else 0 for oks in all_oks])
            aps.append(ap)
        mAP = np.mean(aps) if aps else 0.0

        if self._verbose:
            self._logger.info(f"Mean keypoints error (mKptsERR): {mKptsERR:.4f}")
            self._logger.info(f"Percentage of Correct Keypoints (PCKh@0.5): {pckh_05:.4f}")
            self._logger.info(f"Mean Average Precision (mAP@[.5:.95]): {mAP:.4f}")

        return {"mKptsERR": mKptsERR, "PCKh@0.5": pckh_05, "mAP@[.5:.95]": mAP}

    def _eval_diastolic_systolic_predictions(self, predictions):
        if self._verbose:
            self._logger.info("Eval stats for Diastolic/Systolic Frame Detection")

        dist_pred_gt_SD = []
        for prediction in predictions.values():
            dist_pred_gt_SD.append(abs(prediction["sd"] - prediction["sd_prediction"]))
        mSD_ERR = np.mean(np.stack(dist_pred_gt_SD))
        if self._verbose:
            self._logger.info("Average Frame Distance is {}".format(mSD_ERR))

        return {"mSD_ERR": mSD_ERR}

    def _save_results_to_txt(self):
        """Saves all computed evaluation metrics to a text file."""
        file_path = os.path.join(self._output_dir, "evaluation_results.txt")
        with open(file_path, "w") as f:
            f.write("--- Evaluation Results ---\n\n")
            for task, results in self._results.items():
                f.write(f"Task: {task.upper()}\n")
                if isinstance(results, dict):
                    for metric_name, value in results.items():
                        f.write(f"  - {metric_name}: {value:.4f}\n")
                else:
                    # Fallback for single-value results
                    f.write(f"  - Result: {results:.4f}\n")
                f.write("\n")
        if self._verbose:
            self._logger.info(f"Evaluation results saved to {file_path}")

    def _plot_EF_prediction(self, ax, data_path_from_root, ef_prediction, keypoints_prediction=None, sd_prediction=None):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        img = data["img"]
        frames_inds = data["frames_inds"]
        ef = data["ef"]

        frames = [img[:, :, :, ii] for ii in range(self._dataset.num_frames)]

        if keypoints_prediction is not None:
            extrema_indices = [0, self._dataset.num_frames - 1]
            if sd_prediction is not None:
                extrema_indices = sd_prediction
            for ii, extermum_index in enumerate(extrema_indices):
                if extermum_index > -1:
                    thumbnail = img[:, :, :, extermum_index]
                    thumbnail_keypoints = self._dataset.denormalize_pose(keypoints_prediction[:, :, ii], thumbnail)
                    frames[extermum_index] = draw_kpts(thumbnail, thumbnail_keypoints,
                                                       kpts_connections=self._dataset.kpts_info["connections"],
                                                       colors_pts=self._dataset.kpts_info["colors"])

        seq_plot = plot_grid(frames=frames, labels=frames_inds, thumbnail_size=112)

        ax.imshow(np.array(seq_plot))
        prediction_text = "EF={:.1f}, EF_prediction={:.1f}, EF_L1={:.2f}".format(ef, ef_prediction, abs(ef - ef_prediction))
        if sd_prediction is not None:
            prediction_text = "{} ED={}, ED_prediction={}, ES={}, ES_prediction={}".format(prediction_text,
                                                                                           data["index_frame1"], sd_prediction[0],
                                                                                           data["index_frame2"], sd_prediction[1])
        ax.set_title(prediction_text)
        ax.axis('off')

        return ax

    def _plot_kpts_single_frame(self, fig, data_path_from_root, keypoints_prediction):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        
        img = data["img"]
        keypoints = data["kpts"]

        # --- START OF FIX ---
        # If image data is 4D (H, W, C, num_frames), select the first frame.
        if img.ndim == 4:
            img = img[:, :, :, 0]

        # If ground truth keypoints data is 3D (num_kpts, 2, num_frames), select the first frame.
        if keypoints.ndim == 3:
            keypoints = keypoints[:, :, 0]
            
        # Do the same for the predicted keypoints.
        if keypoints_prediction.ndim == 3:
            keypoints_prediction = keypoints_prediction[:, :, 0]
        # --- END OF FIX ---
        
        keypoints = self._dataset.normalize_pose(keypoints, img)
        img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        keypoints_prediction = self._dataset.denormalize_pose(keypoints_prediction, img)
        keypoints = self._dataset.denormalize_pose(keypoints, img)

        plot_kpts_pred_and_gt(fig, img, gt_kpts=keypoints, pred_kpts=keypoints_prediction,
                              kpts_info=self._dataset.kpts_info, closed_contour=self._dataset.kpts_info['closed_contour'])

        return fig
    
    def _plot_ef_scatters(self, ef: np.ndarray, ef_prediction: np.ndarray) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(18, 18))
        metrics = ['%', '%']
        labels = ['ef', 'ef prediction']

        for rr in range(1):
            for cc in range(1):
                L1 = abs(ef - ef_prediction)
                L1_mean, L1_std = np.mean(L1), np.std(L1)
                axs[rr, cc].plot(ef, ef_prediction, marker='.', linestyle='None', color='black', markersize=2.5)
                axs[rr, cc].set(xlabel=labels[0], ylabel=labels[1])
                axs[rr, cc].set_title("EF[{}] vs. EF Prediction. L1: Mean={}, Std={}".format(metrics[0], L1_mean, L1_std))
                axs[rr, cc].set_aspect('equal', adjustable='box')
                axs[rr, cc].plot([0, 1], [0, 1], color='red', ls="--", transform=axs[rr, cc].transAxes)

        return fig