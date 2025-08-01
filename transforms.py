import cv2
import numpy as np
import albumentations as A

class HorizontalFlipKeepIndexOrder(A.HorizontalFlip):
    """Flip the input horizontally but keep keypoint indices in the same order."""

    def apply_to_keypoints(self, keypoints, **params):
        # image width is required to compute flipped x-coordinates
        width = params.get("cols", 1)
        flipped_keypoints = []

        for keypoint in keypoints:
            x, y = keypoint[:2]
            flipped_x = width - x
            flipped = (flipped_x, y) + tuple(keypoint[2:])
            flipped_keypoints.append(flipped)

        # Reverse keypoints to keep consistent indexing order
        flipped_keypoints.reverse()
        return np.array(flipped_keypoints)

def load_transform(augmentation_type: str, augmentation_probability: float, input_size: int = 256, num_frames: int =1) ->  A.core.composition.Compose:

    crop_ratio = 1.0 #0.78125

    if augmentation_type == "strongkeep":
        input_transformer = A.Compose([
            HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),
                    # FIX: Changed height and width to the 'size' argument
                    A.RandomResizedCrop(size=(input_size, input_size), scale=(crop_ratio, crop_ratio), ratio=(1.0, 1.0), p=1.00),
                ],
                p=1.00
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                ],
                p=0.5
            ),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            additional_targets={'teacher_kpts': 'keypoints'},
            p=augmentation_probability)

    elif augmentation_type == "strongkeep_echo":
        input_img_size = 112
        input_transformer = A.Compose([
            HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),
                    # FIX: Changed height and width to the 'size' argument
                    A.RandomResizedCrop(size=(input_img_size, input_img_size), scale=(crop_ratio, crop_ratio), ratio=(1.0, 1.0), p=1.00),

                ],
                p=1.00
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                ],
                p=0.5
            ),

        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    elif augmentation_type == "strong_echo_cycle":
        input_img_size = 112
        additionaltargets = {}
        num_images = num_frames
        for i in range(1, num_images):
            additionaltargets = {**additionaltargets, f'image{i}': 'image'}
        additionaltargets['keypoints1']='keypoints'

        input_transformer = A.Compose([HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),
                    # FIX: Changed height and width to the 'size' argument
                    A.RandomResizedCrop(size=(input_img_size, input_img_size), scale=(crop_ratio, crop_ratio), ratio=(1.0, 1.0), p=1.00),
                ],
                p=1.00
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                ],
                p=0.5
            ),
        ],
            additional_targets=additionaltargets,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    else:
        raise NotImplementedError(f"Augmentation method '{augmentation_type}' is currently not implemented..")

    return input_transformer


if __name__ == '__main__':

    input_size = 224
    augmentation_type = "2chkeep"
    input_transform = None
    input_transform = load_transform(augmentation_type=augmentation_type, augmentation_probability=1.0, input_size=input_size)