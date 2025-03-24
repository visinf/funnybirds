import os
import re
import cv2
import torch
import torch.utils.data
import numpy as np

from .helpers import find_high_activation_crop


class ppnetexplain:
    def __init__(self, model):
        self.model = model
        self.ppnet = model.model
        self.ppnet_multi = torch.nn.DataParallel(self.ppnet)
        self.img_size = self.ppnet_multi.module.img_size

    # It follows the interface of image and target, just like in part_importances.py at 200
    def attribute(self, input, target: int):
        """Returns a generator yielding prototypes with
        in order,  their bounding boxes, activaiton maps, max_activation value and conn_score"""

        prototype_shape = self.ppnet.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        _, min_distances = self.ppnet_multi(input)
        idx = 0
        distances = self.ppnet.push_forward(input)[1]
        prototype_activations = self.ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = self.ppnet.distance_2_similarity(distances)
        if self.ppnet.prototype_activation_function == "linear":
            prototype_activations = prototype_activations + max_dist
            prototype_activation_patterns = prototype_activation_patterns + max_dist
        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])

        prototype_info = np.load(
            os.path.join(
                self.model.load_model_dir, "img",
                "epoch-" + self.model.epoch_number_str,
                "bb" + self.model.epoch_number_str + ".npy",
            )
        )
        prototype_img_identity = prototype_info[:, -1]

        def PrototypeGenerator():
            for i in range(1, len(sorted_indices_act)):
                prototype_class = prototype_img_identity[sorted_indices_act[-i].item()]
                if target == prototype_class:
                    conn_score = self.ppnet.last_layer.weight[target[0]][
                        sorted_indices_act[-i].item()
                    ]

                    max_activation = array_act[-i]

                    activation_pattern = (
                        prototype_activation_patterns[idx][
                            sorted_indices_act[-i].item()
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    upsampled_activation_pattern = cv2.resize(
                        activation_pattern,
                        dsize=(self.img_size, self.img_size),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    activation_map = torch.from_numpy(upsampled_activation_pattern)

                    bbox = find_high_activation_crop(upsampled_activation_pattern)

                    yield (bbox, activation_map, max_activation, conn_score)

        return PrototypeGenerator()