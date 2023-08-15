import pathlib
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .io import PathLike
from .patches import Patch

try:
    import torch
except ImportError as e:
    raise ImportError(
        "The torch (PyTorch) package is not installed but is required by the tiger.torch module"
    ) from e


class NeuralNetwork:
    """Base class for neural network wrappers"""

    def __init__(self, network: torch.nn.Module, device: Union[int, str], dtype: str):
        self.device = torch.device(device)
        self.dtype = np.dtype(dtype)
        self.network = network.to(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None

        if device == "cuda":
            self.n_devices = torch.cuda.device_count()
            if self.n_devices > 1:
                self.network = torch.nn.DataParallel(self.network)
        else:
            self.n_devices = 1

    def __len__(self) -> int:
        """Number of parameters in the network"""
        n_params = 0
        for param in self.network.parameters():
            n_params += np.prod(param.size())
        return n_params

    def __str__(self) -> str:
        return repr(self.network)

    def snapshot(self, filename: PathLike):
        """Store network weights and optimizer state in file"""
        # Make sure output directory exists
        filename = pathlib.Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Make sure to never store the parameters with "module." in front of the name even if DataParallel was used
        if self.n_devices == 1:
            network_state_dict = self.network.state_dict()
        else:
            network_state_dict = self.network.module.state_dict()  # type: ignore

        torch.save(
            {
                "network": network_state_dict,
                "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            },
            str(filename),
        )

    def restore(self, filename: PathLike):
        """Restore network weights and optionally the optimizer's state from a file"""
        state_dicts = torch.load(str(filename))
        network_state_dict = state_dicts["network"]
        try:
            optimizer_state_dict = state_dicts["optimizer"]
        except KeyError:
            optimizer_state_dict = None

        # If multiple devices are used, DataParallel requires all parameter names to begin with "module."
        if self.n_devices > 1:
            self.network.module.load_state_dict(network_state_dict)  # type: ignore
        else:
            self.network.load_state_dict(network_state_dict)

        # Set optimizer state dict only if both components are available
        if self.optimizer is not None and optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def _from_numpy(self, data: Iterable) -> torch.Tensor:
        """Convert numpy array to pytorch tensor on the correct device"""
        array = np.asarray(data, dtype=self.dtype)
        return torch.from_numpy(array).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> Union[float, np.ndarray]:
        """Download pytorch tensor from GPU and convert into float (single value) or numpy array (multiple values)"""
        array = tensor.to("cpu").detach().numpy()
        try:
            return array.item()
        except ValueError:
            return array


class _DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float):
        super().__init__()

        if abs(smooth) > 0:
            self.smooth = smooth
        else:
            raise ValueError("Smoothing factor needs to be non-zero to avoid numerical instability")


class SigmoidDiceLoss(_DiceLoss):
    """Dice-score loss for sigmoid outputs and binary targets

    Parameters
    ----------
    smooth
        Smoothness term, must be non-zero to ensure numerical stability
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__(smooth)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the Dice-score loss for a sigmoid output and a label mask

        Parameters
        ----------
        input
            Sigmoid output of shape (N, X, Y, Z)

        target
            Integer label  mask of shape (N, X, Y, Z)

        weight
            Optional masks that can be used to assign weights to parts of the image

        Returns
        -------
            Dice-score loss, defined as 1 - Dice
        """
        # Flatten tensors
        p = input.contiguous().view(-1)
        t = target.contiguous().view(-1)

        if weight is None:
            weight = 1  # type: ignore
        else:
            weight = weight.contiguous().view(-1)

        # Compute dice score
        intersection = torch.sum(p * t * weight)
        cardinality = torch.sum((p + t) * weight)
        dice_score = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        # Return dice loss
        return 1 - dice_score


class SoftmaxDiceLoss(_DiceLoss):
    """Dice-score loss for softmax outputs and integer targets

    Parameters
    ----------
    smooth
        Smoothness term, must be non-zero to ensure numerical stability

    mode
        One of "binary", "average" or "average_present_classes":

        - "binary" means that the Dice score is calculated across all classes by only considering correctly and
          incorrectly classified voxels.
        - "average" means that the average Dice scores per class are computed first. These scores are then averaged
          to obtain a single score.
        - "average_present_classes" means that the average Dice scores per class are computed first, but only for
          classes that are present in the groundtruth.
    """

    def __init__(self, smooth: float = 1e-6, mode: str = "binary"):
        super().__init__(smooth)

        if mode not in ("binary", "average", "average_present_classes"):
            raise ValueError(f'Unknown mode "{mode}"')
        self.mode = mode

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the Dice-score loss from a softmax output and a label mask

        Parameters
        ----------
        input
            Softmax output of shape (N, C, X, Y, Z)

        target
            Label mask, either a one-hot encoding with shape (N, C, X, Y, Z) or an integer label mask with shape
            (N, X, Y, Z).

        weight
            Optional masks that can be used to assign weights to parts of the image.

        Returns
        -------
            Dice-score loss, which is defined as 1 - Dice
        """
        n_classes = input.size(1)
        n_dims = input.ndimension()

        if weight is None:  # no mask = create completely positive mask
            weight = 1  # type: ignore

        # Convert class labels into a one hot encoding if necessary
        if input.shape == target.shape:
            targets_onehot = target
        else:
            eye = torch.eye(n_classes, dtype=input.dtype, device=input.device)
            dims = (0, n_dims - 1) + tuple(range(1, n_dims - 1))
            targets_onehot = eye[target.long().squeeze(1)].permute(*dims).contiguous()

        # Compute dice scores
        if self.mode == "binary":
            dims = tuple(range(0, n_dims))  # compute one score
        else:
            dims = (0,) + tuple(range(2, n_dims))  # keep classes separated

        intersection = torch.sum(input * targets_onehot * weight, dims)
        cardinality = torch.sum((input + targets_onehot) * (weight != 0), dims)
        dice_scores = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        # Return dice loss
        if self.mode == "binary":
            return 1 - dice_scores
        elif self.mode == "average":
            return 1 - dice_scores.mean()
        else:
            n_present_classes = torch.sum(targets_onehot.sum(dims) > 0, dtype=dice_scores.dtype)
            mean_dice_score = dice_scores.sum() / n_present_classes
            return 1 - mean_dice_score


class PolynomialLearningRateDecay(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        exponent: float = 0.9,
        last_epoch: int = -1,
    ):
        if not 0 < exponent <= 1:
            raise ValueError(
                f"Exponent should be larger than 0 and smaller or equal 1, got {exponent}"
            )

        super().__init__(optimizer, lambda epoch: (1 - epoch / max_epochs) ** exponent, last_epoch)


def collate_patches(patches: Iterable[Patch]) -> Tuple[torch.Tensor, List[Patch]]:
    """Can be used as collate_fn in a DataLoader when using e.g. SlidingCuboid"""
    array = np.stack([patch.array for patch in patches])
    tensor = torch.from_numpy(array[:, None, :, :, :])
    return tensor, list(patches)
