import numpy as np
import pytest
import torch

import tiger.torch


class ApproxTorchDecorator:
    """Enable floating point comparison for PyTorch tensors using pytest's approx function"""

    # Tell numpy to use our `__eq__` operator instead of its.
    __array_ufunc__ = None
    __array_priority__ = 100

    def __init__(self, expected):
        self.expected = pytest.approx(self._detensorify(expected), rel=0, abs=0.0001)

    @staticmethod
    def _detensorify(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            else:
                return value.detach().numpy()
        else:
            return value

    def __eq__(self, other):
        return self._detensorify(other) == self.expected


def approx(expected):
    return ApproxTorchDecorator(expected)


class DeterministicNetwork(torch.nn.Module):
    def __init__(self, init_zero=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=2, out_features=3, bias=True)

        with torch.no_grad():
            if init_zero:
                self.linear.weight.zero_()
                self.linear.bias.zero_()
            else:
                self.linear.weight.copy_(torch.Tensor([[1, 2], [3, 4], [5, 6]]))
                self.linear.bias.copy_(torch.Tensor([1, 2, 3]))

    def forward(self, incoming):
        return self.linear(incoming)


@pytest.fixture
def nn():
    return tiger.torch.NeuralNetwork(DeterministicNetwork(), device="cpu", dtype="float32")


@pytest.fixture
def snapshot_file(tmp_path):
    file = tmp_path / "snapshot.pkl"
    yield file

    try:
        file.unlink()
    except FileNotFoundError:
        pass


def test_nn_parameter_count(nn):
    """Length of NeuralNetwork instance should be number of trainable parameters"""
    assert len(nn) == 6 + 3


def test_nn_numpy_conversion(nn):
    """Test conversion from numpy array to Torch tensor and vice versa"""
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    tensor = nn._from_numpy(image)
    assert isinstance(tensor, torch.Tensor)

    image2 = nn._to_numpy(tensor)
    assert image2 == approx(image)


def test_nn_save_restore(nn, snapshot_file):
    """Test saving and loading of network parameters"""
    input_tensor = torch.Tensor([1, 2])
    output_tensor = nn.network(input_tensor)
    nn.snapshot(snapshot_file)

    # Lets create a new network instance, initialized with all zeros
    nn2 = tiger.torch.NeuralNetwork(
        DeterministicNetwork(init_zero=True), device="cpu", dtype="float32"
    )
    output_tensor2 = nn2.network(input_tensor)
    assert output_tensor2 == approx(np.array([0, 0, 0]))
    assert output_tensor2 != approx(output_tensor)

    # Restore parameters and check if the output is now the same
    nn2.restore(snapshot_file)
    output_tensor2 = nn2.network(input_tensor)
    assert output_tensor2 == approx(output_tensor)


def test_sigmoid_dice_loss():
    loss = tiger.torch.SigmoidDiceLoss()

    # Everything predicted as background, no foreground structure present, loss should be 0
    probabilities = torch.zeros([1, 1, 2, 2])
    targets = torch.zeros([1, 2, 2])
    assert loss(probabilities, targets) == approx(0)

    # ... everything is foreground, loss should be 1
    targets = torch.ones([1, 2, 2])
    assert loss(probabilities, targets) == approx(1)

    # ... some foreground, loss should be 1 since nothing of the foreground was detected
    targets = torch.FloatTensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert loss(probabilities, targets) == approx(1)

    # Everything predicted as foreground, no foreground structure present, loss should be 1
    probabilities = torch.ones([1, 1, 2, 2])
    targets = torch.zeros([1, 2, 2])
    assert loss(probabilities, targets) == approx(1)

    # ... everything is foreground, loss should be 0
    targets = torch.ones([1, 2, 2])
    assert loss(probabilities, targets) == approx(0)

    # ... some foreground, loss should have specific value (1 - Dice)
    targets = torch.FloatTensor([[[0.0, 0.0], [1.0, 0.0]]])
    expected_value = 1 - (2 * 1) / (4 + 1)
    assert loss(probabilities, targets) == approx(expected_value)

    # Test masks
    masks_1 = torch.ones((1, 2, 2))  # batch size 1, 2 x 2 mask

    # completely positive mask (all ones) should give the same result as no mask
    assert loss(probabilities, targets, masks_1) == approx(loss(probabilities, targets))

    targets = torch.FloatTensor([[[1.0, 0.0], [1.0, 0.0]]])
    masks_2 = torch.FloatTensor([[[1.0, 1.0], [0.0, 0.0]]])
    expected_value = 1 - (2 * 1) / (2 + 1)
    assert loss(probabilities, targets, masks_2) == approx(expected_value)


def test_softmax_dice_loss():
    probabilities = torch.zeros([1, 3, 2, 2])
    targets = torch.zeros([1, 2, 2])

    loss_binary = tiger.torch.SoftmaxDiceLoss()
    loss_average = tiger.torch.SoftmaxDiceLoss(mode="average")
    loss_average_present = tiger.torch.SoftmaxDiceLoss(mode="average_present_classes")

    # Default mode should be binary
    assert loss_binary.mode == "binary"

    # Everything predicted as background, no foreground structures present, loss should be 0
    probabilities[:, 0, :, :] = 1
    assert loss_binary(probabilities, targets) == approx(0)

    # ... everything is class 1, so all pixels are incorrectly labeled, loss should be 1
    targets[:, :, :] = 1
    assert loss_binary(probabilities, targets) == approx(1)

    # ... some foreground, some background (p: 1,0,0,2 / t: 0,2,0,2)
    probabilities = torch.zeros([1, 3, 2, 2])
    probabilities[:, 1, 0, 0] = 1
    probabilities[:, 0, 0, 1] = 1
    probabilities[:, 0, 1, 0] = 1
    probabilities[:, 2, 1, 1] = 1

    targets = torch.zeros([1, 2, 2])
    targets[:, :, 1] = 2

    # Binary loss ignores classes, only cares about TP, FP and FN
    # | X n Y | = 2, | X | = 4, | Y | = 4  =>  2 * 2 / (4 + 4) = 1/2  =>  loss = 1 - 1/2 = 1/2
    assert loss_binary(probabilities, targets) == approx(1 / 2)

    # Average per class is slightly different
    # 0: 1 TP, 1 FP, 1 FN => (2 * 1) / (2 * 1 + 1 + 1) = 1/2
    # 1: 0 TP, 1 FP, 0 FN => 0 / 1 = 0
    # 2: 1 TP, 0 FP, 1 FN => (2 * 1) / (1 * 1 + 0 + 1) = 2/3
    # Average: 1/3 * (1/2 + 2/3 + 0) = 7/18  =>  loss = 1 - 7/18 = 11/18
    assert loss_average(probabilities, targets) == approx(11 / 18)

    # Average per present class ignores class 1, which is not present in the targets
    # 0: 1/2, 2: 2/3  => average: 1/2 * (1/2 + 2/3) = 7/12  =>  loss = 1 - 7/12 = 5/12
    assert loss_average_present(probabilities, targets) == approx(5 / 12)

    # Test masking loss function
    masks_1 = torch.ones((1, 2, 2))  # batch size 1, 2 x 2 mask

    # completely positive mask (all ones) should give the same result as no mask in all modes
    assert loss_binary(probabilities, targets) == approx(
        loss_binary(probabilities, targets, masks_1)
    )
    assert loss_average(probabilities, targets) == approx(
        loss_average(probabilities, targets, masks_1)
    )
    assert loss_average_present(probabilities, targets) == approx(
        loss_average_present(probabilities, targets, masks_1)
    )

    # Don't account for the top-right value in the 2x2 prediction
    masks_2 = torch.ones((1, 2, 2))
    masks_2[:, 0, 1] = 0  # [[1, 0], [1, 1]]

    # Binary with mask
    # | X n Y | = 2, | X | = 3, | Y | = 3  =>  2 * 2 / (3 + 3) = 2/3  =>  loss = 1 - 2/3 = 1/3
    assert loss_binary(probabilities, targets, masks_2) == approx(1 / 3)

    # Average per class with mask
    # 0: 1 TP, 0 FP, 1 FN => (2 * 1) / (2 * 1 + 0 + 1) = 2/3
    # 1: 0 TP, 1 FP, 0 FN => 0 / 1 = 0
    # 2: 1 TP, 0 FP, 0 FN => (2 * 1) / (2 * 1 + 0 + 0) = 1
    # Average: 1/3 * (2/3 + 0 + 1) = 5/9  =>  loss = 1 - 5/9 = 4/9
    assert loss_average(probabilities, targets, masks_2) == approx(4 / 9)

    # Average per present class with mask
    # 0: 2/3, 2: 1  => average: 1/2 * (2/3 + 1) = 5/6  =>  loss = 1 - 5/6 = 1/6
    assert loss_average_present(probabilities, targets, masks_2) == approx(1 / 6)
