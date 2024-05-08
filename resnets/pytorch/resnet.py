import math
from torch import nn
from torch.nn import init

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_batchnorm(self) -> None:
    if self.track_running_stats:
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
    if self.affine:
        init.ones_(self.weight)
        init.zeros_(self.bias)


def _init_kaiming_5(self):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with uniform(-1/sqrt(k), 1/sqrt(k)).
    # For details, see github.com/pytorch/pytorch/issues/57109, github.com/pytorch/pytorch/issues/15314
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)


INITIALIZATION = dict(
    Conv2d=_init_kaiming_5,
    Linear=_init_kaiming_5,
    BatchNorm2d=_init_batchnorm,
)


def override_initialization_method(cls, init_method):
    assert isinstance(cls, type)
    cls_name = cls.__name__

    INITIALIZATION[cls_name] = init_method


def initialize_all_submodules(module: nn.Module):
    for submodule in module.modules():
        submodule_type = type(submodule).__name__
        if submodule_type in INITIALIZATION:
            logging.info(f"Using {INITIALIZATION[submodule_type].__name__} to initialize {submodule}")
            INITIALIZATION[submodule_type](submodule)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        bootleneck,
    ):
        """Versatile building block of ResNet neural networks, using BN-ReLU-Conv2D pattern."""
        super(ResNetBlock, self).__init__()

        self.bootleneck = bootleneck

        if self.bootleneck:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

            self.bn3 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

        else:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # First bn and relu are common for both branches
        x = self.bn1(x)
        x = nn.functional.relu(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)

        if self.bootleneck:
            out = self.bn3(out)
            out = nn.functional.relu(out)
            out = self.conv3(out)

        out += self.shortcut(x)
        return out


class ResNetStarter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling):
        super(ResNetStarter, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            padding="same",
        )

        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=pooling, padding=1)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNetHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNetHead, self).__init__()

        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetFactory(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        block_channels,
        block_rescale,
        bootleneck,
        starter_kernel_size,
        starter_channels,
        starter_stride,
        starter_pooling,
    ):
        super(ResNetFactory, self).__init__()

        logging.info(f"Adding starter with {in_channels} -> {starter_channels} channels")
        self.starter = ResNetStarter(
            in_channels,
            starter_channels,
            kernel_size=starter_kernel_size,
            stride=starter_stride,
            pooling=starter_pooling,
        )

        self.blocks = []
        in_channels = starter_channels
        in_scale = starter_stride * starter_pooling
        for out_channels, out_scale in zip(block_channels, block_rescale):
            assert out_scale % in_scale == 0, "Stride values must be divisible by the previous stride"
            stride = out_scale // in_scale
            in_scale = out_scale

            logging.info(f"Adding block with {in_channels} -> {out_channels} channels and stride {stride}")
            block = ResNetBlock(in_channels, out_channels, stride=stride, bootleneck=bootleneck)
            self.register_module(f"block_{len(self.blocks)}", block)
            self.blocks.append(block)
            in_channels = out_channels

        logging.info(f"Adding head with {in_channels} -> {n_classes} channels")
        self.head = ResNetHead(in_channels, n_classes)
        initialize_all_submodules(self)

    def forward(self, x):
        x = self.starter(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


def cifar_resnet(preset):
    if preset == 20:
        return ResNetFactory(
            in_channels=3,
            n_classes=10,
            block_channels=[16] * 3 + [32] * 3 + [64] * 3,
            block_rescale=[2] * 3 + [4] * 3 + [8] * 3,
            bootleneck=False,
            starter_kernel_size=3,
            starter_channels=16,
            starter_stride=1,
            starter_pooling=1,
        )
    elif preset == 32:
        return ResNetFactory(
            in_channels=3,
            n_classes=10,
            block_channels=[16] * 5 + [32] * 5 + [64] * 5,
            block_rescale=[2] * 5 + [4] * 5 + [8] * 5,
            bootleneck=False,
            starter_kernel_size=3,
            starter_channels=16,
            starter_stride=1,
            starter_pooling=1,
        )
    elif preset == 44:
        return ResNetFactory(
            in_channels=3,
            n_classes=10,
            block_channels=[16] * 7 + [32] * 7 + [64] * 7,
            block_rescale=[2] * 7 + [4] * 7 + [8] * 7,
            bootleneck=False,
            starter_kernel_size=3,
            starter_channels=16,
            starter_stride=1,
            starter_pooling=1,
        )
    elif preset == 56:
        return ResNetFactory(
            in_channels=3,
            n_classes=10,
            block_channels=[16] * 9 + [32] * 9 + [64] * 9,
            block_rescale=[2] * 9 + [4] * 9 + [8] * 9,
            bootleneck=False,
            starter_kernel_size=3,
            starter_channels=16,
            starter_stride=1,
            starter_pooling=1,
        )
    elif preset == 110:
        return ResNetFactory(
            in_channels=3,
            n_classes=10,
            block_channels=[16] * 18 + [32] * 18 + [64] * 18,
            block_rescale=[2] * 18 + [4] * 18 + [8] * 18,
            bootleneck=False,
            starter_kernel_size=3,
            starter_channels=16,
            starter_stride=1,
            starter_pooling=1,
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")
