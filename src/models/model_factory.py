"""
Lightweight model factory: MobileNetV3-Large and EfficientNet-B0
(ImageNet-pretrained backbones from torchvision, classifier head replaced
for whatever num_classes the configured dataset has).

Also provides a two-stream LATE-FUSION model for the `fused` training
variant (original + bbox_cropped): each stream keeps its own pretrained
backbone + pooled feature vector; the two feature vectors are concatenated
and passed through a single new classifier head. This is deliberately NOT
early/pixel fusion (channel-stacking the two images) - the two crops aren't
spatially aligned (one is the full frame, one is a sub-region), so there's
no pixel-level correspondence for early conv filters to exploit. Late
fusion lets each stream learn its own representation first.

Also provides an EARLY-FUSION model for the `early_fused` training variant
(original + bbox_cropped): the two images are channel-stacked into one
(6, H, W) input (see EarlyFusedWasteDataset in dataset_loader.py) and fed
through a SINGLE pretrained backbone whose first conv layer has been
"inflated" from 3 to 6 input channels (pretrained 3-channel filters are
duplicated across the extra 3 channels and rescaled, so the very first
epoch still sees sensible activation statistics instead of a randomly
initialized first layer). This is the cheaper, single-backbone alternative
to `fused` - one set of conv filters learns to combine both images'
signal from the earliest layer, at the cost of assuming some pixel-level
correspondence between the two streams.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_name: {model_name}. Expected 'mobilenet_v3' or 'efficientnet_b0'.")


def build_backbone(model_name: str):
    """Returns (backbone, feature_dim). backbone(x) -> (B, feature_dim) pooled
    feature vector, using the full pretrained trunk (conv features + pooling +
    all classifier layers EXCEPT the final logits layer) - so each fusion
    stream still benefits from the same pretrained representation the
    single-stream models use, not just raw conv features."""
    model_name = model_name.lower()

    if model_name == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        feat_dim = m.classifier[-1].in_features   # 1280 (after hidden Linear+Hardswish+Dropout)
        m.classifier = m.classifier[:-1]           # drop only the final Linear
        return m, feat_dim

    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = m.classifier[-1].in_features   # 1280 (after avgpool, just Dropout left)
        m.classifier = m.classifier[:-1]           # drop only the final Linear
        return m, feat_dim

    raise ValueError(f"Unknown model_name: {model_name}. Expected 'mobilenet_v3' or 'efficientnet_b0'.")


class FusedTwoStreamModel(nn.Module):
    """Two-stream late-fusion model for the `fused` variant.

    Expects input of shape (B, 2, C, H, W): stream 0 = `original` image,
    stream 1 = `bbox_cropped` image (this stacking is produced by
    FusedWasteDataset in dataset_loader.py). Each stream has its own
    pretrained backbone; pooled features from both are concatenated and fed
    through one new classifier head trained from scratch.
    """

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.stream_original, feat_dim_a = build_backbone(model_name)
        self.stream_bbox, feat_dim_b = build_backbone(model_name)
        self.classifier = nn.Linear(feat_dim_a + feat_dim_b, num_classes)

    def forward(self, x):
        # x: (B, 2, C, H, W)
        feat_original = self.stream_original(x[:, 0])
        feat_bbox = self.stream_bbox(x[:, 1])
        fused = torch.cat([feat_original, feat_bbox], dim=1)
        return self.classifier(fused)


def build_fused_model(model_name: str, num_classes: int):
    return FusedTwoStreamModel(model_name, num_classes)


def _inflate_first_conv(conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    """Returns a new Conv2d with `new_in_channels` input channels, initialized
    by tiling the pretrained weight's input-channel slices and rescaling so
    the summed activation magnitude stays roughly the same as the original
    3-channel conv (weights are divided by new_in_channels/old_in_channels).
    This is the standard trick for adapting an ImageNet-pretrained first
    layer to extra input channels (e.g. RGB -> RGB+RGB, RGB -> RGBD) without
    throwing away the pretrained filters."""
    old_weight = conv.weight.data  # (out_c, in_c, kh, kw)
    out_c, in_c, kh, kw = old_weight.shape

    reps = new_in_channels // in_c
    remainder = new_in_channels % in_c
    new_weight = old_weight.repeat(1, reps, 1, 1)
    if remainder:
        new_weight = torch.cat([new_weight, old_weight[:, :remainder]], dim=1)
    new_weight = new_weight * (in_c / new_in_channels)  # rescale to preserve activation stats

    new_conv = nn.Conv2d(
        new_in_channels, out_c, kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = new_weight
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def build_early_fused_model(model_name: str, num_classes: int):
    """Single-backbone early-fusion model for the `early_fused` variant.
    Builds the normal pretrained classifier (via build_model) then inflates
    its first conv layer to accept 6 input channels. Expects input of shape
    (B, 6, H, W): channels 0-2 = `original`, channels 3-5 = `bbox_cropped`
    (produced by EarlyFusedWasteDataset in dataset_loader.py)."""
    model = build_model(model_name, num_classes)
    model_name = model_name.lower()

    if model_name in ("mobilenet_v3", "efficientnet_b0"):
        old_conv = model.features[0][0]
        if not isinstance(old_conv, nn.Conv2d):
            raise RuntimeError(
                f"Expected model.features[0][0] to be the first Conv2d for '{model_name}', "
                f"got {type(old_conv)}. torchvision's internal layout may have changed."
            )
        model.features[0][0] = _inflate_first_conv(old_conv, new_in_channels=6)
        return model

    raise ValueError(f"Unknown model_name: {model_name}. Expected 'mobilenet_v3' or 'efficientnet_b0'.")
