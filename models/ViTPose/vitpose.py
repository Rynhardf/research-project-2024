from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
import torch


class ViTPose(nn.Module):
    def __init__(
        self,
        num_joints=17,
        img_size=(256, 192),
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.3,
        decoder="simple",
    ):
        super(ViTPose, self).__init__()

        # Load the ViT backbone from timm
        self.backbone = VisionTransformer(
            img_size=img_size,  # Set the image size here
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = self.backbone.embed_dim

        self.img_size = img_size

        # Define the keypoint head
        if decoder == "classic":
            self.keypoint_head = ClassicDecoder(
                in_channels=self.embed_dim,
                out_channels=num_joints,
            )
        else:
            self.keypoint_head = SimpleDecoder(
                in_channels=self.embed_dim,
                out_channels=num_joints,
            )

    def forward(self, x):
        x = self.backbone.forward_features(x)

        x = x[:, 1:, :]

        # convert 192 to 16x12
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], x.shape[1], 16, 12)
        # Forward pass through the keypoint head
        x = self.keypoint_head(x)
        return x

    def init_weights(self, weights):
        if "backbone.last_norm.weight" in weights:
            weights["backbone.norm.weight"] = weights["backbone.last_norm.weight"]
            weights["backbone.norm.bias"] = weights["backbone.last_norm.bias"]
            del weights["backbone.last_norm.weight"]
            del weights["backbone.last_norm.bias"]

        # Get the model's state dict
        model_state_dict = self.state_dict()

        # Filter out weights with size mismatch
        filtered_weights = {}
        for k in weights.keys():
            if k in model_state_dict and weights[k].size() == model_state_dict[k].size():
                filtered_weights[k] = weights[k]
            else:
                print(f"Skipping key '{k}' due to size mismatch or because it does not exist in the model.")

        # Load the filtered weights
        r = self.load_state_dict(filtered_weights, strict=False)

        if r.missing_keys:
            print("Missing keys: ", r.missing_keys)
            # Initialize missing parameters
            for key in r.missing_keys:
                module = self
                attrs = key.split('.')
                for attr_name in attrs[:-1]:
                    if hasattr(module, attr_name):
                        module = getattr(module, attr_name)
                    else:
                        module = None
                        break
                if module is None:
                    continue
                param_name = attrs[-1]
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    if isinstance(param, nn.Parameter):
                        # Initialize weights and biases appropriately
                        if isinstance(module, nn.Conv2d):
                            if param_name == 'weight':
                                nn.init.normal_(param, std=0.001)
                            elif param_name == 'bias':
                                nn.init.constant_(param, 0)
                        elif isinstance(module, nn.Linear):
                            if param_name == 'weight':
                                nn.init.normal_(param, std=0.001)
                            elif param_name == 'bias':
                                nn.init.constant_(param, 0)
                        else:
                            # Initialize other layer types if needed
                            pass



vit_models = {
    "S": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 12,
        "drop_path_rate": 0.1,
    },
    "B": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "drop_path_rate": 0.3,
    },
    "L": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "drop_path_rate": 0.5,
    },
    "H": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "drop_path_rate": 0.55,
    },
}


def get_vitpose_model(variant="B", num_joints=17, decoder="simple"):
    model = ViTPose(
        num_joints=num_joints,
        img_size=(256, 192),
        embed_dim=vit_models[variant]["embed_dim"],
        depth=vit_models[variant]["depth"],
        num_heads=vit_models[variant]["num_heads"],
        drop_path_rate=vit_models[variant]["drop_path_rate"],
        decoder=decoder,
    )
    return model


# Define the Keypoint Head
class ClassicDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Simple decoder with two deconvolution blocks, each followed by batch normalization
        and ReLU, then a 1x1 convolution to get the keypoint heatmaps.

        Args:
        - in_channels (int): Number of input channels (from the ViT or backbone).
        - out_channels (int): Number of output channels (typically the number of keypoints or heatmaps).
        """
        super(ClassicDecoder, self).__init__()

        internal_channels = 256
        # First deconvolution block (upsample by 2)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=internal_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # Second deconvolution block (upsample by 2)
            nn.ConvTranspose2d(
                in_channels=internal_channels,
                out_channels=internal_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )

        # Final 1x1 convolution to output keypoint heatmaps
        self.final_layer = nn.Conv2d(
            in_channels=internal_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # Apply deconvolution layers
        x = self.deconv_layers(x)

        # Apply final 1x1 convolution to get keypoint heatmaps
        x = self.final_layer(x)

        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Simple decoder block with bilinear upsampling, ReLU, and a 3x3 convolution.

        Args:
        - in_channels (int): Number of input channels (typically the output from the ViT backbone).
        - out_channels (int): Number of output channels (typically the number of keypoints or heatmaps).
        """
        super(SimpleDecoder, self).__init__()

        # Bilinear upsampling by a factor of 4
        self.upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Final 3x3 convolution to get the heatmaps
        self.final_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # Bilinear upsampling
        x = self.upsample(x)

        # ReLU activation
        x = self.relu(x)

        # 3x3 convolution to produce the heatmaps
        x = self.final_layer(x)

        return x
