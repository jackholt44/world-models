import torch


class Nets(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_size = 32
        self.encoder = self._build_encoder()
        self.encoder_target = self._build_encoder()
        self._bootstrap_target_encoder()

        self.predictor = self._build_predictor()
        self.probe = self._build_probe()
        self.probe_t_minus_1 = self._build_probe()

    def load_nets_checkpoint(self, checkpoint: dict) -> None:
        self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.encoder_target.load_state_dict(checkpoint["encoder_target"], strict=True)
        self.predictor.load_state_dict(checkpoint["predictor"], strict=True)

    def _bootstrap_target_encoder(self) -> None:
        """Bootstrap the target encoder to the same weights as the encoder."""
        # load the weights of the encoder into the target encoder
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # freeze the target encoder
        for parameter in self.encoder_target.parameters():
            parameter.requires_grad = False

    def _build_encoder(self) -> torch.nn.Module:
        """
        Build the encoder module.
        The encoder is a convolutional neural network that takes in a 6 channel input (2 frames of 3 channels)
        and outputs a 32 channel feature map.
        The network accepts two frames of 3 channels each. Each image is 128 pixels.

        Returns:
            torch.nn.Module, the encoder module
        """
        return torch.nn.Sequential(
            # Input: (N, 6, H, W) — two RGB frames stacked on the channel axis
            # (e.g. [obs_tminus1, obs_t]). H/W come from the data (128), not
            # from this layer; Conv2d only fixes channel count.
            #
            # Conv2d(6 -> 32): 32 filters, each a 3x3x6 kernel shared across
            # space. At every pixel, mix the local 3x3 neighborhood across all
            # 6 channels into one activation. Output: (N, 32, H, W) with
            # stride=1, padding=1 (spatial size unchanged).
            torch.nn.Conv2d(6, self.feature_size, kernel_size=3, stride=1, padding=1),
            # Zero negative pre-activations; only positive filter responses pass on.
            torch.nn.ReLU(),
            # (N, 32, H, W) -> (N, 32, H/2, W/2); for 128x128 -> 64x64.
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # refine the features with a second convolutional layer
            torch.nn.Conv2d(
                self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            # (N, 32, H, W) -> (N, 32, H/2, W/2); for 64x64 -> 32x32.
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            # (N, 32, H, W) -> (N, 32, H/2, W/2); for 32x32 -> 16x16.
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            # (N, 32, H, W) -> (N, 32, H/2, W/2); for 16x16 -> 8x8.
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (N, 32, 16, 16) -> (N, 32 * 16 * 16). Spatial layout becomes a flat vector.
            torch.nn.Flatten(),
            # Project the 32 * 16 * 16 pooled features into the 32-D latent z.
            # This is the actual squash from "32 maps × 16×16 cells" to 32 features.
            torch.nn.Linear(32 * 8 * 8, self.feature_size),
            # This prevents the deeper encoder’s latent scale—and consequently the EMA
            # target—from exploding.
            # With LayerNorm, initial total loss should be roughly single-digit, not rapidly
            # grow from 26 to 100,000.
            torch.nn.LayerNorm(self.feature_size),
        )

    # the NN responsible for predicting the next latent representation (z_hat)
    # it has 32 input features: 32 from the encoder and 2 from the action
    # it has 2 layers: 34 -> 32 -> 32
    # The first layer receives the concatenated encoder output and action.
    # The second layer predicts the next latent representation.
    # the output of the second layer is the next latent representation
    # we have two linear layers because we want to add a non-linearity between the input and output
    def _build_predictor(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(self.feature_size + 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.feature_size),
        )

    def _build_probe(self) -> torch.nn.Module:
        """Build the probe module.
        Responsible for checking the encoder is producing usable latent representations.
        Returns:
            torch.nn.Module, the probe module
        """
        return torch.nn.Sequential(
            # 32 -> 32 underfitted and hit a position ceiling of RMSE of 9px so bumping to 256
            torch.nn.Linear(self.feature_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),  # output the 4D vector of x, y, vx, vy
        )
