import torch

from wm.nets.inputs import frame_pair, frames_to_float
from wm.nets.jepa import Nets
from wm.train.dataset import RolloutBatch, RolloutBatches

LEARNING_RATE = 1e-4
TARGET_EMA = 0.99
# because we use layernorm, forcing the variance to be close to 1
# put the loss and the layer norm in tension - leading the layernorm gamma
# to be inflated to 1.3 over training. this led to a drift in absolute latent size
# which made it harder to analyse training runs
VARIANCE_HINGE_GAMMA = 0.85
# forward passes run in bfloat16 (about 1.5x faster on MPS); losses and statistics are
# still computed in float32. bfloat16 keeps float32's exponent range, so unlike float16
# it needs no gradient scaling
AUTOCAST_DTYPE = torch.bfloat16


class RolloutTrainer:
    def __init__(self, device: torch.device, nets: Nets):
        self.device = device
        self.nets = nets

        self.optimizer = torch.optim.Adam(
            list(self.nets.encoder.parameters()) + list(self.nets.predictor.parameters()),
            lr=LEARNING_RATE,
        )

    def to(self, device: torch.device):
        self.device = device
        self.nets.to(device)

    def autocast(self) -> torch.autocast:
        return torch.autocast(device_type=self.device.type, dtype=AUTOCAST_DTYPE)

    def to_device(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = frames_to_float(batch.frames, self.device)
        # (N, H + 2, 3, 128, 128) -> (N, H, 6, 128, 128): online pairs are frames (k-1, k),
        # target pairs are frames (k, k+1)
        obs_online = frame_pair(frames[:, :-2], frames[:, 1:-1])
        obs_target = frame_pair(frames[:, 1:-1], frames[:, 2:])
        action = batch.action.to(self.device, dtype=torch.float32)
        return obs_online, obs_target, action

    def test_epoch(self, loader: RolloutBatches) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad(), self.autocast():
            mse_losses = torch.zeros(loader.horizon, device=self.device, dtype=torch.float32)
            r2_scores = torch.zeros(loader.horizon, device=self.device, dtype=torch.float32)
            copy_z_mse_losses = torch.zeros(loader.horizon, device=self.device, dtype=torch.float32)
            z_hat_variances = torch.zeros(loader.horizon, device=self.device, dtype=torch.float32)
            z_target_variances = torch.zeros(
                loader.horizon, device=self.device, dtype=torch.float32
            )

            for batch in loader:
                obs_online, obs_target, action = self.to_device(batch)

                z = None
                z_0_target = None

                for step in range(obs_online.shape[1]):
                    online = obs_online[:, step, :, :, :]
                    target = obs_target[:, step, :, :, :]
                    a = action[:, step, :]

                    if step == 0:
                        z = self.nets.encoder(online)
                        z_0_target = self.nets.encoder_target(online).float()

                    z_hat = self.nets.predictor(torch.cat([z, a], dim=1))
                    z = z_hat
                    # statistics in float32, not the autocast dtype
                    z_hat = z_hat.float()
                    z_target = self.nets.encoder_target(target).float()
                    mse_loss = torch.nn.functional.mse_loss(z_hat, z_target)  # (N,32) -> ()
                    target_var = torch.var(z_target, dim=0).mean()  # (N,32) -> ()
                    r2 = 1 - (mse_loss / target_var)  # ()
                    copy_z_mse = torch.nn.functional.mse_loss(z_0_target, z_target)  # (N,32) -> ()
                    z_hat_var = torch.var(z_hat, dim=0).mean()  # (N,32) -> ()
                    z_target_var = torch.var(z_target, dim=0).mean()  # (N,32) -> ()

                    mse_losses[step] += mse_loss
                    r2_scores[step] += r2
                    copy_z_mse_losses[step] += copy_z_mse
                    z_hat_variances[step] += z_hat_var
                    z_target_variances[step] += z_target_var

            mse_losses = mse_losses / len(loader)
            r2_scores = r2_scores / len(loader)
            copy_z_mse_losses = copy_z_mse_losses / len(loader)
            z_hat_variances = z_hat_variances / len(loader)
            z_target_variances = z_target_variances / len(loader)

            return (
                mse_losses,  # (horizon,)
                r2_scores,  # (horizon,)
                copy_z_mse_losses,  # (horizon,)
                z_hat_variances,  # (horizon,)
                z_target_variances,  # (horizon,)
            )

    def train_epoch(
        self, loader: RolloutBatches
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        epoch_loss = []
        epoch_variance_loss = []
        epoch_grad_norm = []
        for batch in loader:
            # the batch is built on CPU so we need to transfer it to the device
            obs_online, obs_target, action = self.to_device(batch)

            batch_variance_loss = None
            batch_loss = torch.zeros(obs_online.shape[1], device=self.device)
            z = None

            self.optimizer.zero_grad()
            # autocast covers the forward pass only; backward reuses the forward dtypes
            with self.autocast():
                for step in range(obs_online.shape[1]):
                    online = obs_online[:, step, :, :, :]
                    target = obs_target[:, step, :, :, :]
                    a = action[:, step, :]
                    z_hat, variance_loss, prediction_loss = self.forward(online, target, a, z, step)
                    if variance_loss is not None:
                        batch_variance_loss = variance_loss
                        epoch_variance_loss.append(variance_loss.detach())
                    batch_loss[step] = prediction_loss
                    z = z_hat

            # this gives variance and prediction equal weight
            total_batch_loss = batch_variance_loss + batch_loss.mean()
            epoch_loss.append(batch_loss.detach())
            total_batch_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.nets.encoder.parameters()) + list(self.nets.predictor.parameters()),
                max_norm=1.0,
            )
            epoch_grad_norm.append(grad_norm.detach())
            self.optimizer.step()
            self.update_target_encoder()

        return (
            torch.stack(epoch_loss).mean(dim=0),  # (N, horizon) -> (horizon,)
            torch.stack(epoch_variance_loss).mean(),  # (N) -> ()
            torch.stack(epoch_grad_norm).mean(),  # (N) -> ()
        )

    def forward(
        self,
        obs_online: torch.Tensor,
        obs_target: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor | None,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variance_loss = None
        # we only compute the variance loss for the first step
        # as we want to apply pressure to the encoder to force apart the latent representations
        # of the online and target frames at the start of the sequence
        if step == 0:
            z = self.nets.encoder(obs_online)
            variance_loss = self.variance_loss(z.float())

        z_hat = self.nets.predictor(torch.cat([z, action], dim=1))
        z_target = self.nets.encoder_target(obs_target)
        prediction_loss = torch.nn.functional.mse_loss(z_hat.float(), z_target.float())

        return z_hat, variance_loss, prediction_loss

    def update_target_encoder(self):
        for online_param, target_param in zip(
            self.nets.encoder.parameters(), self.nets.encoder_target.parameters()
        ):
            target_param.data.mul_(TARGET_EMA).add_(online_param.data, alpha=1 - TARGET_EMA)

    def load_optimizer_checkpoint(self, checkpoint: dict) -> None:
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.std(z, dim=0)
        # low std = low variance = high loss
        relu = torch.nn.functional.relu(VARIANCE_HINGE_GAMMA - std)
        # average the loss over the batch
        return torch.mean(relu)
