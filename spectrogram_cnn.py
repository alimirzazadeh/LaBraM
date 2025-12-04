import numpy as np
import torch 
import h5py
import os 
from tqdm import tqdm
import random
from pdb import set_trace as bp
from torchaudio.transforms import Spectrogram
import torchaudio
import torchvision.transforms as transforms
import torch
from scipy.signal.windows import dpss
import torch.nn as nn
import torch.nn.functional as F
import pickle
from mne.time_frequency import psd_array_multitaper, psd_array_welch
import time

""" 
To do: reorder the channels to make sense for the conv2d model 
add metrics (same as paper) and create training script 
compare and modify the model architectures 

"""

def conv3x3(in_planes, out_planes, stride=(1, 1)):
    """3x3 conv with padding, stride can be (sh, sw)."""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=(1, 1)):
    """1x1 conv for skip/projection."""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    """
    Input:  x of shape [B, 23, 160, 6]  (C=23, H=160, W=6)

    Pipeline:
      1) Conv1d (kernel=7, stride=2, padding=3) along height
      2) MaxPool1d (kernel=3, stride=2, padding=1) along height
         -> H: 160 -> 80 -> 40, W: 6 unchanged
      3) 4 ResNet-18-style 2D layers, downsampling both H and W:
         - layer1: stride (1, 2)   : H=40, W=6 -> 40 x 3
         - layer2: stride (2, 2)   : H=40->20, W=3->2
         - layer3: stride (2, 2)   : H=20->10, W=2->1
         - layer4: stride (2, 1)   : H=10->5,  W=1
      4) Output features: [B, 512, 5, 1]
    """

    def __init__(self, num_classes=None, dataset='TUAB'):
        super().__init__()
        if dataset == 'TUAB':
            self.data_length = 10
        elif dataset == 'TUEV':
            self.data_length = 5
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        # -------- 1D stem along height --------
        self.conv1d = nn.Conv1d(
            in_channels=23,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1d = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # After this stem, height: 160 -> 80 -> 40, channels=64, width=6

        # -------- ResNet-18 body (2D) --------
        self.inplanes = 64

        # layer1: 64 channels, no height downsample, width /2 : 40x6 -> 40x3
        self.layer1 = self._make_layer(64, blocks=2, stride_h=1, stride_w=2)

        # layer2: 128 channels, H /2, W /2 : 40x3 -> 20x2
        self.layer2 = self._make_layer(128, blocks=2, stride_h=2, stride_w=2)

        # layer3: 256 channels, H /2, W /2 : 20x2 -> 10x1
        self.layer3 = self._make_layer(256, blocks=2, stride_h=2, stride_w=2)

        # layer4: 512 channels, H /2, W same: 10x1 -> 5x1
        self.layer4 = self._make_layer(512, blocks=2, stride_h=2, stride_w=1)

        # No final spatial pooling, so forward_features returns [B, 512, 5, 1]
        self.num_classes = num_classes
        if num_classes is not None:
            # Flatten 512*5*1 -> num_classes
            self.fc = nn.Linear(512 * self.data_length * 1, num_classes)

    def _make_layer(self, planes, blocks, stride_h, stride_w):
        """Create one ResNet-18 stage with arbitrary (stride_h, stride_w)."""
        stride = (stride_h, stride_w)
        downsample = None

        if stride_h != 1 or stride_w != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride=stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(
            BasicBlock(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        """
        x: [B, 23, 160, 6]
        -> [B, 512, 5, 1]
        """
        x = torch.moveaxis(x, 1, 3)
        B, C, H, W = x.shape
        #assert C == 23 and H == 160 and W == 6, "Expected [B, 23, 160, 6], got [B, %d, %d, %d]" % (C, H, W)

        # ---- 1D stem over height ----
        # Treat each width position as a separate 1D sequence over height.
        # [B, 23, 160, 6] -> [B, 6, 23, 160] -> [B*6, 23, 160]
        x = x.permute(0, 3, 1, 2).reshape(B * W, C, H)

        x = self.conv1d(x)     # [B*6, 64, 80]
        x = self.bn1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)  # [B*6, 64, 40]

        # Reshape to 2D feature map: [B, 64, H=40, W=6]
        _, C1, H1 = x.shape
        x = x.view(B, W, C1, H1).permute(0, 2, 3, 1)  # [B, 64, 40, 6]

        # ---- 2D ResNet-18 body ----
        x = self.layer1(x)  # [B, 64, 40, 3]
        x = self.layer2(x)  # [B, 128, 20, 2]
        x = self.layer3(x)  # [B, 256, 10, 1]
        x = self.layer4(x)  # [B, 512,  5, 1]

        return x  # [B, 512, 5, 1]

    def forward(self, x):
        x = self.forward_features(x)  # [B, 512, 5, 1]

        if self.num_classes is None:
            return x

        x = torch.flatten(x, 1)      # [B, 512*5*1]
        x = self.fc(x)               # [B, num_classes]
        return x



class SpectrogramCNN1D(nn.Module):
    """
    1D CNN that processes frequency bins for each time step independently.
    
    Input shape: (batch, 6, 19, 160)
    - 6 time steps
    - 19 EEG channels (treated as input channels)
    - 160 frequency bins
    
    Architecture:
    - Applies 1D convolutions along frequency axis for each time step
    - Concatenates features from all 6 time steps
    - Passes through fully connected layers for classification
    """
    
    def __init__(self, num_classes=1, dropout=0.3):
        super(SpectrogramCNN1D, self).__init__()
        self.num_channels = 23
        self.num_time_steps = 6
        self.num_freq_bins = 150
        # 1D Convolutional layers (applied along frequency axis)
        # Input channels = 23 (EEG channels)
        self.conv1 = nn.Conv1d(in_channels=self.num_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions: 160 -> 80 -> 40 -> 20
        self.feature_size = 256 * (self.num_freq_bins // 8) * self.num_time_steps  # channels * freq_bins // 8 * time_steps
        ## note: feature size is pretty large: 256 * 19 * 6 = 29568
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    def forward(self, x):
        # Input shape: (batch, 6, 19, 160)
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Reshape to process each time step: (batch * 6, 19, 160)
        x = x.reshape(batch_size * self.num_time_steps, self.num_channels, self.num_freq_bins)
        
        # Apply 1D convolutions along frequency axis
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Reshape back: (batch, 6, 256, 20)
        x = x.reshape(batch_size, self.num_time_steps, 256, -1)
        
        # Flatten and concatenate features from all time steps
        x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class SpectrogramCNN2D(nn.Module):
    """
    2D CNN that treats the spectrogram as a 2D image.
    
    Input shape: (batch, 6, 19, 160)
    - Reshaped to (batch, 6, 19, 160) where 6 is treated as channels
    - Spatial dimensions: 19 (EEG channels) x 160 (frequency bins)
    
    Architecture:
    - Standard 2D convolutions across channel-frequency space
    - Time dimension treated as input channels
    - Global pooling and fully connected layers for classification
    """
    
    def __init__(self, num_classes=1, dropout=0.3):
        super(SpectrogramCNN2D, self).__init__()
        self.num_channels = 23
        self.num_freq_bins = 150
        self.num_time_steps = 6
        # 2D Convolutional layers
        # Input: 6 channels (time steps), spatial dims: 19 x 160
        self.conv1 = nn.Conv2d(in_channels=self.num_time_steps, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_classes)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    def forward(self, x):
        # Input shape: (batch, 6, 19, 160)
        # Already in the right format for 2D conv
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


import torch
from torchaudio.transforms import Spectrogram


class WelchSpectrogramTransform:
    def __init__(self, fs=200, resolution=0.1, win_length=1000, hop_length=1000, pad=0, min_freq=0, max_freq=32, resolution_factor=1):
        n_fft = int(fs / resolution)
        self.fs = fs
        self.resolution = resolution
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resolution_factor = resolution_factor
        
        # For Welch method: use non-overlapping segments (noverlap=0)
        # Each segment is processed independently with the spectrogram
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=win_length, pad=0, power=2, center=False)
        
        self.freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1)
        self.freq_mask = (self.freqs >= self.min_freq) & (self.freqs < self.max_freq)
    
    def __call__(self, data):
        """
        Args:
            data (Tensor): Time series data to be transformed (channels, samples)
        
        Returns:
            Tensor: Welch spectrogram (channels, frequencies, time_segments)
        """
        n_channels, n_samples = data.shape
        
        # Calculate number of segments based on hop_length
        n_segments = (n_samples - self.win_length) // self.hop_length + 1
        
        segment_specs = []
        
        # Process each segment separately
        for i in range(n_segments):
            start_idx = i * self.hop_length
            end_idx = start_idx + self.win_length
            
            segment = data[:, start_idx:end_idx]
            
            # Compute spectrogram for this segment (will have only 1 time bin since hop_length=win_length)
            spec_segment = self.spec(segment)  # (channels, freqs, 1)
            segment_specs.append(spec_segment.squeeze(-1))  # (channels, freqs)
        
        # Stack all segments along time dimension
        spec = torch.stack(segment_specs, dim=-1)  # (channels, freqs, n_segments)
        
        # Take the log
        spec = torch.log(spec + 1)
        
        # Apply frequency masking
        spec = spec[:, self.freq_mask, :]
        
        # Apply resolution factor if needed
        if self.resolution_factor > 1:
            old_shape = spec.shape
            mag_bands = spec.view(spec.size(0), spec.size(1) // self.resolution_factor, self.resolution_factor, spec.size(2)).mean(dim=2)
            new_shape = mag_bands.shape
            print(f"Old shape: {old_shape}, New shape: {new_shape}")
            return mag_bands
        
        return spec
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fs={self.fs}, resolution={self.resolution}, min_freq={self.min_freq}, max_freq={self.max_freq})"


class MultitaperSpectrogramTransform:
    def __init__(
        self,
        fs=200,
        resolution=0.1,
        win_length=1000,
        hop_length=1000,
        pad=0,
        min_freq=0,
        max_freq=32,
        resolution_factor=1,
        NW=3.5,
        K=None,
        center=True,
    ):
        """
        Multitaper spectrogram using DPSS tapers and torchaudio Spectrogram.

        Args:
            fs: sampling rate (Hz)
            resolution: frequency resolution (Hz) → n_fft = fs / resolution
            win_length: STFT window length (samples)
            hop_length: STFT hop length (samples)
            pad: STFT padding
            min_freq, max_freq: keep freqs in [min_freq, max_freq)
            resolution_factor: optional pooling factor along frequency axis
            NW: time-bandwidth product for DPSS
            K: number of tapers; if None, uses K = int(2 * NW - 1)
            center: passed to torchaudio Spectrogram
        """
        n_fft = int(fs / resolution)

        self.fs = fs
        self.resolution = resolution
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resolution_factor = resolution_factor
        self.NW = NW
        self.center = center

        # --- DPSS tapers (length = win_length) ---
        if K is None:
            K = int(2 * NW - 1)
        tapers_np = dpss(win_length, NW, Kmax=K)  # (K, win_length)
        tapers = torch.tensor(tapers_np.copy(), dtype=torch.float32)  # store on CPU
        self.K = tapers.shape[0]
        self._tapers = tapers  # (K, win_length)

        # --- Build a Spectrogram transform per taper ---
        self.specs = []
        for k in range(self.K):
            taper_k = self._tapers[k]

            def make_window_fn(taper):
                # window_fn signature: (win_length, *, dtype, layout, device, **kwargs)
                def window_fn(win_length, dtype=None, layout=None, device=None, **kwargs):
                    t = taper
                    if device is not None and t.device != device:
                        t = t.to(device)
                    if dtype is not None and t.dtype != dtype:
                        t = t.to(dtype)
                    # win_length is ignored because taper already has correct length
                    return t
                return window_fn

            spec = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad=pad,
                power=2.0,
                center=center,
                window_fn=make_window_fn(taper_k),
            )
            self.specs.append(spec)

        # --- Frequency axis + mask (same as your class) ---
        self.freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1)
        self.freq_mask = (self.freqs >= self.min_freq) & (self.freqs < self.max_freq)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: time series, shape (T,) or (T, C).
                  Internally we use (C, T) for torchaudio.

        Returns:
            Multitaper spectrogram:
            - shape (C, F', frames) after freq selection (+ optional pooling)
        """
        # Ensure shape (T, C)
        if data.dim() == 1:
            data = data.unsqueeze(1)  # (T, 1)
        elif data.dim() != 2:
            raise ValueError("data must be 1D (T,) or 2D (T, C)")

        # (C, T) for torchaudio
        x = data.T

        # --- Multitaper accumulation ---
        spec_accum = None
        for spec in self.specs:
            # spec(x): (C, F, frames)
            spec_k = spec(x)
            if spec_accum is None:
                spec_accum = spec_k
            else:
                spec_accum += spec_k

        # Average over tapers
        spec_mt = spec_accum / self.K  # (C, F, frames)

        # Log-power
        spec_mt = torch.log(spec_mt + 1.0)

        # Frequency mask
        spec_mt = spec_mt[:, self.freq_mask, :]  # (C, F_masked, frames)

        # Optional pooling along frequency axis
        if self.resolution_factor > 1:
            C, F, T_frames = spec_mt.shape
            F_new = F // self.resolution_factor
            # Drop extra bins that don't fit evenly
            spec_mt = spec_mt[:, :F_new * self.resolution_factor, :]
            spec_mt = spec_mt.view(
                C, F_new, self.resolution_factor, T_frames
            ).mean(dim=2)  # (C, F_new, T_frames)

        return spec_mt

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(fs={self.fs}, resolution={self.resolution}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, NW={self.NW}, K={self.K})"
        )



class SpectrogramTransform:
    def __init__(self, fs=200, resolution=0.1, win_length=1000, hop_length=1000, pad=0, min_freq=0, max_freq=32, resolution_factor=1):
        n_fft = int(fs / resolution)
        self.fs = fs
        self.resolution = resolution
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resolution_factor = resolution_factor
        
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad=pad, power=2, center=True)
        self.freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1)
        self.freq_mask = (self.freqs >= self.min_freq) & (self.freqs < self.max_freq)
    
    def __call__(self, data):
        """
        Args:
            data (Tensor): Time series data to be transformed 
        
        Returns:
            Tensor: Spectrogram 
        """
        # Convert PIL Image to tensor if needed
        spec = self.spec(data.T)
        ## take the log
        spec = torch.log(spec + 1)
        spec = spec[:,self.freq_mask,:]
        if self.resolution_factor > 1:
            old_shape = spec.shape
            mag_bands = spec.view(spec.size(0), spec.size(1) // self.resolution_factor, self.resolution_factor, spec.size(2)).mean(dim=2)
            new_shape = mag_bands.shape
            print(f"Old shape: {old_shape}, New shape: {new_shape}")
            return mag_bands
        return spec
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fs={self.fs}, resolution={self.resolution}, min_freq={self.min_freq}, max_freq={self.max_freq})"

class SpecNorm:
    def __init__(self):
        self.mean_vector = [16.19405746459961, 16.372554779052734, 14.878664016723633, 14.022847175598145, 13.58432674407959, 13.20881462097168, 12.811266899108887, 12.446072578430176, 12.185214042663574, 11.980947494506836, 11.74456787109375, 11.492725372314453, 11.296367645263672, 11.109781265258789, 10.95837116241455, 10.80350399017334, 10.637757301330566, 10.504066467285156, 10.367981910705566, 10.266942024230957, 10.453786849975586, 11.32030200958252, 11.116671562194824, 10.05437183380127, 10.540855407714844, 10.345865249633789, 9.646970748901367, 9.872892379760742, 9.886451721191406, 9.501070976257324, 9.545226097106934, 9.424946784973145, 9.28183650970459, 9.278555870056152, 9.226778030395508, 9.163461685180664, 9.211518287658691, 9.575961112976074, 9.369407653808594, 9.047714233398438, 8.984261512756348, 9.018536567687988, 9.089254379272461, 9.15302848815918, 9.337921142578125, 9.904399871826172, 10.004055976867676, 9.568471908569336, 9.012505531311035, 8.794772148132324, 8.72099494934082, 8.57557487487793, 8.557533264160156, 8.51176643371582, 8.493462562561035, 8.476357460021973, 8.357659339904785, 8.360045433044434, 8.362343788146973, 8.28121280670166, 8.238194465637207, 8.446574211120605, 8.256462097167969, 7.898650646209717, 7.731672286987305, 7.3945465087890625, 7.011244773864746, 6.8522210121154785, 6.409739017486572, 6.488124370574951, 7.33641242980957, 6.915400505065918, 4.728175163269043, 4.362795352935791, 4.324657917022705, 4.315652847290039, 4.298262596130371, 4.274855613708496, 4.271076202392578, 4.438477993011475, 4.606916904449463, 4.465102195739746, 4.2505035400390625, 4.215749263763428, 4.198803424835205, 4.189742565155029, 4.171741008758545, 4.156506538391113, 4.143888473510742, 4.1591949462890625, 4.155059814453125, 4.121484756469727, 4.109125137329102, 4.1154465675354, 4.122100830078125, 4.110441207885742, 4.1073126792907715, 4.093348503112793, 4.082906246185303, 4.0964741706848145, 4.088315963745117, 4.070733070373535, 4.059831619262695, 4.051235198974609, 4.039475917816162, 4.019139289855957, 4.011837005615234, 4.0166168212890625, 4.015143394470215, 4.005015850067139, 3.986874580383301, 3.9810943603515625, 3.9885947704315186, 3.999724864959717, 3.9672634601593018, 3.957221269607544, 3.9587490558624268, 3.957681894302368, 3.9527149200439453, 3.9339470863342285, 3.9222521781921387, 3.914283275604248, 3.9116389751434326, 3.9002127647399902, 3.88554048538208, 3.882930278778076, 3.8832969665527344, 3.8800737857818604, 3.860673666000366, 3.8444712162017822, 3.8368284702301025, 3.8372747898101807, 3.837620258331299, 3.82710337638855, 3.8218657970428467, 3.8136775493621826, 3.8123018741607666, 3.813462734222412, 3.8370001316070557, 3.9213759899139404, 3.9728503227233887, 3.904034376144409, 3.8038666248321533, 3.7653608322143555, 3.7536873817443848, 3.750148057937622, 3.7515838146209717, 3.7396981716156006, 3.730522871017456, 3.7299838066101074, 3.7336368560791016, 3.7355737686157227, 3.7211010456085205, 3.713090658187866, 3.712925910949707, 3.7168877124786377, 3.7139055728912354, 3.7089686393737793, 3.7243728637695312, 3.775678873062134, 3.834108352661133]
        self.std_vector = [3.147909164428711, 2.4149012565612793, 2.116851806640625, 2.025953531265259, 1.9675747156143188, 1.935352087020874, 1.9442299604415894, 1.9451260566711426, 1.9531354904174805, 1.9444916248321533, 1.9349740743637085, 1.9317876100540161, 1.9237139225006104, 1.9158562421798706, 1.900648832321167, 1.8901748657226562, 1.8834400177001953, 1.8732874393463135, 1.8626047372817993, 1.8471314907073975, 1.7914551496505737, 1.5164088010787964, 1.53233802318573, 1.8000432252883911, 1.619699478149414, 1.6524431705474854, 1.8177744150161743, 1.7455884218215942, 1.7214194536209106, 1.8093042373657227, 1.7804793119430542, 1.8079562187194824, 1.8299609422683716, 1.8336124420166016, 1.8434783220291138, 1.846662163734436, 1.8379515409469604, 1.7526646852493286, 1.7806909084320068, 1.8891808986663818, 1.903706669807434, 1.9484293460845947, 1.9642775058746338, 2.02201247215271, 2.145425796508789, 2.1286253929138184, 2.0439279079437256, 2.1257944107055664, 2.0642240047454834, 1.9896281957626343, 1.9730279445648193, 1.919041633605957, 1.8884228467941284, 1.8708550930023193, 1.8590688705444336, 1.8519514799118042, 1.8306846618652344, 1.8180043697357178, 1.8132506608963013, 1.8182448148727417, 1.788440465927124, 1.7096165418624878, 1.732823371887207, 1.7708914279937744, 1.739167332649231, 1.723270297050476, 1.709932804107666, 1.6523345708847046, 1.6358976364135742, 1.4527297019958496, 1.2357174158096313, 1.1842788457870483, 1.1946330070495605, 1.0223896503448486, 0.9974275827407837, 0.969287633895874, 0.9620851278305054, 0.9681277871131897, 0.9683551788330078, 0.9616838097572327, 0.9392152428627014, 0.9624047875404358, 0.9787255525588989, 0.9730421304702759, 0.9613466858863831, 0.9672926068305969, 0.9852530360221863, 0.9797184467315674, 0.9732064008712769, 0.9915655255317688, 0.9984943270683289, 0.9854306578636169, 0.9687443375587463, 0.9827690124511719, 0.9877626895904541, 0.9796301126480103, 0.9614128470420837, 0.960037112236023, 0.9687541127204895, 0.9684630632400513, 0.9763955473899841, 0.9711956977844238, 0.9673265814781189, 0.9636749625205994, 0.958168625831604, 0.9634952545166016, 0.9670047760009766, 0.962928831577301, 0.9574726223945618, 0.9571653008460999, 0.9648820757865906, 0.967556893825531, 0.970548689365387, 0.9655284881591797, 0.9638271331787109, 0.9682614803314209, 0.9735844731330872, 0.9678427577018738, 0.9650533199310303, 0.9722646474838257, 0.9741992950439453, 0.9720882177352905, 0.9647112488746643, 0.9659121632575989, 0.9722320437431335, 0.9737118482589722, 0.9723458290100098, 0.9670812487602234, 0.9702871441841125, 0.9749603271484375, 0.9748452305793762, 0.971749484539032, 0.969305694103241, 0.9774158000946045, 0.9791685342788696, 0.9788796305656433, 0.9764131307601929, 0.9754301905632019, 0.9831199645996094, 0.9740940928459167, 0.9588984251022339, 0.9746151566505432, 0.9791516661643982, 0.9821842908859253, 0.980713427066803, 0.9769508242607117, 0.9738174676895142, 0.9788573980331421, 0.983623743057251, 0.9825431704521179, 0.9780592918395996, 0.9788674116134644, 0.9849519729614258, 0.9860051870346069, 0.9849647879600525, 0.9789862036705017, 0.9816703796386719, 0.989321231842041, 0.9927940368652344, 0.9951319098472595, 0.9801490902900696]
    def __call__(self, data):
        return (data - self.mean_vector) / self.std_vector

class TUABBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', window_length=5, resolution=0.2):
        assert mode in ['train','val','test']
        self.mode = mode
        self.root = '/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/' + self.mode
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith('.pkl')]
        self.resolution=resolution
        self.window_length=window_length
        self.stride_length=1
        self.min_freq = 0
        self.max_freq = 32
        self.fs=200
        self.spec_transform = SpectrogramTransform(
                fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                pad=self.fs * self.window_length // 2, min_freq=self.min_freq, max_freq=self.max_freq)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        
        X = sample["X"]
        Y = int(sample["y"])
        X = torch.from_numpy(X).float()
        X = self.spec_transform(X.T)
        return X, Y

class TUEVBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', window_length=5, resolution=0.2, stride_length=1):
        assert mode in ['train','val','test']
        self.mode = mode
        if self.mode == 'val':
            self.mode = 'eval'
        self.root = '/data/netmit/sleep_lab/EEG_FM/TUEV/data/v2.0.1/edf/processed/processed_' + self.mode
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith('.pkl')]
        self.resolution=resolution
        self.window_length=window_length
        self.stride_length=stride_length
        self.min_freq = 0
        self.max_freq = 32
        self.fs=200
        self.spec_transform = SpectrogramTransform(
                fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                min_freq=self.min_freq, max_freq=self.max_freq) #pad=self.fs * self.window_length // 2, 
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"][0] - 1)
        X = torch.from_numpy(X).float()
        X = self.spec_transform(X.T)
        return X, Y

class SpectrogramCNN(nn.Module):
    def __init__(self, model='conv1d', num_classes=6, dataset='TUAB') -> None:
        super().__init__()
        assert model in ['conv1d','conv2d','resnet']
        

        self.model_type = model 
        if self.model_type == 'conv1d':
            self.model = SpectrogramCNN1D(num_classes=num_classes)
        elif self.model_type == 'conv2d':
            self.model = SpectrogramCNN2D(num_classes=num_classes)
        elif self.model_type == 'resnet':
            self.model = CustomResNet18(num_classes=num_classes, dataset=dataset)
        
    def preprocess_input(self,x):
        return self.spec_transform(x)
    def forward(self, x):
        x = torch.moveaxis(x, 3, 1)
        x = self.model(x)
        return x 


def validate_welch_against_mne(
    win_length: int = 1000,
    fs: float = 200.0,
    n_per_seg: int = 1000,
    n_overlap: int = 0,
    min_freq: float = 0.0,
    max_freq: float = 32.0,
):
    """
    Validate WelchSpectrogramTransform against MNE's psd_array_welch
    on a single 1D signal window.

    This checks that both implementations produce numerically similar
    spectra (up to an overall scaling factor).
    
    Args:
        win_length: Length of the entire signal
        fs: Sampling frequency
        n_per_seg: Length of each Welch segment (should match win_length in transform)
        n_overlap: Number of samples overlap between segments
        min_freq: Minimum frequency to keep
        max_freq: Maximum frequency to keep
    """
    # ---- 1. Create test signal ----
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(win_length).astype(np.float32)  # (T,)
    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # (1, T) - add channel dimension

    # ---- 2. Instantiate Welch transform ----
    # Choose resolution so that n_fft matches expected frequency resolution
    resolution = fs / n_per_seg

    welch_transform = WelchSpectrogramTransform(
        fs=fs,
        resolution=resolution,
        win_length=n_per_seg,
        hop_length=n_per_seg - n_overlap,
        pad=0,
        min_freq=min_freq,
        max_freq=max_freq,
        resolution_factor=1,
    )

    # ---- 3. Our Welch spectrogram → PSD ----
    # welch_spec: shape (C=1, F_masked, frames)
    welch_spec_log = welch_transform(x_torch)           # (1, F, frames)
    welch_spec_log = welch_spec_log.squeeze(0)          # (F, frames)
    
    # Average across time frames (Welch averaging)
    welch_spec_log_avg = welch_spec_log.mean(dim=-1)   # (F,)

    # Undo log to compare on linear scale
    welch_psd = torch.exp(welch_spec_log_avg) - 1.0    # (F,)
    welch_psd_np = welch_psd.detach().cpu().numpy()

    # ---- 4. MNE Welch PSD with same settings ----
    psd_mne, freqs_mne = psd_array_welch(
        x_np[np.newaxis, :],    # shape (n_epochs=1, n_times)
        sfreq=fs,
        fmin=min_freq,
        fmax=max_freq,
        n_fft=int(fs / resolution),
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average='mean',
        window='hann',
        verbose=False,
    )
    psd_mne = psd_mne[0]        # (F,)

    # ---- 5. Basic shape check ----
    if psd_mne.shape[0] != welch_psd_np.shape[0]:
        raise RuntimeError(
            f"Frequency dimension mismatch: "
            f"MNE {psd_mne.shape[0]} vs ours {welch_psd_np.shape[0]}"
        )

    # ---- 6. Compare up to an overall scaling factor ----
    # Different implementations often differ by a constant scale
    # (e.g., density vs. power, window normalization, etc.).
    # Normalize both to unit mean and then compare.
    psd_mne_norm = psd_mne / psd_mne.mean()
    welch_psd_norm = welch_psd_np / welch_psd_np.mean()

    max_abs_diff = np.max(np.abs(psd_mne_norm - welch_psd_norm))
    l2_diff = np.linalg.norm(psd_mne_norm - welch_psd_norm) / np.linalg.norm(psd_mne_norm)

    print(f"Max abs diff (after normalization): {max_abs_diff:.3e}")
    print(f"Relative L2 error (after normalization): {l2_diff:.3e}")

    # Optionally return values for plotting/debugging
    return {
        "freqs": freqs_mne,
        "psd_mne": psd_mne,
        "psd_ours": welch_psd_np,
        "psd_mne_norm": psd_mne_norm,
        "psd_ours_norm": welch_psd_norm,
        "max_abs_diff": max_abs_diff,
        "rel_l2_diff": l2_diff,
    }

def validate_multitaper_against_mne(
    win_length: int = 1000,
    fs: float = 200.0,
    NW: float = 3.5,
    min_freq: float = 0.0,
    max_freq: float = 32.0,
):
    """
    Validate MultitaperSpectrogramTransform against MNE's psd_array_multitaper
    on a single 1D signal window of length `win_length`.

    This checks that both implementations produce numerically similar
    spectra (up to an overall scaling factor).
    """
    # ---- 1. Create test signal ----
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(win_length).astype(np.float32)  # (T,)
    x_torch = torch.from_numpy(x_np)  # (T,)

    # ---- 2. Instantiate your multitaper transform ----
    # Choose resolution so that n_fft == win_length for clean alignment:
    # n_fft = fs / resolution → resolution = fs / win_length
    resolution = fs / win_length

    mt_transform = MultitaperSpectrogramTransform(
        fs=fs,
        resolution=resolution,
        win_length=win_length,
        hop_length=win_length,   # one frame
        pad=0,
        min_freq=min_freq,
        max_freq=max_freq,
        resolution_factor=1,
        NW=NW,
        K=None,                  # uses K = int(2*NW - 1)
        center=False,            # avoid torchaudio centering/padding
    )

    # ---- 3. Our multitaper spectrogram → PSD ----
    # mt_spec: shape (C=1, F_masked, frames=1)
    mt_spec_log = mt_transform(x_torch)          # (1, F, 1)
    mt_spec_log = mt_spec_log.squeeze(0).squeeze(-1)  # (F,)

    # Undo log to compare on linear scale
    mt_psd = torch.exp(mt_spec_log) - 1.0        # (F,)
    mt_psd_np = mt_psd.detach().cpu().numpy()

    # ---- 4. MNE multitaper PSD with same settings ----
    # MNE uses a 'bandwidth' in Hz. For DPSS:
    #   time-bandwidth NW = T_sec * bandwidth_Hz
    # → bandwidth_Hz = NW / T_sec = NW * fs / win_length
    bandwidth_hz = NW * fs / win_length
    psd_mne, freqs_mne = psd_array_multitaper(
        x_np[np.newaxis, :],    # shape (n_epochs=1, n_times)
        sfreq=fs,
        fmin=min_freq,
        fmax=max_freq,
        bandwidth=bandwidth_hz,
        adaptive=False,
        low_bias=True,
        normalization="full",
        verbose=False,
    )
    psd_mne = psd_mne[0]        # (F,)

    # ---- 5. Basic shape check ----
    if psd_mne.shape[0] != mt_psd_np.shape[0]:
        raise RuntimeError(
            f"Frequency dimension mismatch: "
            f"MNE {psd_mne.shape[0]} vs ours {mt_psd_np.shape[0]}"
        )

    # ---- 6. Compare up to an overall scaling factor ----
    # Different implementations often differ by a constant scale
    # (e.g., density vs. power, window normalization, etc.).
    # Normalize both to unit mean and then compare.
    psd_mne_norm = psd_mne / psd_mne.mean()
    mt_psd_norm = mt_psd_np / mt_psd_np.mean()

    max_abs_diff = np.max(np.abs(psd_mne_norm - mt_psd_norm))
    l2_diff = np.linalg.norm(psd_mne_norm - mt_psd_norm) / np.linalg.norm(psd_mne_norm)

    print(f"Max abs diff (after normalization): {max_abs_diff:.3e}")
    print(f"Relative L2 error (after normalization): {l2_diff:.3e}")

    # Optionally return values for plotting/debugging
    return {
        "freqs": freqs_mne,
        "psd_mne": psd_mne,
        "psd_ours": mt_psd_np,
        "psd_mne_norm": psd_mne_norm,
        "psd_ours_norm": mt_psd_norm,
        "max_abs_diff": max_abs_diff,
        "rel_l2_diff": l2_diff,
    }


if __name__ == "__main__":
    multitaper_results = validate_multitaper_against_mne()
    welch_results = validate_welch_against_mne()
    print(welch_results)
    print(multitaper_results)
    bp() 
    # trainset = TUEVBaselineDataset(mode='train', window_length=5, resolution=0.2)
    # aa = trainset[0]
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    # model = SpectrogramCNN(model='conv1d', num_classes=6)
    # model2 = SpectrogramCNN(model='conv2d', num_classes=6)
    # for X, Y in trainloader:
    #     bp() 
    #     output = model(X)
    #     output2 = model2(X)
    #     bp() 
    #     print(output.shape, output2.shape, Y)
    #     break