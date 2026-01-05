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
from scipy.integrate import trapezoid
from scipy.signal.windows import dpss
from scipy.signal import get_window
import torch.nn as nn
import torch.nn.functional as F
import pickle
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import time
from scipy.fft import rfftfreq, rfft
from scipy.signal.windows import dpss as sp_dpss
import matplotlib.pyplot as plt


""" 
To do: reorder the channels to make sense for the conv2d model 
add metrics (same as paper) and create training script 
compare and modify the model architectures 

"""

def _mt_spectra(x, dpss, sfreq, n_fft=None, remove_dc=True):
    """Compute tapered spectra.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        Input signal
    dpss : array, shape=(n_tapers, n_times)
        The tapers
    sfreq : float
        The sampling frequency
    n_fft : int | None
        Length of the FFT. If None, the number of samples in the input signal
        will be used.
    %(remove_dc)s

    Returns
    -------
    x_mt : array, shape=(..., n_tapers, n_freqs)
        The tapered spectra
    freqs : array, shape=(n_freqs,)
        The frequency points in Hz of the spectra
    """
    if n_fft is None:
        n_fft = x.shape[-1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    if remove_dc:
        x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = rfftfreq(n_fft, 1.0 / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)), dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.0)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.0)
    return x_mt, freqs

def dpss_windows(N, half_nbw, Kmax, *, sym=True, norm=None, low_bias=True):
    """Compute Discrete Prolate Spheroidal Sequences.

    Will give of orders [0,Kmax-1] for a given frequency-spacing multiple
    NW and sequence length N.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    N : int
        Sequence length.
    half_nbw : float
        Standardized half bandwidth corresponding to 2 * half_bw = BW*f0
        = BW*N/dt but with dt taken as 1.
    Kmax : int
        Number of DPSS windows to return is Kmax (orders 0 through Kmax-1).
    sym : bool
        Whether to generate a symmetric window (``True``, for filter design) or
        a periodic window (``False``, for spectral analysis). Default is
        ``True``.

        .. versionadded:: 1.3
    norm : 2 | ``'approximate'`` | ``'subsample'`` | None
        Window normalization method. If ``'approximate'`` or ``'subsample'``,
        windows are normalized by the maximum, and a correction scale-factor
        for even-length windows is applied either using
        ``N**2/(N**2+half_nbw)`` ("approximate") or a FFT-based subsample shift
        ("subsample"). ``2`` uses the L2 norm. ``None`` (the default) uses
        ``"approximate"`` when ``Kmax=None`` and ``2`` otherwise.

        .. versionadded:: 1.3
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.

    Returns
    -------
    v, e : tuple,
        The v array contains DPSS windows shaped (Kmax, N).
        e are the eigenvalues.

    Notes
    -----
    Tridiagonal form of DPSS calculation from :footcite:`Slepian1978`.

    References
    ----------
    .. footbibliography::
    """
    # TODO VERSION can be removed with SciPy 1.16 is min,
    # workaround for https://github.com/scipy/scipy/pull/22344
    if N <= 1:
        dpss, eigvals = np.ones((1, 1)), np.ones(1)
    else:
        dpss, eigvals = sp_dpss(
            N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True
        )
    if low_bias:
        idx = eigvals > 0.9
        if not idx.any():
            
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive, verbose=None):
    """Triage windowing and multitaper parameters."""
    # Compute standardized half-bandwidth
    if isinstance(bandwidth, str):
        
        window_fun = get_window(bandwidth, n_times)[np.newaxis]
        return window_fun, np.ones(1), False

    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / (2.0 * sfreq)
    else:
        half_nbw = 4.0
    if half_nbw < 0.5:
        raise ValueError(
            f"bandwidth value {bandwidth} yields a normalized half-bandwidth of "
            f"{half_nbw} < 0.5, use a value of at least {sfreq / n_times}"
        )

    # Compute DPSS windows
    n_tapers_max = int(2 * half_nbw)
    window_fun, eigvals = dpss_windows(
        n_times, half_nbw, n_tapers_max, sym=False, low_bias=low_bias
    )


    if adaptive and len(eigvals) < 3:
        adaptive = False

    return window_fun, eigvals, adaptive


def _psd_from_mt(x_mt, weights):
    """Compute PSD from tapered spectra.

    Parameters
    ----------
    x_mt : array, shape=(..., n_tapers, n_freqs)
        Tapered spectra
    weights : array, shape=(n_tapers,)
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array, shape=(..., n_freqs)
        The computed PSD
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd

def _psd_from_mt_adaptive(x_mt, eigvals, freq_mask, max_iter=250, return_weights=False):
    r"""Use iterative procedure to compute the PSD from tapered spectra.

    .. note:: Modified from NiTime.

    Parameters
    ----------
    x_mt : array, shape=(n_signals, n_tapers, n_freqs)
        The DFTs of the tapered sequences (only positive frequencies)
    eigvals : array, length n_tapers
        The eigenvalues of the DPSS tapers
    freq_mask : array
        Frequency indices to keep
    max_iter : int
        Maximum number of iterations for weight computation.
    return_weights : bool
        Also return the weights

    Returns
    -------
    psd : array, shape=(n_signals, np.sum(freq_mask))
        The computed PSDs
    weights : array shape=(n_signals, n_tapers, np.sum(freq_mask))
        The weights used to combine the tapered spectra

    Notes
    -----
    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`
    """
    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError("Need one eigenvalue for each taper")

    if n_tapers < 3:
        raise ValueError("Not enough tapers to compute adaptive weights.")

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = trapezoid(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
    del psd_est

    # allocate space for output
    psd = np.empty((n_signals, np.sum(freq_mask)))

    # only keep the frequencies of interest
    x_mt = x_mt[:, :, freq_mask]

    if return_weights:
        weights = np.empty((n_signals, n_tapers, psd.shape[1]))

    for i, (xk, var) in enumerate(zip(x_mt, x_var)):
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = _psd_from_mt(xk[:2, :], rt_eig[:2, np.newaxis])

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = psd_iter / (
                eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var
            )
            d_k *= rt_eig[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged
            err -= d_k
            if np.max(np.mean(err**2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        psd[i, :] = psd_iter

        if return_weights:
            weights[i, :, :] = d_k

    if return_weights:
        return psd, weights
    else:
        return psd

def our_psd_array_multitaper(
    x,
    sfreq,
    fmin=0.0,
    fmax=np.inf,
    bandwidth=None,
    adaptive=False,
    low_bias=True,
    normalization="length",
    remove_dc=True,
    output="power",
    n_jobs=None,
    *,
    max_iter=150,
    verbose=None,
):
    r"""Compute power spectral density (PSD) using a multi-taper method.

    The power spectral density is computed with DPSS
    tapers :footcite:p:`Slepian1978`.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    %(fmin_fmax_psd)s
    bandwidth : float
        Frequency bandwidth of the multi-taper window function in Hz. For a
        given frequency, frequencies at ``± bandwidth / 2`` are smoothed
        together. The default value is a bandwidth of
        ``8 * (sfreq / n_times)``.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    %(normalization)s
    %(remove_dc)s
    output : str
        The format of the returned ``psds`` array, ``'complex'`` or
        ``'power'``:

        * ``'power'`` : the power spectral density is returned.
        * ``'complex'`` : the complex fourier coefficients are returned per
          taper.
    %(n_jobs)s
    %(max_iter_multitaper)s
    %(verbose)s

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or (..., n_tapers, n_freqs)
        The power spectral densities. All dimensions up to the last (or the
        last two if ``output='complex'``) will be the same as input.
    freqs : array
        The frequency points in Hz of the PSD.
    weights : ndarray
        The weights used for averaging across tapers. Only returned if
        ``output='complex'``.

    See Also
    --------
    csd_multitaper
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd

    Notes
    -----
    .. versionadded:: 0.14.0

    References
    ----------
    .. footbibliography::
    """

    # Reshape data so its 2-D for parallelization
    ndim_in = x.ndim
    x = np.atleast_2d(x)
    n_times = x.shape[-1]
    dshape = x.shape[:-1]
    x = x.reshape(-1, n_times)

    dpss, eigvals, adaptive = _compute_mt_params(
        n_times, sfreq, bandwidth, low_bias, adaptive
    )
    n_tapers = len(dpss)
    print('Number of tapers:', n_tapers)
    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]

    # decide which frequencies to keep
    freqs = rfftfreq(n_times, 1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs < fmax)
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    if output == "complex":
        psd = np.zeros((x.shape[0], n_tapers, n_freqs), dtype="complex")
    else:
        psd = np.zeros((x.shape[0], n_freqs))

    # Let's go in up to 50 MB chunks of signals to save memory
    n_chunk = max(50000000 // (len(freq_mask) * len(eigvals) * 16), 1)
    offsets = np.concatenate((np.arange(0, x.shape[0], n_chunk), [x.shape[0]]))
    for start, stop in zip(offsets[:-1], offsets[1:]):
        x_mt = _mt_spectra(x[start:stop], dpss, sfreq, remove_dc=remove_dc)[0]
        if output == "power":
            if not adaptive:
                psd[start:stop] = _psd_from_mt(x_mt[:, :, freq_mask], weights)
            else:
                parallel, my_psd_from_mt_adaptive, n_jobs = parallel_func(
                    _psd_from_mt_adaptive, n_jobs
                )
                n_splits = min(stop - start, n_jobs)
                out = parallel(
                    my_psd_from_mt_adaptive(x, eigvals, freq_mask, max_iter)
                    for x in np.array_split(x_mt, n_splits)
                )
                psd[start:stop] = np.concatenate(out)
        else:
            psd[start:stop] = x_mt[:, :, freq_mask]

    if normalization == "full":
        psd /= sfreq

    # Combining/reshaping to original data shape
    last_dims = (n_freqs,) if output == "power" else (n_tapers, n_freqs)
    psd.shape = dshape + last_dims
    if ndim_in == 1:
        psd = psd[0]

    if output == "complex":
        return psd, freqs, weights
    else:
        return psd, freqs

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

    def __init__(self, num_classes=None, dataset='TUAB', num_channels=23, data_length=10):
        super().__init__()
        self.data_length = data_length
        # -------- 1D stem along height --------
        self.conv1d = nn.Conv1d(
            in_channels=num_channels,
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
            self.fc = nn.Linear(512 * (self.data_length), num_classes)

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
    
    def __init__(self, num_classes=1, dropout=0.3, num_channels=23):
        super(SpectrogramCNN1D, self).__init__()
        self.num_channels = num_channels
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
    
    def __init__(self, num_classes=1, dropout=0.3, num_channels=23):
        super(SpectrogramCNN2D, self).__init__()
        self.num_channels = num_channels
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
        print('Welch: Number of segments:', n_segments)
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

class PercentileNormalize:
    def __init__(self, low_value=0, high_value=20):
        self.low_value = low_value
        self.high_value = high_value
    
    def __call__(self, data):
        """
        Normalize data to [-1, 1] using percentile-based scaling.
        
        Formula: (data - p_low) / (p_high - p_low) * 2 - 1
        This maps [p_low, p_high] to [-1, 1]
        """
        # Avoid division by zero
        range_val = self.high_value - self.low_value
        range_val = max(range_val, 1e-8)
        normalized = (data - self.low_value) / range_val * 2.0 - 1.0
        normalized = torch.clamp(normalized, min=-1.0, max=1.0)
        return normalized

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
        bandwidth=2.0,
        center=True,
        normalization="full",
        device=None,
    ):
        """
        Multitaper spectrogram using DPSS tapers and PyTorch FFT.
        Optimized for GPU acceleration and dataloader preprocessing.
        Matches MNE's psd_array_multitaper implementation.
        
        Performance Optimizations:
        1. Pre-computes DPSS tapers, weights, and frequency mask as torch tensors
        2. Eliminates CPU-GPU transfers (all operations stay on device)
        3. Fully vectorized FFT computation (no channel loops)
        4. Pre-computes normalization constants
        5. Uses torch boolean indexing instead of numpy masks
        
        Expected speedup: 5-10x faster than MNE on CPU, 10-50x on GPU
        (depending on batch size and number of channels).

        Args:
            fs: sampling rate (Hz)
            resolution: frequency resolution (Hz) → n_fft = fs / resolution
            win_length: STFT window length (samples)
            hop_length: STFT hop length (samples)
            pad: STFT padding
            min_freq, max_freq: keep freqs in [min_freq, max_freq]
            resolution_factor: optional pooling factor along frequency axis
            NW: time-bandwidth product for DPSS
            K: number of tapers; if None, uses K = int(2 * NW - 1)
            center: passed to torchaudio Spectrogram (not used in optimized version)
            normalization: "length" or "full" (matches MNE). If "full", divides by sfreq.
            device: device to store pre-computed tensors on (None = CPU, will move to data device)
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
        self.bandwidth = bandwidth #NW * fs / win_length
        self.NW = bandwidth * win_length / fs
        self.center = center
        self.normalization = normalization
        self.n_fft = n_fft
        self.device = device  # Store device preference

        # Use MNE's _compute_mt_params to get tapers and eigenvalues (exact match)
        # Compute bandwidth from NW: bandwidth = NW * fs / win_length
        
        dpss_np, eigvals_np, _ = _compute_mt_params(
            win_length, fs, bandwidth, low_bias=True, adaptive=False
        )
        
        
        self.K = len(eigvals_np)
        
        # Pre-compute as torch tensors (major speedup: no conversion on each call)
        # Store on CPU initially, will move to data device when needed
        self._dpss = torch.from_numpy(dpss_np.copy()).float()  # (K, win_length)
        self._eigvals = torch.from_numpy(eigvals_np.copy()).float()  # (K,)
        
        # Compute weights (sqrt of eigenvalues) as MNE does
        # Store as real, will convert to complex when needed for multiplication
        self._weights = torch.sqrt(self._eigvals)  # (K,) real
        
        # Pre-compute weight normalization constant (major speedup)
        # weights are real, so |weights|^2 = weights^2
        self._weight_norm = (self._weights * self._weights).sum().item()  # scalar
        
        # Frequency axis (matches MNE's rfftfreq - uses signal length, not n_fft)
        # MNE uses: freqs = rfftfreq(n_times, 1.0 / sfreq) where n_times = win_length
        freqs_np = rfftfreq(win_length, 1.0 / fs)
        freq_mask_np = (freqs_np >= min_freq) & (freqs_np < max_freq)
        
        # Pre-compute frequency mask as torch tensor
        self._freq_mask = torch.from_numpy(freq_mask_np.copy()).bool()  # (n_freqs,)
        self._n_freqs_masked = int(self._freq_mask.sum().item())
        
        # Pre-compute constants
        self._sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self._n_fft_mt = win_length  # MNE uses signal length by default
        self._n_freqs = self._n_fft_mt // 2 + 1
        
        # Normalization factor (pre-compute)
        self._norm_factor = 1.0 / fs if normalization == "full" else 1.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Optimized GPU-accelerated multitaper spectrogram computation.
        
        Args:
            data: time series, shape (T,) or (T, C).
                  Internally we use (C, T) for processing.

        Returns:
            Multitaper spectrogram:
            - shape (C, F', frames) after freq selection (+ optional pooling)
        """
        # Ensure shape (T, C)
        if data.dim() == 1:
            data = data.unsqueeze(1)  # (T, 1)
        elif data.dim() != 2:
            raise ValueError("data must be 1D (T,) or 2D (T, C)")

        # (C, T) for processing
        x = data.T  # (C, T)
        
        if self.hop_length < x.shape[1]:#self.win_length < data.shape[1]:
            ## pad each side 
            middle_idx = x.shape[1] // 2
            counter = middle_idx % self.hop_length
            counter -= self.win_length // 2
            counter *= -1 
            if counter < 0:
                counter = 0
            # print('Shape of data: ', data.shape)
            # print(f'Paddying left and right by {counter / self.fs:.2f}s')
            # print('T before padding: ', x.shape[1])
            x = torch.nn.functional.pad(x, (counter, counter), mode="reflect")
        
        
        C, T = x.shape
        
        # Calculate number of segments based on hop_length
        if T < self.win_length:
            # Pad if needed
            x_padded = torch.nn.functional.pad(x, (0, self.win_length - T), mode='constant', value=0)
            n_segments = 1
        else:
            # print('T: ', T)
            # print('win_length: ', self.win_length)
            # print('hop_length: ', self.hop_length)
            n_segments = (T - self.win_length) // self.hop_length + 1
            # print('n_segments: ', n_segments)
            x_padded = None  # Not needed
        
        device = x.device
        dtype = x.dtype
        
        # Move pre-computed tensors to data device (only once per device)
        # This is much faster than converting numpy->torch on each call
        dpss = self._dpss.to(device=device, dtype=dtype)  # (K, win_length)
        weights = self._weights.to(device=device, dtype=dtype)  # (K,)
        freq_mask = self._freq_mask.to(device=device)  # (n_freqs,)
        sqrt_2 = self._sqrt_2.to(device=device, dtype=dtype)
        
        segment_specs = []
        
        # Process each segment
        for i in range(n_segments):
            start_idx = i * self.hop_length
            end_idx = start_idx + self.win_length
            
            if T < self.win_length:
                # Use padded data
                segment = x_padded[:, :self.win_length]
            else:
                segment = x[:, start_idx:end_idx]
            
            # Remove DC component in torch (no CPU-GPU transfer)
            # segment: (C, win_length)
            segment = segment - segment.mean(dim=-1, keepdim=True)  # (C, win_length)
            
            # VECTORIZED FFT COMPUTATION (major speedup)
            # Instead of looping over channels, reshape to process all at once
            # segment: (C, win_length) -> (C, 1, win_length)
            # dpss: (K, win_length) -> (1, K, win_length)
            # Broadcast: (C, K, win_length)
            segment_expanded = segment.unsqueeze(1)  # (C, 1, win_length)
            dpss_expanded = dpss.unsqueeze(0)  # (1, K, win_length)
            tapered = segment_expanded * dpss_expanded  # (C, K, win_length)
            
            # Reshape for batched FFT: (C*K, win_length)
            C_seg, K, win_len = tapered.shape
            tapered_flat = tapered.reshape(C_seg * K, win_len)  # (C*K, win_length)
            
            # Single batched FFT for all channels and tapers (much faster than loop)
            x_mt_flat = torch.fft.rfft(tapered_flat, n=self._n_fft_mt, dim=-1)  # (C*K, n_freqs) complex
            
            # Reshape back: (C, K, n_freqs)
            x_mt = x_mt_flat.reshape(C_seg, K, self._n_freqs)  # (C, K, n_freqs) complex
            
            # Adjust DC and Nyquist (matches MNE's _mt_spectra exactly)
            x_mt[:, :, 0] = x_mt[:, :, 0] / sqrt_2
            if self._n_fft_mt % 2 == 0:
                x_mt[:, :, -1] = x_mt[:, :, -1] / sqrt_2
            
            # Apply frequency mask using torch indexing (faster than numpy mask)
            x_mt_masked = x_mt[:, :, freq_mask]  # (C, K, n_freqs_masked)
            
            # Use MNE's _psd_from_mt logic with torch operations
            # Convert weights to complex for multiplication with complex x_mt
            weights_complex = weights.to(torch.complex64)  # (K,) complex
            weights_expanded = weights_complex.view(1, -1, 1)  # (1, K, 1) complex
            
            # _psd_from_mt: psd = weights * x_mt
            psd = weights_expanded * x_mt_masked  # (C, K, n_freqs_masked) complex
            
            # psd *= psd.conj() to get |weights * x_mt|^2
            psd = psd * psd.conj()  # (C, K, n_freqs_masked) complex
            psd = psd.real  # (C, K, n_freqs_masked) real
            
            # Sum over tapers (axis=1, which is -2 in MNE's notation)
            psd = psd.sum(dim=1)  # (C, n_freqs_masked)
            
            # Normalize using pre-computed weight_norm (major speedup)
            psd = psd * (2.0 / self._weight_norm)  # (C, n_freqs_masked)
            
            # Apply normalization using pre-computed factor
            if self.normalization == "full":
                psd = psd * self._norm_factor
            
            # Log-power
            # psd = torch.log(psd + 1.0)  # (C, n_freqs_masked)
            psd = 10.0 * torch.log10(psd + 1e-10) 
            
            segment_specs.append(psd)  # (C, n_freqs_masked)
        
        # Stack all segments along time dimension
        spec_mt = torch.stack(segment_specs, dim=-1)  # (C, n_freqs_masked, n_segments)

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
        self.n_fft = n_fft
        self.fs = fs
        self.resolution = resolution
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resolution_factor = resolution_factor
        self.num_paddings = self.n_fft - self.win_length if self.win_length < self.n_fft else 0
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad=self.num_paddings//2, power=2, center=False)  
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
        data = data.T
        if self.hop_length < data.shape[1]:#self.win_length < data.shape[1]:
            ## pad each side 
            middle_idx = data.shape[1] // 2
            counter = middle_idx % self.hop_length
            counter -= self.win_length // 2
            counter *= -1 
            if counter < 0:
                counter = 0
            # print('Shape of data: ', data.shape)
            # print(f'Paddying left and right by {counter / self.fs:.2f}s')
            
            x_padded = F.pad(data, (counter, counter), mode="reflect")
            # print('Shape of x_padded: ', x_padded.shape)
            data = x_padded
            
            data = data.unfold(dimension=1, size=self.win_length, step=self.hop_length)
            
            # data = data.reshape(-1, data.shape[1])
        # if self.win_length < self.n_fft:
            
            ## add zeros to the right 
            # print(f'Paddying right by {num_paddings} samples')
            # data = F.pad(data, (0, num_paddings), mode="constant", value=0)
        # print('Shape of data: ', data.shape)

        try:
            spec = self.spec(data).squeeze(-1)
            if len(spec.shape) < 3:
                spec = spec.unsqueeze(1)
            spec = spec.transpose(1, 2)
        except Exception as e:
            print(f'Error in spectrogram: {e}')
            bp()
        # print('Shape of spectrogram: ', spec.shape)
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
    def __init__(self, args, mode='train', window_length=5, resolution=0.2, stride_length=1, multitaper=False, bandwidth=2.0):
        assert mode in ['train','val','test']
        self.mode = mode
        self.args = args
        self.root = '/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/' + self.mode + '_with_spec/'
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith('.pkl')]
        self.resolution=resolution
        self.window_length=window_length
        self.stride_length=stride_length
        self.min_freq = 0
        self.max_freq = 32
        self.fs=200
        self.normalize_spec = args.normalize_spec
        self.percentile_low = args.percentile_low
        self.percentile_high = args.percentile_high
        if multitaper:
            self.spec_transform = MultitaperSpectrogramTransform(
                fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                min_freq=self.min_freq, max_freq=self.max_freq, bandwidth=bandwidth)
        else:
            self.spec_transform = SpectrogramTransform(
                    fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                    pad=self.fs * self.window_length // 2, min_freq=self.min_freq, max_freq=self.max_freq)
        
        if self.args.normalize_spec:
            self.spec_transform = [self.spec_transform]
            self.spec_transform.append(PercentileNormalize(low_value=self.percentile_low, high_value=self.percentile_high))
            self.spec_transform = transforms.Compose(self.spec_transform)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        
        if self.args.load_spec_true:
            X = torch.from_numpy(sample["spec_true"]).float()
        elif self.args.load_spec_recon:
            X = torch.from_numpy(sample["spec_recon"]).float()
        else:
            X = sample["X"]
            X = torch.from_numpy(X).float()
            X = self.spec_transform(X.T)
            # X2 = self.spec_transform(torch.from_numpy(sample["X"]).float().T)
        Y = int(sample["y"])
        return X, Y

class TUEVBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train', window_length=5, resolution=0.2, stride_length=1, multitaper=False, bandwidth=2.0):
        assert mode in ['train','val','test']
        self.mode = mode
        self.args = args
        if self.mode == 'val':
            self.mode = 'eval'
        self.root = '/data/netmit/sleep_lab/EEG_FM/TUEV/data/v2.0.1/edf/processed/processed_' + self.mode + '_with_spec/'
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith('.pkl')]
        self.resolution=resolution
        self.window_length=window_length
        self.stride_length=stride_length
        self.min_freq = 0
        self.max_freq = 32
        self.fs=200
        if multitaper:
            self.spec_transform = MultitaperSpectrogramTransform(
                fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                min_freq=self.min_freq, max_freq=self.max_freq, bandwidth=bandwidth)
        else:
            self.spec_transform = SpectrogramTransform(
                fs=self.fs, resolution=self.resolution, win_length=self.fs * self.window_length, hop_length=self.fs * self.stride_length, 
                min_freq=self.min_freq, max_freq=self.max_freq) #pad=self.fs * self.window_length // 2, 
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        Y = int(sample["label"][0] - 1)
        if self.args.load_spec_true:
            X = torch.from_numpy(sample["spec_true"]).float()
        elif self.args.load_spec_recon:
            X = torch.from_numpy(sample["spec_recon"]).float()
        else:
            X = sample["X"]
            X = torch.from_numpy(X).float()
            X = self.spec_transform(X.T)
        return X, Y

class SpectrogramCNN(nn.Module):
    def __init__(self, model='conv1d', num_classes=6, dataset='TUAB', num_channels=23, data_length=10) -> None:
        super().__init__()
        assert model in ['conv1d','conv2d','resnet']

        self.model_type = model 
        if self.model_type == 'conv1d':
            self.model = SpectrogramCNN1D(num_classes=num_classes, num_channels=num_channels)
        elif self.model_type == 'conv2d':
            self.model = SpectrogramCNN2D(num_classes=num_classes, num_channels=num_channels)
        elif self.model_type == 'resnet':
            self.model = CustomResNet18(num_classes=num_classes, dataset=dataset, num_channels=num_channels, data_length=data_length)
        
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
    bandwidth: float = 1.0,
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
        bandwidth=bandwidth,
        center=False,            # avoid torchaudio centering/padding
        normalization="full",    # matches MNE default
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

    psd_mne, freqs_mne = psd_array_multitaper(
        x_np[np.newaxis, :],    # shape (n_epochs=1, n_times)
        sfreq=fs,
        fmin=min_freq,
        fmax=max_freq,
        bandwidth=bandwidth,
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
    print('Correlation between MNE and our multitaper: ', np.corrcoef(psd_mne_norm, mt_psd_norm)[0, 1])
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


def compare_compute_times(
    win_length: int = 1000,
    fs: float = 200.0,
    n_channels: int = 23,
    n_iterations: int = 100,
    resolution: float = 0.2,
    min_freq: float = 0.0,
    max_freq: float = 32.0,
    bandwidth: float = 1.0,
):
    """
    Compare compute times between different spectrogram methods.
    
    Args:
        win_length: Length of signal window in samples
        fs: Sampling frequency (Hz)
        n_channels: Number of channels (EEG channels)
        n_iterations: Number of iterations for timing (more = more accurate)
        resolution: Frequency resolution (Hz)
        min_freq: Minimum frequency to keep
        max_freq: Maximum frequency to keep
        bandwidth: Bandwidth for multitaper
    
    Returns:
        Dictionary with timing results for each method
    """
    # Calculate n_fft for signal length calculation
    n_fft = int(fs / resolution)
    
    # Create test signal - all methods use win_length for fair comparison
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((n_channels, win_length)).astype(np.float32)  # (C, T)
    x_torch = torch.from_numpy(x_np)  # (C, T)
    x_torch_2d = x_torch.T  # (T, C) for multitaper multi-channel
    x_np_win = x_np  # (C, win_length) - same as x_np
    x_torch_win = x_torch  # (C, win_length) - same as x_torch
    
    # Initialize transforms
    # For SpectrogramTransform, create a timing version with center=False to avoid padding issues
    # This matches the behavior but without the center=True padding that causes problems
    n_fft_timing = int(fs / resolution)
    spec_timing = Spectrogram(n_fft=n_fft_timing, win_length=win_length, hop_length=win_length, pad=0, power=2, center=False)
    freqs_timing = torch.linspace(0, fs / 2, n_fft_timing // 2 + 1)
    freq_mask_timing = (freqs_timing >= min_freq) & (freqs_timing < max_freq)
    
    # Create a simple wrapper function for timing (mimics SpectrogramTransform but with center=False)
    # Note: torchaudio Spectrogram expects (channels, time), and data is already (channels, time)
    def spec_transform_timing(data):
        spec = spec_timing(data)  # data is (channels, time), no transpose needed
        spec = torch.log(spec + 1)
        spec = spec[:, freq_mask_timing, :]
        return spec
    
    welch_transform = WelchSpectrogramTransform(
        fs=fs,
        resolution=resolution,
        win_length=win_length,
        hop_length=win_length,
        pad=0,
        min_freq=min_freq,
        max_freq=max_freq,
        resolution_factor=1,
    )
    
    mt_transform = MultitaperSpectrogramTransform(
        fs=fs,
        resolution=resolution,
        win_length=win_length,
        hop_length=win_length,
        pad=0,
        min_freq=min_freq,
        max_freq=max_freq,
        resolution_factor=1,
        bandwidth=bandwidth,
        center=False,
        normalization="full",  # matches MNE default
    )
    
    # MNE settings (n_fft already calculated above)
    n_per_seg = win_length
    bandwidth_hz = bandwidth 
    
    results = {}
    
    # 1. SpectrogramTransform (our custom) - timing version with center=False
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = spec_transform_timing(x_torch_win)  # Uses win_length segment
    end = time.perf_counter()
    results['SpectrogramTransform'] = (end - start) / n_iterations
    print('Shape of SpectrogramTransform output:', out.shape)
    # 2. WelchSpectrogramTransform (our custom) - uses win_length segment
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = welch_transform(x_torch_win)  # Uses win_length segment
    end = time.perf_counter()
    results['WelchSpectrogramTransform'] = (end - start) / n_iterations
    print('Shape of WelchSpectrogramTransform output:', out.shape)
    # 3. MultitaperSpectrogramTransform (our custom) - uses win_length segment
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    # Create win_length segment in (T, C) format
    x_torch_2d_win = x_torch_win.T  # (win_length, C)
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = mt_transform(x_torch_2d_win)  # (T, C) with win_length
    end = time.perf_counter()
    results['MultitaperSpectrogramTransform'] = (end - start) / n_iterations
    print('Shape of MultitaperSpectrogramTransform output:', out.shape)
    # 4. MNE Welch - uses win_length segment
    start = time.perf_counter()
    for _ in range(n_iterations):
        out, _ = psd_array_welch(
            x_np_win,  # (n_channels, win_length)
            sfreq=fs,
            fmin=min_freq,
            fmax=max_freq,
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=0,
            average='mean',
            window='hann',
            verbose=False,
        )
    print('Shape of MNE Welch output:', out.shape)
    end = time.perf_counter()
    results['MNE_Welch'] = (end - start) / n_iterations
    
    # 5. MNE Multitaper - uses win_length segment
    start = time.perf_counter()
    for _ in range(n_iterations):
        out, _ = psd_array_multitaper(
            x_np_win,  # (n_channels, win_length)
            sfreq=fs,
            fmin=min_freq,
            fmax=max_freq,
            bandwidth=bandwidth_hz,
            adaptive=False,
            low_bias=True,
            normalization="full",
            verbose=False,
        )
    print('Shape of MNE Multitaper output:', out.shape)
    end = time.perf_counter()
    results['MNE_Multitaper'] = (end - start) / n_iterations
    
    # Print results
    print("\n" + "="*60)
    print("Compute Time Comparison (seconds per call)")
    print("="*60)
    print(f"Signal shape: ({n_channels}, {win_length})")
    print(f"Frequency range: [{min_freq}, {max_freq}] Hz")
    print(f"Resolution: {resolution} Hz")
    print(f"Iterations: {n_iterations}")
    print("Note: SpectrogramTransform uses center=False for timing")
    print("-"*60)
    
    # Sort by time
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for method, time_per_call in sorted_results:
        print(f"{method:35s}: {time_per_call*1000:8.3f} ms")
    
    print("-"*60)
    baseline_time = sorted_results[0][1]
    print(f"\nSpeedup relative to fastest ({sorted_results[0][0]}):")
    for method, time_per_call in sorted_results:
        speedup = time_per_call / baseline_time
        print(f"{method:35s}: {speedup:8.2f}x")
    
    print("="*60 + "\n")
    
    return results

def load_h5(file):
    fs = 200  # sampling rate
    
    h5file = h5py.File(file, 'r')
    data = h5file['recording']['data'][:]  # (time, channels)
    data = data.T  # (time, channels)
    h5file.close()
    return data
def visualize_comparison(file):
    """
    Load a random 2 min segment from EEG data and visualize three spectrogram methods side by side.
    
    Args:
        file: Path to h5 file containing EEG data
    """
    fs = 200  # sampling rate
    
    if file.endswith('.h5'):
        segment_length_samples = 2 * 60 * fs  # 2 minutes in samples = 24000
        h5file = h5py.File(file, 'r')
        
        # Get the data - assuming structure like f['recording']['data'] with shape (time, channels)
        if 'recording' in h5file:
            data = h5file['recording']['data'][:]  # (time, channels)
        else:
            # Try to find first subject/group with 'eeg' or 'data' key
            first_key = list(h5file.keys())[0]
            if 'eeg' in h5file[first_key]:
                data = h5file[first_key]['eeg'][:]  # (channels, time) - need to transpose
                data = data.T  # (time, channels)
            elif 'data' in h5file[first_key]:
                data = h5file[first_key]['data'][:]  # (time, channels)
            else:
                raise ValueError("Could not find 'data' or 'eeg' in h5 file structure")
        
        # Close the file
        h5file.close()
    elif file.endswith('.pkl'):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data = data['signal'].T
            segment_length_samples = 5 * fs  # 2 minutes in samples = 24000
    else:
        raise ValueError(f"Unsupported file extension: {file}")
    
    # Parameters
    
    
    
    # Check if we have enough data
    if data.shape[0] < segment_length_samples:
        raise ValueError(f"Data length ({data.shape[0]}) is less than required 2 min segment ({segment_length_samples} samples)")
    
    # Get random start index
    max_start = data.shape[0] - segment_length_samples
    start_idx = random.randint(0, max_start)
    
    # Extract 2 min segment: (time, channels)
    segment = data[start_idx:start_idx + segment_length_samples, :]
    
    # Convert to torch tensors
    # For normal spec and welch: need (channels, time)
    segment_ct = torch.from_numpy(segment.T).float()  # (channels, time)
    
    # For multitaper: need (time, channels)
    segment_tc = torch.from_numpy(segment).float()  # (time, channels)
    
    # Initialize transforms
    resolution = 0.1
    win_length = 1000  # 5 seconds at 200 Hz
    hop_length = 200  # non-overlapping
    
    # Normal spectrogram
    spec_transform = SpectrogramTransform(
        fs=fs, resolution=resolution, win_length=win_length, 
        hop_length=hop_length, pad=0, min_freq=0, max_freq=32
    )
    now = time.time()
    spec_normal = spec_transform(segment_ct.T)  # (channels, freqs, time)
    print(f"Time taken for normal spectrogram: {time.time() - now} seconds")
    # # Welch spectrogram
    # welch_transform = WelchSpectrogramTransform(
    #     fs=fs, resolution=resolution, win_length=win_length//5,
    #     hop_length=hop_length//5, pad=0, min_freq=0, max_freq=32
    # )
    # spec_welch = welch_transform(segment_ct)  # (channels, freqs, time)
    
    # Multitaper spectrogram
    specs = {} 
    specs_hop = {}
    specs_hop_mt = {}
    mt_bandwidths = [1.0, 2.0, 4.0]
    mt_hop_lengths = [1*fs,2*fs,3*fs, 5*fs]
    for bandwidth in mt_bandwidths:
        mt_transform = MultitaperSpectrogramTransform(
            fs=fs, resolution=resolution, win_length=win_length,
            hop_length=hop_length, pad=0, min_freq=0, max_freq=32,
            bandwidth=bandwidth, normalization="full",
        )
        now = time.time()
        spec_multitaper = mt_transform(segment_tc)  # (channels, freqs, time)
        spec_multitaper_np = spec_multitaper[0].detach().cpu().numpy()  # (freqs, time)
        print(f"Time taken for bandwidth {bandwidth}: {time.time() - now} seconds")
        specs[bandwidth] = {'spec': spec_multitaper_np.copy(), 'K': mt_transform.K, 'bandwidth': mt_transform.bandwidth}
    # Convert to numpy for plotting (first channel)
    spec_normal_np = spec_normal[0].detach().cpu().numpy()  # (freqs, time)
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, len(mt_bandwidths) + 1, figsize=(18, 5))
    
    # Helper function to create time and frequency axes for each spectrogram
    def plot_spectrogram(ax, spec_data, title, vmax=None):
        n_freq_bins, n_time_bins = spec_data.shape
        # print(n_freq_bins, n_time_bins)
        # Time axis: each bin represents hop_length/fs seconds
        # For pcolormesh with shading='auto', we need edges (n_bins + 1 points)
        time_edges = np.arange(n_time_bins + 1) * (hop_length / fs)
        # Frequency axis: 0-32 Hz (edges)
        freq_edges = np.linspace(0, 32, n_freq_bins + 1)
        
        im = ax.pcolormesh(time_edges, freq_edges, spec_data, cmap='jet', shading='auto', vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        plt.colorbar(im, ax=ax)
        return im
    
    # Plot each spectrogram
    plot_spectrogram(axes[0], spec_normal_np, 'Normal Spectrogram', vmax=22)
    # plot_spectrogram(axes[1], spec_welch_np, 'Welch Spectrogram')
    for i, bandwidth in enumerate(sorted(specs.keys())):
        plot_spectrogram(axes[i + 1], specs[bandwidth]['spec'], f'Multitaper {specs[bandwidth]["K"]} Tapers {specs[bandwidth]["bandwidth"]:.1f} BW')
    
    for i, hop_length in enumerate(mt_hop_lengths):
        mt_transform = MultitaperSpectrogramTransform(
            fs=fs, resolution=resolution, win_length=win_length,
            hop_length=hop_length, pad=0, min_freq=0, max_freq=32,
            bandwidth=1.0, normalization="full",
        )
        spec_transform = SpectrogramTransform(
            fs=fs, resolution=resolution, win_length=win_length,
            hop_length=hop_length, pad=0, min_freq=0, max_freq=32,
        )
        spec_transform_np = spec_transform(segment_ct.T)  # (channels, freqs, time)
        spec_transform_np = spec_transform_np[0].detach().cpu().numpy()  # (freqs, time)
        specs_hop[hop_length] = {'spec': spec_transform_np.copy()}
        now = time.time()
        spec_multitaper = mt_transform(segment_tc)  # (channels, freqs, time)
        spec_multitaper_np = spec_multitaper[0].detach().cpu().numpy()  # (freqs, time)
        specs_hop_mt[hop_length] = {'spec': spec_multitaper_np.copy(), 'K': mt_transform.K, 'bandwidth': mt_transform.bandwidth}
    fig, axs = plt.subplots(2, len(mt_hop_lengths), figsize=(18, 10))
    for i, hop_length in enumerate(sorted(specs_hop.keys())):
        plot_spectrogram(axs[0, i], specs_hop[hop_length]['spec'], f"Spec Hop Len {hop_length/fs:.2f}s, shape {specs_hop[hop_length]['spec'].shape}")
        plot_spectrogram(axs[1, i], specs_hop_mt[hop_length]['spec'], f'Multitaper {specs_hop_mt[hop_length]["K"]} Tapers {specs_hop_mt[hop_length]["bandwidth"]:.1f} BW')
    plt.tight_layout()
    plt.show()
    
    return fig

def get_multitaper_shapes(file, fs, resolution, win_length, hop_length, min_freq, max_freq, bandwidth, data_length=200 * 60 ):
    data = load_h5(file)
    
    data = data[:,:data_length]
    data = torch.from_numpy(data.T).float()
    mt_transform = MultitaperSpectrogramTransform(
        fs=fs, resolution=resolution, win_length=win_length,
        hop_length=hop_length, pad=0, min_freq=min_freq, max_freq=32,
        bandwidth=bandwidth, normalization="full",
    )
    spec_multitaper = mt_transform(data)  # (channels, freqs, time)
    print('Data is of length (seconds): ', data.shape[0] / fs)
    print('Spec is of shape: ', spec_multitaper.shape)

if __name__ == "__main__":
    # visualize_comparison('sample_eeg/aaaaabor_00000011-16.pkl') #sub-S0001122302611_ses-2_preprocessed-eeg.h5')
    # visualize_comparison('sample_eeg/sub-S0001122302611_ses-2_preprocessed-eeg.h5')

    # multitaper_results = validate_multitaper_against_mne()
    # get_multitaper_shapes('sample_eeg/sub-S0001122302611_ses-2_preprocessed-eeg.h5', 200, 0.1, 1000, 200, 0, 32, 1.0)
    # get_multitaper_shapes('sample_eeg/sub-S0001122302611_ses-2_preprocessed-eeg.h5', 200, 0.1, 1000, 200, 0, 32, 1.0, data_length=200 * 47)
    # get_multitaper_shapes('sample_eeg/sub-S0001122302611_ses-2_preprocessed-eeg.h5', 200, 0.1, 1000, 200, 0, 32, 1.0, data_length=200 * 48)
    # get_multitaper_shapes('sample_eeg/sub-S0001122302611_ses-2_preprocessed-eeg.h5', 200, 0.1, 800, 200, 0, 32, 1.0)
    # bp() 
    # # print((multitaper_results['psd_mne'] - multitaper_results['psd_ours']) / multitaper_results['psd_mne'])
    # # import scipy.stats
    # # print(scipy.stats.pearsonr(multitaper_results['psd_mne'], multitaper_results['psd_ours']))
    
    # welch_results = validate_welch_against_mne()
    # print(welch_results)
    
    # Compare compute times
    # timing_results = compare_compute_times(
    #     win_length=1000,
    #     fs=200.0,
    #     n_channels=23,
    #     n_iterations=100,
    #     resolution=0.2,
    #     min_freq=0.0,
    #     max_freq=32.0,
    #     bandwidth=4.0,
    # )
    # print(timing_results)
    
    # bp() 
    ## window_length=5, resolution=0.2, stride_length=1, multitaper=False, bandwidth=2.0):
    class Object:
        pass
    args = Object()
    args.load_spec_true = False
    args.load_spec_recon = False
    args.normalize_spec = False
    args.percentile_low = -20
    args.percentile_high = 30
    test_cases = [
        {'args': args, 'mode': 'train','window_length': 4, 'resolution': 0.2, 'stride_length': 1, 'multitaper': True, 'bandwidth': 2.0, 'normalize_spec': False, 'percentile_low': -20, 'percentile_high': 30},
        # {'args': args, 'mode': 'train','window_length': 4, 'resolution': 0.2, 'stride_length': 1, 'multitaper': True, 'bandwidth': 1.0},
    ]
    #     {'window_length': 5, 'resolution': 0.2, 'stride_length': 1, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 5, 'resolution': 0.2, 'stride_length': 2, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 5, 'resolution': 0.2, 'stride_length': 4, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 5, 'resolution': 0.2, 'stride_length': 5, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 3, 'resolution': 0.2, 'stride_length': 3, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 3, 'resolution': 0.2, 'stride_length': 1, 'multitaper': False, 'bandwidth': -1},
    #     {'window_length': 1, 'resolution': 0.2, 'stride_length': 1, 'multitaper': False, 'bandwidth': -1},
        
    #     {'window_length': 5, 'resolution': 0.2, 'stride_length': 1, 'multitaper': True, 'bandwidth': 1.0},
    #     {'window_length': 3, 'resolution': 0.2, 'stride_length': 3, 'multitaper': True, 'bandwidth': 1.0},
    #     {'window_length': 3, 'resolution': 0.2, 'stride_length': 1, 'multitaper': True, 'bandwidth': 2.0},
    #     {'window_length': 1, 'resolution': 0.2, 'stride_length': 1, 'multitaper': True, 'bandwidth': 1.0},
    # ]
    all_mins = [] 
    all_maxs = []
    import copy
    comparison_args = copy.deepcopy(args)
    comparison_args.load_spec_true = True
    comparison_args.normalize_spec = False
    comparison_dataset = TUABBaselineDataset(comparison_args, mode='train', window_length=4, resolution=0.2, stride_length=1, multitaper=True)
    
    for test_case in test_cases:
        trainset = TUABBaselineDataset( **test_case)
        print(trainset[0])
        bp() 
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        aa = next(iter(trainloader))
        print(test_case)
        print('Shape of aa: ', aa[0].shape)
        for item in tqdm(trainloader):
            all_mins.append(item[0].detach().cpu().numpy().min())
            all_maxs.append(item[0].detach().cpu().numpy().max())
    print('Mean min: ', np.mean(all_mins))
    print('Mean max: ', np.mean(all_maxs))
    print('Std min: ', np.std(all_mins))
    print('Std max: ', np.std(all_maxs))
    bp() 
    print('done')
    
    ##
    # model = SpectrogramCNN(model='conv1d', num_classes=6)
    # model2 = SpectrogramCNN(model='conv2d', num_classes=6)
    # for X, Y in trainloader:
    #     bp() 
    #     output = model(X)
    #     output2 = model2(X)
    #     bp() 
    #     print(output.shape, output2.shape, Y)
    #     break
