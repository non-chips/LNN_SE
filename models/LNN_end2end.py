"""
LNN

A simple DCCRN-style speech enhancement model with:
1. End-to-end STFT / iSTFT
2. ERB frequency compression
3. CNN encoder-decoder
4. Liquid/CfC temporal module
5. Complex ratio mask

Input:
    noisy waveform: (B, L)

Output:
    enhanced waveform: (B, L)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. ERB frequency mapping
# ============================================================

class ERB(nn.Module):
    """
    ERB band mapping.

    bm:
        full frequency bins -> compressed ERB frequency bins

    bs:
        compressed ERB frequency bins -> full frequency bins
    """

    def __init__(
        self,
        erb_subband_1=65,
        erb_subband_2=64,
        nfft=512,
        high_lim=8000,
        fs=16000,
    ):
        super().__init__()

        erb_filters = self.erb_filter_banks(
            erb_subband_1=erb_subband_1,
            erb_subband_2=erb_subband_2,
            nfft=nfft,
            high_lim=high_lim,
            fs=fs,
        )

        nfreqs = nfft // 2 + 1

        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.nfreqs = nfreqs
        self.nfreqs_erb = erb_subband_1 + erb_subband_2

        self.erb_fc = nn.Linear(
            nfreqs - erb_subband_1,
            erb_subband_2,
            bias=False,
        )

        self.ierb_fc = nn.Linear(
            erb_subband_2,
            nfreqs - erb_subband_1,
            bias=False,
        )

        self.erb_fc.weight = nn.Parameter(
            erb_filters,
            requires_grad=False,
        )

        self.ierb_fc.weight = nn.Parameter(
            erb_filters.T,
            requires_grad=False,
        )

    @staticmethod
    def hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1.0)

    @staticmethod
    def erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1.0) / 0.00437

    def erb_filter_banks(
        self,
        erb_subband_1,
        erb_subband_2,
        nfft=512,
        high_lim=8000,
        fs=16000,
    ):
        low_lim = erb_subband_1 / nfft * fs

        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)

        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)

        erb_filters = np.zeros(
            [erb_subband_2, nfft // 2 + 1],
            dtype=np.float32,
        )

        erb_filters[0, bins[0]:bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)

        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)

            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = (
            1.0 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        )

        erb_filters = erb_filters[:, erb_subband_1:]

        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """
        ERB band mapping.

        Args:
            x: (B, C, T, F)

        Returns:
            x_erb: (B, C, T, F_erb)
        """
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])

        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """
        Inverse ERB band mapping.

        Args:
            x_erb: (B, C, T, F_erb)

        Returns:
            x: (B, C, T, F)
        """
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])

        return torch.cat([x_erb_low, x_erb_high], dim=-1)


# ============================================================
# 2. CNN encoder / decoder
# ============================================================

class ConvBlock(nn.Module):
    """
    DCCRN-style encoder block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 5),
        stride=(1, 2),
        padding=(0, 2),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """
    DCCRN-style decoder block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 5),
        stride=(1, 2),
        padding=(0, 2),
        output_padding=(0, 0),
        is_last=False,
    ):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class Encoder(nn.Module):
    """
    Encoder for ERB-compressed complex features.

    Input:
        x: (B, 2, T, 129)

    Output:
        x: (B, 64, T, 17)
    """

    def __init__(self):
        super().__init__()

        self.en_convs = nn.ModuleList([
            ConvBlock(2, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        ])

    def forward(self, x):
        en_outs = []

        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)

        return x, en_outs


class Decoder(nn.Module):
    """
    Decoder for ERB-compressed mask.

    Input:
        x: (B, 64, T, 17)

    Output:
        mask_erb: (B, 2, T, 129)
    """

    def __init__(self):
        super().__init__()

        self.de_convs = nn.ModuleList([
            DeconvBlock(64 + 64, 32),
            DeconvBlock(32 + 32, 16),
            DeconvBlock(16 + 16, 2, is_last=True),
        ])

    @staticmethod
    def _match_freq(x, ref):
        target_f = ref.shape[-1]
        current_f = x.shape[-1]

        if current_f > target_f:
            x = x[..., :target_f]
        elif current_f < target_f:
            x = F.pad(x, (0, target_f - current_f))

        return x

    def forward(self, x, en_outs):
        n_layers = len(self.de_convs)

        for i in range(n_layers):
            skip = en_outs[n_layers - 1 - i]

            x = self._match_freq(x, skip)
            x = torch.cat([x, skip], dim=1)

            x = self.de_convs[i](x)

        return x


# ============================================================
# 3. LNN / CfC temporal module
# ============================================================

class LiquidCfCCell(nn.Module):
    """
    Lightweight Liquid / CfC-style recurrent cell.

    This is a simple verification-oriented recurrent unit,
    not a strict LTC ODE solver.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.candidate_layer = nn.Linear(
            input_size + hidden_size,
            hidden_size,
        )

        self.gate_layer = nn.Linear(
            input_size + hidden_size,
            hidden_size,
        )

        self.tau_layer = nn.Linear(
            input_size + hidden_size,
            hidden_size,
        )

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=-1)

        candidate = torch.tanh(self.candidate_layer(combined))
        gate = torch.sigmoid(self.gate_layer(combined))

        tau = F.softplus(self.tau_layer(combined)) + 1e-4
        decay = torch.exp(-tau)

        alpha = gate * decay

        h_t = alpha * h_prev + (1.0 - alpha) * candidate

        return h_t


class LiquidCfC(nn.Module):
    """
    Multi-step Liquid/CfC sequence module.

    Input:
        x: (B, T, input_size)

    Output:
        y: (B, T, hidden_size)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional=False,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.fw_cell = LiquidCfCCell(input_size, hidden_size)

        if bidirectional:
            self.bw_cell = LiquidCfCCell(input_size, hidden_size)
            self.out_size = hidden_size * 2
        else:
            self.bw_cell = None
            self.out_size = hidden_size

    def _run_direction(self, x, cell, reverse=False):
        B, T, _ = x.shape

        h = torch.zeros(
            B,
            cell.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        indices = range(T - 1, -1, -1) if reverse else range(T)

        for t in indices:
            h = cell(x[:, t], h)
            outputs.append(h)

        if reverse:
            outputs = outputs[::-1]

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        y_fw = self._run_direction(x, self.fw_cell, reverse=False)

        if self.bidirectional:
            y_bw = self._run_direction(x, self.bw_cell, reverse=True)
            return torch.cat([y_fw, y_bw], dim=-1)

        return y_fw


class LNNBottleneck(nn.Module):
    """
    DCCRN bottleneck temporal module.

    Input:
        x: (B, C, T, F_enc)

    Process:
        (B, C, T, F_enc)
        -> (B, T, C * F_enc)
        -> projection
        -> LNN/CfC
        -> projection back
        -> (B, C, T, F_enc)
    """

    def __init__(
        self,
        channels=64,
        freq_bins=17,
        proj_size=256,
        bidirectional=False,
    ):
        super().__init__()

        self.channels = channels
        self.freq_bins = freq_bins
        self.input_size = channels * freq_bins

        self.in_proj = nn.Linear(self.input_size, proj_size)

        self.lnn = LiquidCfC(
            input_size=proj_size,
            hidden_size=proj_size,
            bidirectional=bidirectional,
        )

        lnn_out_size = proj_size * 2 if bidirectional else proj_size

        self.out_proj = nn.Linear(lnn_out_size, self.input_size)
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x):
        B, C, T, Freq = x.shape

        residual = x

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * Freq)

        x = self.in_proj(x)
        x = self.lnn(x)
        x = self.out_proj(x)
        x = self.norm(x)

        x = x.view(B, T, C, Freq)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x + residual


# ============================================================
# 4. Complex mask
# ============================================================

class Mask(nn.Module):
    """
    Complex ratio mask.

    mask:
        (B, 2, T, F)

    spec:
        (B, 2, T, F)
    """

    def forward(self, mask, spec):
        mask_real = mask[:, 0]
        mask_imag = mask[:, 1]

        spec_real = spec[:, 0]
        spec_imag = spec[:, 1]

        enh_real = spec_real * mask_real - spec_imag * mask_imag
        enh_imag = spec_real * mask_imag + spec_imag * mask_real

        return torch.stack([enh_real, enh_imag], dim=1)


# ============================================================
# 5. Complete ERB-DCCRN-LNN model
# ============================================================

class LNN(nn.Module):
    """
    ERB-DCCRN-LNN.

    Input:
        x: noisy waveform, (B, L)

    Output:
        y: enhanced waveform, (B, L)
    """

    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        fs=16000,
        erb_subband_1=65,
        erb_subband_2=64,
        lnn_proj_size=256,
        bidirectional=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.fs = fs

        self.n_freqs = n_fft // 2 + 1
        self.n_freqs_erb = erb_subband_1 + erb_subband_2

        self.erb = ERB(
            erb_subband_1=erb_subband_1,
            erb_subband_2=erb_subband_2,
            nfft=n_fft,
            high_lim=fs // 2,
            fs=fs,
        )

        self.encoder = Encoder()

        # F_erb = 129
        # after 3 stride-2 conv layers:
        # 129 -> 65 -> 33 -> 17
        self.temporal = LNNBottleneck(
            channels=64,
            freq_bins=17,
            proj_size=lnn_proj_size,
            bidirectional=bidirectional,
        )

        self.decoder = Decoder()

        self.mask = Mask()

    @staticmethod
    def _match_freq(x, target_f):
        current_f = x.shape[-1]

        if current_f > target_f:
            x = x[..., :target_f]
        elif current_f < target_f:
            x = F.pad(x, (0, target_f - current_f))

        return x

    def forward(self, x):
        """
        Args:
            x: noisy waveform, (B, L)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L), but got {x.shape}")

        device = x.device
        dtype = x.dtype
        n_samples = x.shape[1]

        window = torch.hann_window(
            self.win_len,
            device=device,
            dtype=dtype,
        )

        stft_kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": window,
            "onesided": True,
        }

        # ----------------------------------------------------
        # STFT
        # ----------------------------------------------------
        spec = torch.stft(
            x,
            **stft_kwargs,
            return_complex=True,
        )  # (B, F, T)

        spec_ri = torch.view_as_real(spec)  # (B, F, T, 2)

        spec_for_mask = spec_ri.permute(0, 3, 2, 1).contiguous()
        # (B, 2, T, F)

        # ----------------------------------------------------
        # ERB mapping for network input
        # ----------------------------------------------------
        feat = self.erb.bm(spec_for_mask)
        # (B, 2, T, F_erb=129)

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------
        feat, en_outs = self.encoder(feat)
        # (B, 64, T, 17)

        # ----------------------------------------------------
        # LNN temporal module
        # ----------------------------------------------------
        feat = self.temporal(feat)
        # (B, 64, T, 17)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------
        mask_erb = self.decoder(feat, en_outs)
        # (B, 2, T, around 129)

        mask_erb = self._match_freq(mask_erb, self.n_freqs_erb)
        # (B, 2, T, 129)

        # ----------------------------------------------------
        # Inverse ERB mapping: mask_erb -> mask_full
        # ----------------------------------------------------
        mask = self.erb.bs(mask_erb)
        # (B, 2, T, 257)

        mask = self._match_freq(mask, self.n_freqs)

        # ----------------------------------------------------
        # Complex ratio mask
        # ----------------------------------------------------
        enhanced_spec = self.mask(mask, spec_for_mask)
        # (B, 2, T, F)

        # ----------------------------------------------------
        # iSTFT
        # ----------------------------------------------------
        enhanced_spec = enhanced_spec.permute(0, 3, 2, 1).contiguous()
        # (B, F, T, 2)

        enhanced_spec = torch.complex(
            enhanced_spec[..., 0],
            enhanced_spec[..., 1],
        )
        # (B, F, T)

        y = torch.istft(
            enhanced_spec,
            **stft_kwargs,
            length=n_samples,
        )

        return y


# ============================================================
# 6. Quick test
# ============================================================

if __name__ == "__main__":
    model = LNN(
        n_fft=512,
        hop_len=256,
        win_len=512,
        fs=16000,
        erb_subband_1=65,
        erb_subband_2=64,
        lnn_proj_size=256,
        bidirectional=False,
    ).eval()

    x = torch.randn(2, 16000)

    with torch.no_grad():
        y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e3:.2f} K")