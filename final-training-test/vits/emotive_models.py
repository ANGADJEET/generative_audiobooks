import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import emotive_modules as modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

###############################
# Emotion and AdaIN Modules
###############################

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions, emotion_embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
    
    def forward(self, emotion_labels):
        return self.embedding(emotion_labels)  # [B, emotion_embedding_dim]

class AdaIN(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        # If style is [B, D] then unsqueeze; if already [B, D, 1], leave as is.
        if style.dim() == 2:
            style = style.unsqueeze(-1)
        style = self.fc(style)  # [B, 2*C, 1]
        gamma, beta = torch.chunk(style, 2, dim=1)
        x = self.norm(x)
        return gamma * x + beta

###############################
# Duration Predictors
###############################

class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, 
                 gin_channels=0, emotion_embedding_dim=0):
        super().__init__()
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3,
                                             gin_channels=gin_channels, 
                                             emotion_embedding_dim=emotion_embedding_dim))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, 
                                        p_dropout=p_dropout, gin_channels=gin_channels,
                                        emotion_embedding_dim=emotion_embedding_dim)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3,
                                                  gin_channels=gin_channels,
                                                  emotion_embedding_dim=emotion_embedding_dim))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, 
                                   p_dropout=p_dropout, gin_channels=gin_channels,
                                   emotion_embedding_dim=emotion_embedding_dim)
        
        # Emotion conditioning: if emotion_embedding_dim > 0, project to the conv channels.
        self.emotion_proj = None
        if emotion_embedding_dim > 0:
            self.emotion_proj = nn.Conv1d(emotion_embedding_dim, filter_channels, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0, emotion_emb=None):
        # Stop gradients from flowing back into the encoder if that is desired.
        x = x.detach()
        x = self.pre(x)
        
        # Process emotion embedding: if not already [B, D, 1], unsqueeze.
        emotion_proj = None
        if emotion_emb is not None and self.emotion_proj is not None:
            if emotion_emb.dim() == 2:
                emotion_proj = self.emotion_proj(emotion_emb.unsqueeze(-1))  # [B, D, 1]
            else:
                emotion_proj = self.emotion_proj(emotion_emb)
        
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)
        
        if emotion_proj is not None:
            x = x + emotion_proj

        x = self.convs(x, x_mask, g=g, emotion_emb=emotion_proj)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0 
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask, g=g, emotion_emb=emotion_proj)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1) 
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = x.detach()
    if g is not None:
      g = g.detach()
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

###############################
# Text Encoder and Posterior
###############################

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, 
                 n_heads, n_layers, kernel_size, p_dropout, emotion_embedding_dim=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        # Emotion conditioning: project raw emotion embedding if provided.
        self.emotion_proj = None
        if emotion_embedding_dim > 0:
            self.emotion_proj = nn.Conv1d(emotion_embedding_dim, hidden_channels, 1)

    def forward(self, x, x_lengths, emotion_emb=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)

        # Add emotion conditioning (check input dimension).
        if emotion_emb is not None and self.emotion_proj is not None:
            if emotion_emb.dim() == 2:
                emotion_proj = self.emotion_proj(emotion_emb.unsqueeze(-1))  # [B, D, 1]
            else:
                emotion_proj = self.emotion_proj(emotion_emb)
            x = x + emotion_proj

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 n_flows=4, gin_channels=0, emotion_embedding_dim=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, emotion_embedding_dim=emotion_embedding_dim,
                mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, emotion_emb=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, emotion_emb=emotion_emb, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, emotion_emb=emotion_emb, reverse=reverse)
        return x

class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask

###############################
# Generator and Discriminators
###############################

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, 
                 gin_channels=0, emotion_embedding_dim=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), 
                              upsample_initial_channel//(2**(i+1)),
                              k, u, padding=(k-u)//2)))

        # Emotion conditioning with AdaIN
        self.adain_layers = nn.ModuleList()
        if emotion_embedding_dim > 0:
            for i in range(len(self.ups)):
                ch = upsample_initial_channel//(2**(i+1))
                self.adain_layers.append(AdaIN(emotion_embedding_dim, ch))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None, emotion_emb=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            
            # Apply emotion conditioning
            if emotion_emb is not None and len(self.adain_layers) > 0:
                x = self.adain_layers[i](x, emotion_emb)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

###############################
# Synthesizer (Training/Inference)
###############################

class SynthesizerTrn(nn.Module):
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels,
                 hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
                 p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_speakers=0, gin_channels=0, use_sdp=True, num_emotions=0,
                 emotion_embedding_dim=128):
        super().__init__()
        self.num_emotions = num_emotions
        self.emotion_embedding_dim = emotion_embedding_dim
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.segment_size = segment_size  # Ensure segment_size is stored.

        if num_emotions > 0:
            self.emb_e = EmotionEmbedding(num_emotions, emotion_embedding_dim)
        
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                               filter_channels, n_heads, n_layers, kernel_size,
                               p_dropout, emotion_embedding_dim=emotion_embedding_dim if num_emotions > 0 else 0)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                           resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                           upsample_kernel_sizes, gin_channels=gin_channels,
                           emotion_embedding_dim=emotion_embedding_dim if num_emotions > 0 else 0)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, 
                                        gin_channels=gin_channels, emotion_embedding_dim=emotion_embedding_dim)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4,
                                                gin_channels=gin_channels,
                                                emotion_embedding_dim=emotion_embedding_dim if num_emotions > 0 else 0)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None, emotion_labels=None):
        emotion_emb = None
        if self.num_emotions > 0 and emotion_labels is not None:
            emotion_emb = self.emb_e(emotion_labels)

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, emotion_emb=emotion_emb)
        
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g, emotion_emb=emotion_emb)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            # Transpose the matmul outputs so that neg_cent has shape [B, T_text, T_audio]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r).transpose(1,2)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)).transpose(1,2)
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g, emotion_emb=emotion_emb)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g, emotion_emb=emotion_emb)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, emotion_labels=None, noise_scale=1, 
             length_scale=1, noise_scale_w=1., max_len=None):
        emotion_emb = None
        if self.num_emotions > 0 and emotion_labels is not None:
            emotion_emb = self.emb_e(emotion_labels)
        
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, emotion_emb=emotion_emb)
        
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w, emotion_emb=emotion_emb)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, emotion_emb=emotion_emb, reverse=True)
        o = self.dec((z * y_mask)[:,:,:max_len], g=g, emotion_emb=emotion_emb)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, emotion_src=None, emotion_tgt=None):
        assert self.n_speakers > 0, "n_speakers must be > 0"
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        
        emotion_emb_src = None
        if self.num_emotions > 0 and emotion_src is not None:
            emotion_emb_src = self.emb_e(emotion_src)
            
        emotion_emb_tgt = None
        if self.num_emotions > 0 and emotion_tgt is not None:
            emotion_emb_tgt = self.emb_e(emotion_tgt)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src, emotion_emb=emotion_emb_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, emotion_emb=emotion_emb_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt, emotion_emb=emotion_emb_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
