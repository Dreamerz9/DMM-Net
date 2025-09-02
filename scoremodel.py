import torch.nn as nn
import torch
import numpy as np
import tqdm
from transformers_encoder.transformer import TransformerEncoder
from einops import rearrange
from vit import ViT

class TimeEncoding(nn.Module):
    '''用于对时间进行特定傅里叶编码'''
    def __init__(self, embed_dim, scale=30.):
        super().__init__()

        self.W = nn.Parameter(torch.randn(embed_dim//2)*scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256, 512, 1024], embed_dim=256):
        super().__init__()
        self.embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))  # 时间编码层
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False)
        self.attention_1 = TransformerEncoder(embed_dim=channels[0],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.conv2_cond = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.attention_2 = TransformerEncoder(embed_dim=channels[1],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.conv3_cond = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.attention_3 = TransformerEncoder(embed_dim=channels[2],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.conv4_cond = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.attention_4 = TransformerEncoder(embed_dim=channels[3],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.conv5_cond = nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.attention_5 = TransformerEncoder(embed_dim=channels[4],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense55 = Dense(embed_dim, channels[4])
        self.gnorm55 = nn.GroupNorm(32, num_channels=channels[4])

        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, stride=2, padding=1, bias=False)
        self.attention_6 = TransformerEncoder(embed_dim=channels[5],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.conv6_cond = nn.Conv2d(channels[4], channels[5], 3, stride=2, padding=1, bias=False)
        self.dense66 = Dense(embed_dim, channels[5])
        self.gnorm66 = nn.GroupNorm(32, num_channels=channels[5])

        self.tconv6 = nn.ConvTranspose2d(channels[5], channels[4], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv6_cond = nn.ConvTranspose2d(channels[5], channels[4], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t6 = TransformerEncoder(embed_dim=channels[4],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense65 = Dense(embed_dim, channels[4])
        self.tgnorm65 = nn.GroupNorm(32, num_channels=channels[4])

        self.tconv5 = nn.ConvTranspose2d(channels[4] + channels[4], channels[3], 3, stride=2, padding=1, bias=False,
                                         output_padding=1)
        self.tconv5_cond = nn.ConvTranspose2d(channels[4], channels[3], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t5= TransformerEncoder(embed_dim=channels[3],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense54 = Dense(embed_dim, channels[3])
        self.tgnorm54 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3] + channels[3], channels[2], 3, stride=2, padding=1, bias=False,
                                         output_padding=1)
        self.tconv4_cond = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t4= TransformerEncoder(embed_dim=channels[2],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv3_cond = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.attention_t3 = TransformerEncoder(embed_dim=channels[1],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, bias=False,
                                         output_padding=1)
        self.tconv2_cond = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t2 = TransformerEncoder(embed_dim=channels[1],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 32, 3, stride=1, padding=1)
        self.tconv1_cond = nn.ConvTranspose2d(channels[0], 32, 3, stride=1, padding=1)
        self.attention_t1 = TransformerEncoder(embed_dim=channels[0],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.convf1 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, t, condition=None):
        # 对时间t进行编码
        embed = self.act(self.embed(t))
        # 编码器部分前向计算
        h1 = self.conv1(x)
        if condition is not None:
            condition = self.conv1(condition)
            condition = self.conv2_cond(condition)  # align condition with h2
            h1_re = rearrange(h1, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h1_with_condition = self.attention_1(h1_re, condition_re, condition_re)
            h1_with_condition = rearrange(h1_with_condition, '(h w) b c -> b c h w', h=256, w=256)
            h1 = h1 + h1_with_condition

        # Incorporate information from t
        h1 += self.dense1(embed)
        # Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        if condition is not None:
            condition = self.conv2_cond(condition)  # align condition with h2
            h2_re = rearrange(h2, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h2_with_condition = self.attention_2(h2_re, condition_re, condition_re)
            h2_with_condition = rearrange(h2_with_condition, '(h w) b c -> b c h w', h=128, w=128)
            h2 = h2 + h2_with_condition


        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        if condition is not None:
            condition = self.conv3_cond(condition)  # align condition with h3   32, 128, 64, 64
            h3_re = rearrange(h3, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h3_with_condition = self.attention_3(h3_re, condition_re, condition_re)
            h3_with_condition = rearrange(h3_with_condition, '(h w) b c -> b c h w', h=64, w=64)
            h3 = h3 + h3_with_condition


        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        if condition is not None:
            condition = self.conv4_cond(condition)  # align condition with h4
            h4_re = rearrange(h4, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h4_with_condition = self.attention_4(h4_re, condition_re, condition_re)
            h4_with_condition = rearrange(h4_with_condition, '(h w) b c -> b c h w', h=32, w=32)
            h4 = h4 + h4_with_condition

        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h5 = self.conv5(h4)
        if condition is not None:
            condition = self.conv5_cond(condition)  # align condition with h4
            h5_re = rearrange(h5, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h5_with_condition = self.attention_5(h5_re, condition_re, condition_re)
            h5_with_condition = rearrange(h5_with_condition, '(h w) b c -> b c h w', h=16, w=16)
            h5 = h5 + h5_with_condition

        h5 += self.dense55(embed)  # 注入时间t
        h5 = self.gnorm55(h5)
        h5 = self.act(h5)

        h6 = self.conv6(h5)
        if condition is not None:
            condition = self.conv6_cond(condition)  # align condition with h4
            h6_re = rearrange(h6, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h6_with_condition = self.attention_6(h6_re, condition_re, condition_re)
            h6_with_condition = rearrange(h6_with_condition, '(h w) b c -> b c h w', h=8, w=8)
            h6 = h6 + h6_with_condition

        h6 += self.dense66(embed)  # 注入时间t
        h6 = self.gnorm66(h6)
        h6 = self.act(h6)

        h = self.tconv6(h6)
        if condition is not None:
            condition = self.tconv6_cond(condition)
            h_re = rearrange(h, 'b c h w -> (h w) b c')  # [4096, 32, 128]
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h_with_condition = self.attention_t6(h_re, condition_re, condition_re)
            h_with_condition = rearrange(h_with_condition, '(h w) b c -> b c h w', h=16, w=16)
            h = h + h_with_condition

        h += self.dense65(embed)  # 注入时间t
        h = self.tgnorm65(h)
        h = self.act(h)

        h = self.tconv5(torch.cat([h, h5], dim=1))  # skip connection
        if condition is not None:
            condition = self.tconv5_cond(condition)
            h_re = rearrange(h, 'b c h w -> (h w) b c')
            condition_re = rearrange(condition, 'b c h w -> (h w) b c')
            h_with_condition = self.attention_t5(h_re, condition_re, condition_re)
            h_with_condition = rearrange(h_with_condition, '(h w) b c -> b c h w', h=32, w=32)
            h = h + h_with_condition

        h += self.dense54(embed)  # 注入时间t
        h = self.tgnorm54(h)
        h = self.act(h)

        # 解码器部分前向计算
        h = self.tconv4(torch.cat([h, h4], dim=1))
        if condition is not None:
            condition = self.tconv4_cond(condition)
            h = h + condition

        h += self.dense5(embed)  # 注入时间t
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        if condition is not None:
            condition = self.tconv3_cond(condition)

            h = h + condition

        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        if condition is not None:
            condition = self.tconv2_cond(condition)
            h = h + condition

        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))
        if condition is not None:
            condition = self.tconv1_cond(condition)
            h = h + condition

        h = self.convf1(h)
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

def loss_fn(model, x, marginal_prob_std, condition=None, eps=1e-5):

    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    if condition is not None:
        perturbed_condition = condition + z * std[:, None, None, None]
        score = model(perturbed_x, random_t, perturbed_condition)
    else:
        score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2)))
    return loss

num_steps = 100
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps = num_steps,
                           device = 'cuda',
                           condition=None,
                           eps=1e-3):

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 256, 256, device=device) \
             * marginal_prob_std(t)[:, None, None, None]

    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
        # for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            if condition is not None:

                perturbed_condition = condition + torch.randn(batch_size, 1, 256, 256, device=device) * marginal_prob_std(batch_time_step)[:, None, None, None]
                mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step, perturbed_condition) * step_size
            else:
                mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    return mean_x