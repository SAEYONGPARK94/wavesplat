"""
tnp_sanity.py

Stage 1.6 Sanity Test — Go/No-go #1 판단용.

목표:
  1. WaveSplatTNPGenerator → Batch 생성 OK
  2. Vanilla TNP 모델 instantiate OK
  3. Forward pass (batch.xc, yc, xt → pred dist) OK
  4. NLL loss 계산 + backward + Adam step 1회 — *학습 가능*한가
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'gridded-tnp'))

from tnp_dataset import WaveSplatTNPGenerator, _list_episode_files  # noqa: E402

from tnp.models.tnp import TNP, TNPDecoder, TNPEncoder  # noqa: E402
from tnp.networks.transformer import TNPTransformerEncoder  # noqa: E402
from tnp.networks.attention_layers import (  # noqa: E402
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from tnp.networks.mlp import MLP  # noqa: E402
from tnp.likelihoods.gaussian import HeteroscedasticNormalLikelihood  # noqa: E402


def build_vanilla_tnp(
    *,
    dim_x: int = 3,         # (x, y, t)
    dim_y: int = 1,         # η
    embed_dim: int = 64,
    num_heads: int = 4,
    head_dim: int = 16,
    num_layers: int = 3,
    feedforward_dim: int = 128,
) -> TNP:
    """Minimal vanilla TNP with self+cross attention."""
    # x, y → token embedding
    # preprocess_observations adds a +1 binary indicator channel to y → dim_y + 1
    # zc = cat(xc, yc_with_indicator) → dim_x + dim_y + 1
    xy_encoder = MLP(
        in_dim=dim_x + dim_y + 1,
        out_dim=embed_dim,
        num_layers=2,
        width=embed_dim,
    )

    # Self-attention on context, cross-attention from target to context
    mhsa = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    mhca = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    transformer = TNPTransformerEncoder(
        num_layers=num_layers,
        mhca_layer=mhca,
        mhsa_layer=mhsa,
    )

    encoder = TNPEncoder(
        transformer_encoder=transformer,
        xy_encoder=xy_encoder,
    )

    # Decoder: z → (μ, σ) for Gaussian likelihood
    z_decoder = MLP(
        in_dim=embed_dim,
        out_dim=2 * dim_y,    # mean + log_var
        num_layers=2,
        width=embed_dim,
    )
    decoder = TNPDecoder(z_decoder=z_decoder)

    likelihood = HeteroscedasticNormalLikelihood(min_noise=1e-3)

    return TNP(encoder=encoder, decoder=decoder, likelihood=likelihood)


def main():
    print('=' * 60)
    print('Stage 1.6 — Sanity test: vanilla TNP on WaveSplat')
    print('=' * 60)

    # ── 1) Dataset ──
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    eps = _list_episode_files(data_dir, 'episode_BF3_2comp_seed0_uniform.h5')
    assert len(eps) > 0, f"No episode found in {data_dir}"

    gen = WaveSplatTNPGenerator(
        episode_paths=eps,
        samples_per_epoch=8,
        batch_size=2,
        window_size=3,
        max_context=128,
        max_target=256,
    )
    batch = next(iter(gen))
    print(f'\n[Batch shapes]')
    print(f'  xc: {tuple(batch.xc.shape)}, yc: {tuple(batch.yc.shape)}')
    print(f'  xt: {tuple(batch.xt.shape)}, yt: {tuple(batch.yt.shape)}')

    # ── 2) Model ──
    model = build_vanilla_tnp(
        dim_x=3, dim_y=1,
        embed_dim=64, num_heads=4, head_dim=16,
        num_layers=2,           # tiny for CPU sanity
        feedforward_dim=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[Model] vanilla TNP, {n_params:,} parameters ({n_params/1e6:.2f}M)')

    # ── 3) Forward pass ──
    print('\n[Forward pass]')
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        pred = model(batch.xc, batch.yc, batch.xt)
        dt = time.time() - t0
    print(f'  pred dist: {type(pred).__name__}')
    print(f'  pred.mean: {tuple(pred.mean.shape)}')
    print(f'  pred.stddev range: [{pred.stddev.min():.4f}, {pred.stddev.max():.4f}]')
    print(f'  Forward time: {dt*1000:.1f} ms (CPU)')

    # ── 4) Loss + backward + step ──
    print('\n[Train step — Go/No-go #1]')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        pred = model(batch.xc, batch.yc, batch.xt)
        # NLL: -log p(yt | pred dist)
        nll = -pred.log_prob(batch.yt).mean()
        nll.backward()
        optimizer.step()
        losses.append(nll.item())
        print(f'  step {step}: NLL = {nll.item():.4f}')

    diff = losses[0] - losses[-1]
    if diff > 0:
        print(f'\n✓ Loss decreasing ({losses[0]:.4f} → {losses[-1]:.4f}, Δ={diff:.4f}) — GRADIENT FLOWS')
        print('✓ Go/No-go #1 PASSED — TNP can be trained on WaveSplat data')
    else:
        print(f'\n✗ Loss not decreasing — needs investigation')


if __name__ == '__main__':
    main()
