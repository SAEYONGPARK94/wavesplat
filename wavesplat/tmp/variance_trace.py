"""
variance_trace.py -- forensic trace of the EXACT divergence point between a good seed (42,
nRMSE 0.0524) and a bad seed (1, nRMSE 0.1682) on data/episode_BF3_2comp_seed0_uniform.h5.

This is deliberately NOT another knob-ablation. It re-runs the shipped default pipeline
UNCHANGED (same config as tmp/estimator_variance.py) and instruments it to snapshot the
model's full parameter state at every 10 iterations, at every coarse-to-fine advance event,
at the prune/grow decision, and at every tracking frame -- then diffs the two seeds' traces
directly.

Does NOT flip GROW_RANDOM_PHI / KONLY_RANDOM_WPHI / PAD_W (all left at their shipped
defaults: True / True / 0.01). Does NOT touch tmp/grow_phase_fix*. Adds zero RNG consumption
of its own (snapshot function only reads .detach() tensors) so it must reproduce the
already-recorded nRMSE values bit-for-bit -- this is checked explicitly below as a self-test.
"""
import sys, os, io, json, time, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import h5py

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.optimizer.coarse_to_fine import CoarseToFineSchedule
from wavesplat.optimizer.initializer import initialize_from_frames
from wavesplat.evaluation.metrics import compute_all_metrics, rmse, nrmse
from wavesplat.run import DEFAULT_CONFIG, set_seed

EPISODE = 'data/episode_BF3_2comp_seed0_uniform.h5'
SEEDS = [42, 1, 3]     # good, bad, and the "medium" seed that exposed the obs_loss inversion
OUT_JSON = 'tmp/variance_trace_data.json'

CFG_BASE = {
    'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
    'adaptive_n': True, 'tracking_window': 3,
}

with h5py.File(EPISODE, 'r') as f:
    TRUE_COMPONENTS = f['config/wave_components'][:].tolist()  # [A, omega, kx, ky, phi, Q]


class TracedUpdater(SequentialUpdater):
    """Subclass that duplicates SequentialUpdater.initialize()'s control flow exactly
    (same calls, same order) and only ADDS read-only snapshots in between -- no new
    randomness, no altered arithmetic."""

    def __init__(self, config):
        super().__init__(config)
        self.trace = []
        self.events = []

    def _snap(self, tag, **extra):
        with torch.no_grad():
            w = self.model.w.detach().cpu().numpy().copy()
            kx = self.model.kx.detach().cpu().numpy().copy()
            ky = self.model.ky.detach().cpu().numpy().copy()
            phi = self.model.phi.detach().cpu().numpy().copy()
        rec = {'tag': tag, 'n': int(self.model.n_gaussians),
               'w': w.tolist(), 'kx': kx.tolist(), 'ky': ky.tolist(), 'phi': phi.tolist()}
        rec.update(extra)
        self.trace.append(rec)

    def initialize(self, frames, x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                   verbose=False):
        t0 = time.time()
        loss_history = []

        default_thresholds = [0.1, 0.05, None] if len(self.stages) == 3 else [None]
        default_max_iters = [200, 200, 500] if len(self.stages) == 3 else [500]
        schedule = CoarseToFineSchedule(
            stages=self.stages,
            loss_thresholds=self.loss_thresholds or default_thresholds,
            max_iters_per_stage=self.max_iters_per_stage or default_max_iters,
            init_method=self.init_method,
            depth=self.depth,
        )

        n_init = schedule.current_n()
        init_kx, init_ky, init_w, init_phi = initialize_from_frames(
            frames, n_init, self.depth, x_range, y_range,
            init_method=self.init_method, seed=self.seed
        )
        self.events.append({'tag': 'init_beamform', 'n': n_init,
                             'kx': init_kx.tolist(), 'ky': init_ky.tolist(),
                             'w': init_w.tolist(), 'phi': init_phi.tolist()})

        self.model = self._build_model(n_init)
        with torch.no_grad():
            self.model.kx.copy_(init_kx.to(self.device))
            self.model.ky.copy_(init_ky.to(self.device))
            self.model.w.copy_(init_w.to(self.device))
            self.model.phi.copy_(init_phi.to(self.device))
        self._build_optimizer()
        self._snap('post_init', iter=0, stage=0, loss=None)

        stage_iters = 0
        total_iters = 0
        max_total = sum(schedule.max_iters)

        for iteration in range(max_total):
            self.optimizer.zero_grad()
            loss = self._compute_loss(frames)
            loss.backward()
            self.optimizer.step()
            self.model.clamp_steepness()

            loss_val = loss.item()
            loss_history.append(loss_val)
            stage_iters += 1
            total_iters += 1

            if iteration % 10 == 0:
                self._snap('iter', iter=iteration, stage=schedule.current_stage, loss=loss_val)

            if schedule.should_advance(loss_val, stage_iters):
                pre_n = self.model.n_gaussians
                schedule.advance(
                    self.model, frames, self._compute_residuals,
                    x_range, y_range
                )
                post_n = self.model.n_gaussians
                with torch.no_grad():
                    new_kx = self.model.kx.detach().numpy()[pre_n:].tolist()
                    new_ky = self.model.ky.detach().numpy()[pre_n:].tolist()
                    new_w = self.model.w.detach().numpy()[pre_n:].tolist()
                    new_phi = self.model.phi.detach().numpy()[pre_n:].tolist()
                self.events.append({'tag': 'advance', 'iter': iteration, 'pre_n': pre_n,
                                     'post_n': post_n, 'new_kx': new_kx, 'new_ky': new_ky,
                                     'new_w': new_w, 'new_phi': new_phi})
                self._build_optimizer()
                stage_iters = 0
                self._snap('post_advance', iter=iteration, stage=schedule.current_stage)
                if not schedule.has_next():
                    continue

        elapsed = time.time() - t0
        if frames:
            self._last_t = frames[-1].t

        self._snap('pre_prune', iter=total_iters, stage='final', loss=loss_history[-1])

        if self.adaptive_n:
            with torch.no_grad():
                energy = self.model.w ** 2
                total_energy = energy.sum().item()
                sort_idx = torch.argsort(energy, descending=True)
                energies_sorted = energy[sort_idx].detach().cpu().numpy().tolist()
                kx_sorted = self.model.kx.detach().cpu().numpy()[sort_idx.numpy()].tolist()
                ky_sorted = self.model.ky.detach().cpu().numpy()[sort_idx.numpy()].tolist()
            self.events.append({
                'tag': 'prune_energy_snapshot',
                'total_energy': total_energy,
                'energies_sorted_desc': energies_sorted,
                'frac_sorted_desc': (np.array(energies_sorted) / total_energy).tolist(),
                'kx_sorted_desc': kx_sorted, 'ky_sorted_desc': ky_sorted,
            })
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._adaptive_prune_grow(frames, x_range, y_range, True)
            self.events.append({'tag': 'prune_grow_log', 'text': buf.getvalue()})

        self._snap('post_prune_grow', iter=total_iters, stage='final')
        self._build_tracking_optimizer()

        return {
            'loss_history': loss_history,
            'n_iters': total_iters,
            'elapsed_s': elapsed,
            'final_n': self.model.n_gaussians,
        }


def run_traced(episode, seed, cfg_extra):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(cfg_extra)
    cfg['seed'] = seed
    if seed is not None:
        set_seed(seed)

    loader = EpisodeDataLoader(episode, device=cfg['device'])
    cfg['depth'] = loader.depth

    init_frames = loader.get_init_window(tp_factor=2.0)
    updater = TracedUpdater(cfg)
    init_result = updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=False,
    )

    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    last_init = init_frames[-1]
    pred_init = updater.predict_field(grid_x, grid_y, last_init.t)
    init_rmse = rmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy())
    init_nrmse = nrmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy(), loader.Hs_true)

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    frame_trace = []
    for i, frame in enumerate(tracking_frames):
        update_result = updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()
        gt_np = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(
            pred_np, gt_np, grid_x.cpu().numpy(), grid_y.cpu().numpy(), loader.Hs_true,
            obs_x=frame.x_obs.cpu().numpy() if len(frame.x_obs) > 0 else None,
            obs_y=frame.y_obs.cpu().numpy() if len(frame.y_obs) > 0 else None,
        )
        rec = {'i': i, 't': frame.t, 'nRMSE': m.get('nRMSE'), 'obs_loss': update_result['loss']}
        if i % 10 == 0 or i == len(tracking_frames) - 1:
            with torch.no_grad():
                rec['w'] = updater.model.w.detach().cpu().numpy().tolist()
                rec['phi'] = updater.model.phi.detach().cpu().numpy().tolist()
        frame_trace.append(rec)

    avg_nrmse = float(np.mean([r['nRMSE'] for r in frame_trace]))
    avg_obs_loss = float(np.mean([r['obs_loss'] for r in frame_trace]))

    final_params = updater.model.get_spectrum_params()

    return {
        'seed': seed,
        'init_result': {k: v for k, v in init_result.items() if k != 'loss_history'},
        'loss_history_len': len(init_result['loss_history']),
        'loss_history_head': init_result['loss_history'][:5],
        'loss_history_tail': init_result['loss_history'][-5:],
        'init_rmse': init_rmse, 'init_nrmse': init_nrmse,
        'avg_nRMSE': avg_nrmse, 'avg_obs_loss': avg_obs_loss,
        'events': updater.events,
        'init_trace': updater.trace,
        'frame_trace': frame_trace,
        'final_params': {k: v.tolist() for k, v in final_params.items()},
    }


def main():
    results = {}
    for s in SEEDS:
        print(f'--- running seed={s} ---', flush=True)
        t0 = time.time()
        results[str(s)] = run_traced(EPISODE, s, CFG_BASE)
        print(f'  avg_nRMSE={results[str(s)]["avg_nRMSE"]:.4f} '
              f'avg_obs_loss={results[str(s)]["avg_obs_loss"]:.6f} '
              f'final_n={results[str(s)]["final_params"]["w"].__len__() if False else len(results[str(s)]["final_params"]["w"])} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    results['_true_components'] = TRUE_COMPONENTS
    os.makedirs('tmp', exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f)
    print(f'Wrote {OUT_JSON}')

    # Self-test: confirm bit-exact reproduction of the previously recorded values
    prev = json.load(open('tmp/estimator_variance_results.json'))
    key = os.path.basename(EPISODE)
    for s in SEEDS:
        prev_v = prev[key][str(s)]
        new_v = results[str(s)]['avg_nRMSE']
        match = abs(prev_v - new_v) < 1e-12
        print(f'  self-test seed={s}: prev={prev_v:.10f} new={new_v:.10f} '
              f'match={match}')


if __name__ == '__main__':
    main()
