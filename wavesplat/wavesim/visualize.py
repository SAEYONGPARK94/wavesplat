"""
visualize.py
WaveSim 시각화 — 파 표면 + LiDAR 관측 포인트

출력:
  - HTML 파일 (브라우저에서 인터랙티브 뷰)
  - PNG 파일 (VSCode에서 바로 미리보기)

창 구성:
  좌상: 3D 파 표면 + LiDAR 포인트 (회전/줌 가능)
  우상: 상공 2D 파고 컬러맵 + 포인트 (색상=신뢰도)
  좌하: 시간별 LiDAR 관측 밀도
  우하: GT Z vs 관측 Z 산점도 (색상=입사각)

실행:
  python3 -m wavesim.visualize                          # 새 데이터 생성
  python3 -m wavesim.visualize --file data/xxx.h5       # 기존 파일 로드
  python3 -m wavesim.visualize --animate                # 애니메이션 HTML
  python3 -m wavesim.visualize --save_png result.png    # PNG 저장
"""

import numpy as np
import argparse
import os
import sys
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
BG = '#0D1117'
PANEL = '#161B22'
GRID_C = '#30363D'
TEXT = '#E6EDF3'
ACCENT = '#58A6FF'
LIDAR_C = '#00FF88'
SENSOR_C = '#FF6B6B'


def _get_lidar_points(data: dict, frame_idx: int):
    valid = data['lidar_valid_mask'][frame_idx]
    pts = data['lidar_points'][frame_idx][valid]
    conf = data['lidar_confidence'][frame_idx][valid]
    theta = data['lidar_incidence_angles'][frame_idx][valid]
    return pts, conf, theta


def plot_static_html(data: dict, frame_idx: int = 0,
                     save_html: str = None,
                     save_png: str = None,
                     open_browser: bool = True):
    """
    4-패널 정적 시각화를 HTML + PNG로 저장.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.interpolate import RegularGridInterpolator

    gt_x = data['gt_grid_x']
    gt_y = data['gt_grid_y']
    Z = data['gt_wave_field'][frame_idx]
    t = data['timestamps'][frame_idx]
    ts = data['timestamps']

    pts, conf, theta_arr = _get_lidar_points(data, frame_idx)
    X2d, Y2d = np.meshgrid(gt_x, gt_y)

    # ── 서브플롯 레이아웃 ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'xy'},   {'type': 'xy'}]],
        subplot_titles=[
            f'3D 파 표면 + LiDAR 포인트  (t={t:.1f}s)',
            '상공 뷰 — 파고 컬러맵 + LiDAR (색상=신뢰도)',
            '시간별 LiDAR 관측 밀도',
            'GT Z vs 관측 Z  (색상=입사각 [°])'
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # ── Panel 1: 3D 표면 ──────────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=gt_x, y=gt_y, z=Z,
        colorscale='RdYlBu_r',
        reversescale=False,
        opacity=0.88,
        showscale=True,
        colorbar=dict(x=0.45, len=0.45,
                      title=dict(text='파고 [m]', font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT)),
        name='Wave Surface',
        hovertemplate='x=%{x:.1f}m<br>y=%{y:.1f}m<br>z=%{z:.3f}m<extra></extra>'
    ), row=1, col=1)

    if len(pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=4, color=LIDAR_C,
                        line=dict(color='#00AA55', width=0.5)),
            name=f'LiDAR ({len(pts)} pts)',
            hovertemplate='x=%{x:.1f}m<br>y=%{y:.1f}m<br>z=%{z:.3f}m<extra>LiDAR</extra>'
        ), row=1, col=1)

    # 센서 위치
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[5.0],
        mode='markers',
        marker=dict(size=8, color=SENSOR_C, symbol='diamond'),
        name='Sensor (h=5m)',
        hovertemplate='Sensor<br>h=5m<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: 2D 상공 뷰 ──────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        x=gt_x, y=gt_y, z=Z,
        colorscale='RdYlBu_r',
        reversescale=False,
        showscale=True,
        colorbar=dict(x=1.02, len=0.45, y=0.77,
                      title=dict(text='파고 [m]', font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT)),
        hovertemplate='x=%{x:.1f}m<br>y=%{y:.1f}m<br>z=%{z:.3f}m<extra></extra>'
    ), row=1, col=2)

    if len(pts) > 0:
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode='markers',
            marker=dict(
                size=7,
                color=conf,
                colorscale='Plasma',
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(x=1.02, len=0.45, y=0.25,
                              title=dict(text='신뢰도', font=dict(color=TEXT)),
                              tickfont=dict(color=TEXT)),
                line=dict(color='#00AA55', width=0.5)
            ),
            name='LiDAR (색=신뢰도)',
            hovertemplate='x=%{x:.1f}m<br>y=%{y:.1f}m<br>conf=%{marker.color:.2f}<extra>LiDAR</extra>'
        ), row=1, col=2)

    # 센서
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=12, color=SENSOR_C, symbol='triangle-up'),
        name='Sensor', showlegend=False,
        hovertemplate='Sensor<extra></extra>'
    ), row=1, col=2)

    # ── Panel 3: 시간별 관측 밀도 ─────────────────────────────────────────
    n_pts_list = [data['lidar_valid_mask'][i].sum() for i in range(len(ts))]

    fig.add_trace(go.Scatter(
        x=ts, y=n_pts_list,
        mode='lines+markers',
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5, color=ACCENT),
        fill='tozeroy', fillcolor=f'rgba(88,166,255,0.15)',
        name='LiDAR pts/frame',
        hovertemplate='t=%{x:.2f}s<br>pts=%{y}<extra></extra>'
    ), row=2, col=1)

    # 현재 프레임 마커 (add_vline은 mixed-type subplot에서 오류 → add_shape 사용)
    fig.add_shape(
        type="line",
        x0=t, x1=t, y0=0, y1=1,
        xref="x2", yref="y2 domain",
        line=dict(width=2, dash='dash', color=SENSOR_C)
    )
    fig.add_annotation(
        x=t, y=1.02, xref="x2", yref="y2 domain",
        text=f't={t:.1f}s', showarrow=False,
        font=dict(color=TEXT, size=11)
    )

    # ── Panel 4: GT Z vs 관측 Z ──────────────────────────────────────────
    if len(pts) > 0:
        interp = RegularGridInterpolator(
            (gt_y, gt_x), Z, bounds_error=False, fill_value=np.nan
        )
        in_range = ((pts[:, 0] >= gt_x.min()) & (pts[:, 0] <= gt_x.max()) &
                    (pts[:, 1] >= gt_y.min()) & (pts[:, 1] <= gt_y.max()))
        pts_in = pts[in_range]

        if len(pts_in) > 0:
            z_gt = interp(pts_in[:, [1, 0]])
            z_obs = pts_in[:, 2]
            theta_in = np.rad2deg(theta_arr[in_range])
            valid_v = ~np.isnan(z_gt)

            rmse = np.sqrt(np.mean((z_gt[valid_v] - z_obs[valid_v])**2))

            fig.add_trace(go.Scatter(
                x=z_gt[valid_v], y=z_obs[valid_v],
                mode='markers',
                marker=dict(
                    size=7,
                    color=theta_in[valid_v],
                    colorscale='Plasma',
                    cmin=60, cmax=90,
                    showscale=True,
                    colorbar=dict(x=1.02, len=0.45, y=0.25,
                                  title=dict(text='입사각 [°]', font=dict(color=TEXT)),
                                  tickfont=dict(color=TEXT)),
                    opacity=0.8
                ),
                name='관측 포인트',
                hovertemplate='GT z=%{x:.3f}m<br>obs z=%{y:.3f}m<br>θ=%{marker.color:.1f}°<extra></extra>'
            ), row=2, col=2)

            # 1:1 선
            z_all = np.concatenate([z_gt[valid_v], z_obs[valid_v]])
            zl = [z_all.min() - 0.05, z_all.max() + 0.05]
            fig.add_trace(go.Scatter(
                x=zl, y=zl,
                mode='lines',
                line=dict(color=SENSOR_C, dash='dash', width=1.5),
                name='완벽 일치',
                hoverinfo='skip'
            ), row=2, col=2)

            # RMSE annotation
            fig.add_annotation(
                x=0.05, y=0.93, xref='x4 domain', yref='y4 domain',
                text=f'RMSE = {rmse:.3f} m',
                showarrow=False, font=dict(color=ACCENT, size=11),
                bgcolor='rgba(13,17,23,0.7)'
            )

    # ── 레이아웃 ─────────────────────────────────────────────────────────
    sea_state = data.get('sea_state', 'BF4')
    Hs_true = data.get('Hs_true', data.get('Hs', 0))
    n_frames = len(ts)
    n_pts_cur = len(pts)

    fig.update_layout(
        title=dict(
            text=(f'WaveSim Visualization  —  {sea_state}  |  '
                  f'Hs={Hs_true:.2f}m  |  {n_frames} frames  |  '
                  f'{n_pts_cur} pts (현재 프레임)'),
            font=dict(color=TEXT, size=14),
            x=0.5
        ),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        legend=dict(
            bgcolor='rgba(22,27,34,0.8)',
            bordercolor=GRID_C,
            font=dict(color=TEXT)
        ),
        height=820,
        width=1300,
        scene=dict(
            bgcolor=PANEL,
            xaxis=dict(title='X [m]', gridcolor=GRID_C, color=TEXT),
            yaxis=dict(title='Y [m]', gridcolor=GRID_C, color=TEXT),
            zaxis=dict(title='Z [m]', gridcolor=GRID_C, color=TEXT),
            camera=dict(eye=dict(x=-1.2, y=-1.5, z=1.0))
        ),
    )

    # 각 2D 패널 축 스타일
    for r, c in [(1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(gridcolor=GRID_C, zerolinecolor=GRID_C,
                         row=r, col=c)
        fig.update_yaxes(gridcolor=GRID_C, zerolinecolor=GRID_C,
                         row=r, col=c)

    fig.update_xaxes(title_text='X [m] (전방)', row=1, col=2)
    fig.update_yaxes(title_text='Y [m]', row=1, col=2)
    fig.update_xaxes(title_text='시간 [s]', row=2, col=1)
    fig.update_yaxes(title_text='LiDAR 포인트 수', row=2, col=1)
    fig.update_xaxes(title_text='GT 파고 [m]', row=2, col=2)
    fig.update_yaxes(title_text='관측 Z [m]', row=2, col=2)

    # ── 저장 ──────────────────────────────────────────────────────────────
    if save_html is None:
        save_html = 'wavesim_visualization.html'

    fig.write_html(save_html, include_plotlyjs='cdn',
                   config={'scrollZoom': True,
                           'displaylogo': False,
                           'modeBarButtonsToAdd': ['toggleSpikelines']})
    print(f"[HTML] {os.path.abspath(save_html)}")

    if save_png:
        try:
            fig.write_image(save_png, scale=1.5)
            print(f"[PNG]  {os.path.abspath(save_png)}")
        except Exception as e:
            # kaleido가 없을 때 matplotlib으로 fallback
            _save_png_matplotlib(data, frame_idx, save_png)

    if open_browser:
        webbrowser.open(f'file://{os.path.abspath(save_html)}')

    return fig


def plot_animation_html(data: dict, save_html: str = None,
                         open_browser: bool = True):
    """
    슬라이더로 시간을 탐색하는 인터랙티브 애니메이션 HTML.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gt_x = data['gt_grid_x']
    gt_y = data['gt_grid_y']
    ts = data['timestamps']
    n_frames = len(ts)

    sea_state = data.get('sea_state', 'BF4')
    Hs_true = data.get('Hs_true', data.get('Hs', 0))

    # 사전 계산
    n_pts_list = [data['lidar_valid_mask'][i].sum() for i in range(n_frames)]

    # Frames 생성
    frames = []
    for fi in range(n_frames):
        Z = data['gt_wave_field'][fi]
        pts, conf, _ = _get_lidar_points(data, fi)
        t = ts[fi]

        frame_data = [
            go.Heatmap(x=gt_x, y=gt_y, z=Z,
                       colorscale='RdYlBu_r', reversescale=False,
                       showscale=False),
        ]
        if len(pts) > 0:
            frame_data.append(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode='markers',
                marker=dict(size=7, color=LIDAR_C,
                            line=dict(color='#00AA55', width=0.4)),
            ))
        else:
            frame_data.append(go.Scatter(x=[], y=[], mode='markers'))

        frames.append(go.Frame(data=frame_data, name=str(fi),
                                layout=go.Layout(
                                    title_text=(
                                        f'WaveSim  —  {sea_state}  |  '
                                        f't = {t:.2f}s  |  {n_pts_list[fi]} pts'
                                    )
                                )))

    # 초기 프레임 데이터
    Z0 = data['gt_wave_field'][0]
    pts0, conf0, _ = _get_lidar_points(data, 0)

    fig = go.Figure(
        data=[
            go.Heatmap(x=gt_x, y=gt_y, z=Z0,
                       colorscale='RdYlBu_r', reversescale=False,
                       showscale=True,
                       colorbar=dict(title=dict(text='파고 [m]', font=dict(color=TEXT)),
                                     tickfont=dict(color=TEXT))),
            go.Scatter(
                x=pts0[:, 0] if len(pts0) > 0 else [],
                y=pts0[:, 1] if len(pts0) > 0 else [],
                mode='markers',
                marker=dict(size=7, color=LIDAR_C,
                            line=dict(color='#00AA55', width=0.4)),
                name='LiDAR',
            ),
            go.Scatter(x=[0], y=[0], mode='markers',
                       marker=dict(size=12, color=SENSOR_C, symbol='triangle-up'),
                       name='Sensor'),
        ],
        frames=frames
    )

    # 슬라이더 + 재생 버튼
    sliders = [{
        'active': 0,
        'currentvalue': {'prefix': 'Frame: ', 'font': {'color': TEXT}},
        'pad': {'b': 10},
        'bgcolor': PANEL,
        'bordercolor': GRID_C,
        'font': {'color': TEXT},
        'steps': [
            {'args': [[str(fi)],
                      {'frame': {'duration': 0, 'redraw': True},
                       'mode': 'immediate'}],
             'label': f'{ts[fi]:.1f}s',
             'method': 'animate'}
            for fi in range(n_frames)
        ]
    }]

    fig.update_layout(
        title=dict(
            text=(f'WaveSim Animation  —  {sea_state}  |  '
                  f'Hs={Hs_true:.2f}m  |  {n_frames} frames'),
            font=dict(color=TEXT, size=13), x=0.5
        ),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        xaxis=dict(title='X [m] (전방)', gridcolor=GRID_C, color=TEXT),
        yaxis=dict(title='Y [m]', gridcolor=GRID_C, color=TEXT,
                   scaleanchor='x', scaleratio=1),
        legend=dict(bgcolor='rgba(22,27,34,0.8)', bordercolor=GRID_C,
                    font=dict(color=TEXT)),
        height=680, width=1000,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'bgcolor': PANEL,
            'font': {'color': TEXT},
            'buttons': [
                {'label': '▶ 재생',
                 'method': 'animate',
                 'args': [None, {'frame': {'duration': 200, 'redraw': True},
                                  'fromcurrent': True}]},
                {'label': '⏸ 정지',
                 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0},
                                    'mode': 'immediate'}]},
            ],
            'x': 0.0, 'y': -0.08
        }],
        sliders=sliders
    )

    if save_html is None:
        save_html = 'wavesim_animation.html'

    fig.write_html(save_html, include_plotlyjs='cdn',
                   config={'scrollZoom': True, 'displaylogo': False})
    print(f"[HTML] {os.path.abspath(save_html)}")

    if open_browser:
        webbrowser.open(f'file://{os.path.abspath(save_html)}')

    return fig


def _save_png_matplotlib(data: dict, frame_idx: int, path: str):
    """Plotly kaleido 없을 때 matplotlib으로 PNG 저장."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D

    gt_x = data['gt_grid_x']
    gt_y = data['gt_grid_y']
    Z = data['gt_wave_field'][frame_idx]
    t = data['timestamps'][frame_idx]
    ts = data['timestamps']
    pts, conf, theta_arr = _get_lidar_points(data, frame_idx)
    X2d, Y2d = np.meshgrid(gt_x, gt_y)
    vabs = max(abs(Z.min()), abs(Z.max())) or 1.0

    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': PANEL,
        'axes.edgecolor': GRID_C,
        'axes.labelcolor': TEXT,
        'axes.titlecolor': TEXT,
        'xtick.color': TEXT,
        'ytick.color': TEXT,
        'text.color': TEXT,
        'font.size': 9,
    })

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
                  left=0.07, right=0.97, top=0.92, bottom=0.08)

    # 3D
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor(PANEL)
    surf = ax1.plot_surface(X2d, Y2d, Z, cmap='RdYlBu_r',
                             vmin=-vabs, vmax=vabs, alpha=0.85,
                             linewidth=0, rcount=35, ccount=35)
    if len(pts) > 0:
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=LIDAR_C, s=16, zorder=5)
    ax1.scatter([0], [0], [5.0], c=SENSOR_C, s=60, marker='^', zorder=6)
    ax1.set_xlabel('X [m]', fontsize=8, labelpad=5)
    ax1.set_ylabel('Y [m]', fontsize=8, labelpad=5)
    ax1.set_zlabel('Z [m]', fontsize=8, labelpad=3)
    ax1.set_title(f'3D Wave + LiDAR  (t={t:.1f}s)',
                  color=TEXT, fontsize=10, fontweight='bold')
    ax1.view_init(elev=25, azim=-50)
    ax1.tick_params(colors=TEXT, labelsize=7)
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_C)
    fig.colorbar(surf, ax=ax1, shrink=0.45, pad=0.09).set_label('파고 [m]', color=TEXT)

    # 2D 상공
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.pcolormesh(X2d, Y2d, Z, cmap='RdYlBu_r',
                         vmin=-vabs, vmax=vabs, shading='gouraud')
    if len(pts) > 0:
        sc = ax2.scatter(pts[:, 0], pts[:, 1], c=conf, cmap='plasma',
                         s=20, vmin=0, vmax=1, zorder=5,
                         edgecolors='#00AA55', linewidths=0.4)
        fig.colorbar(sc, ax=ax2, shrink=0.8).set_label('신뢰도', color=TEXT)
    ax2.plot(0, 0, 'r^', ms=9, zorder=6)
    ax2.set_xlabel('X [m]', fontsize=9)
    ax2.set_ylabel('Y [m]', fontsize=9)
    ax2.set_title('상공 뷰 (색상=신뢰도)', color=TEXT, fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.25)
    fig.colorbar(im, ax=ax2, shrink=0.8).set_label('파고 [m]', color=TEXT)

    # 시간별 밀도
    ax3 = fig.add_subplot(gs[1, 0])
    n_pts_list = [data['lidar_valid_mask'][i].sum() for i in range(len(ts))]
    ax3.fill_between(ts, n_pts_list, alpha=0.3, color=ACCENT)
    ax3.plot(ts, n_pts_list, color=ACCENT, lw=1.8, marker='o', ms=4)
    ax3.axvline(t, color=SENSOR_C, lw=1.8, ls='--', label=f't={t:.1f}s')
    ax3.set_xlabel('시간 [s]', fontsize=9)
    ax3.set_ylabel('LiDAR 포인트 수', fontsize=9)
    ax3.set_title('시간별 LiDAR 관측 밀도', color=TEXT, fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8, facecolor=PANEL)

    # GT vs obs scatter
    ax4 = fig.add_subplot(gs[1, 1])
    if len(pts) > 0:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((gt_y, gt_x), Z,
                                          bounds_error=False, fill_value=np.nan)
        in_range = ((pts[:, 0] >= gt_x.min()) & (pts[:, 0] <= gt_x.max()) &
                    (pts[:, 1] >= gt_y.min()) & (pts[:, 1] <= gt_y.max()))
        pts_in = pts[in_range]
        if len(pts_in) > 0:
            z_gt = interp(pts_in[:, [1, 0]])
            z_obs = pts_in[:, 2]
            theta_in = np.rad2deg(theta_arr[in_range])
            vv = ~np.isnan(z_gt)
            sc2 = ax4.scatter(z_gt[vv], z_obs[vv], c=theta_in[vv],
                               cmap='plasma', s=25, alpha=0.75,
                               vmin=60, vmax=90, edgecolors='none')
            fig.colorbar(sc2, ax=ax4, shrink=0.8).set_label('입사각 [°]', color=TEXT)
            z_all = np.concatenate([z_gt[vv], z_obs[vv]])
            zl = [z_all.min() - 0.05, z_all.max() + 0.05]
            ax4.plot(zl, zl, '--', color=SENSOR_C, lw=1.2, label='1:1')
            rmse = np.sqrt(np.mean((z_gt[vv] - z_obs[vv])**2))
            ax4.text(0.05, 0.92, f'RMSE={rmse:.3f}m', transform=ax4.transAxes,
                     color=ACCENT, fontsize=9, fontweight='bold')
            ax4.legend(fontsize=8, facecolor=PANEL)
    ax4.set_xlabel('GT 파고 [m]', fontsize=9)
    ax4.set_ylabel('관측 Z [m]', fontsize=9)
    ax4.set_title('관측 Z vs GT Z (색상=입사각)', color=TEXT, fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.25)

    sea_state = data.get('sea_state', 'BF4')
    Hs_true = data.get('Hs_true', data.get('Hs', 0))
    fig.suptitle(
        f'WaveSim  —  {sea_state}  |  Hs={Hs_true:.2f}m  |  {len(ts)} frames  |  {len(pts)} pts/frame',
        color=TEXT, fontsize=12, fontweight='bold'
    )

    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"[PNG]  {os.path.abspath(path)}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='WaveSim 시각화')
    parser.add_argument('--file', default=None, help='HDF5 에피소드 파일 경로')
    parser.add_argument('--animate', action='store_true', help='슬라이더 애니메이션 모드')
    parser.add_argument('--frame', type=int, default=0, help='정적 뷰 프레임 인덱스')
    parser.add_argument('--save_html', default=None, help='HTML 저장 경로')
    parser.add_argument('--save_png', default=None, help='PNG 저장 경로')
    parser.add_argument('--no_browser', action='store_true', help='브라우저 자동 열기 비활성화')
    parser.add_argument('--sea_state', default='BF4', choices=['BF2','BF3','BF4','BF5'])
    parser.add_argument('--n_components', type=int, default=30)
    parser.add_argument('--n_pts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--raydrop_model', default='uniform', choices=['uniform', 'slope'])
    args = parser.parse_args()

    # 데이터 로드 또는 생성
    from wavesim.data_writer import load_episode

    if args.file and os.path.exists(args.file):
        print(f"[로드] {args.file}")
        data = load_episode(args.file)
    else:
        print("[생성] 새 에피소드 생성 중...")
        from wavesim.pipeline import run_episode
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = run_episode(
                sea_state=args.sea_state,
                n_components=args.n_components,
                n_pts=args.n_pts,
                seed=args.seed,
                raydrop_model=args.raydrop_model,
                duration=args.duration,
                dt=0.2,
                output_dir=tmpdir,
                verbose=True
            )
            data = load_episode(fp)

    print(f"[데이터] {data['sea_state']}, Hs={data['Hs_true']:.2f}m, "
          f"{data['gt_wave_field'].shape[0]} frames")

    open_browser = not args.no_browser

    if args.animate:
        save_html = args.save_html or 'wavesim_animation.html'
        plot_animation_html(data, save_html=save_html, open_browser=open_browser)
    else:
        save_html = args.save_html or 'wavesim_visualization.html'
        save_png = args.save_png or 'wavesim_visualization.png'
        plot_static_html(data, frame_idx=args.frame,
                         save_html=save_html,
                         save_png=save_png,
                         open_browser=open_browser)


if __name__ == '__main__':
    main()
