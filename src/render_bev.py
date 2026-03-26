"""
BEV (Bird's Eye View) renderer for WOMD scenarios.

Renders agent trajectories and map features onto a top-down image
centered on the SDC at current_time_index.
"""

import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extraction import read_tfrecord, parse_scenario


# Agent type colors
AGENT_COLORS = {
    1: '#1f77b4',  # VEHICLE - blue
    2: '#2ca02c',  # PEDESTRIAN - green
    3: '#ff7f0e',  # CYCLIST - orange
}
SDC_COLOR = '#d62728'  # red

# Map feature colors
LANE_COLOR = '#888888'
ROAD_EDGE_COLOR = '#444444'
CROSSWALK_COLOR = '#dddd00'
STOP_SIGN_COLOR = '#ff0000'


def render_scenario(scenario, radius=80, figsize=(12, 12), show_map=True,
                    show_trajectories=True, show_agents=True, trail_alpha=0.4):
    """
    Render a scenario as a BEV image.

    Args:
        scenario: parsed Scenario proto
        radius: meters around SDC to show
        figsize: figure size
        show_map: draw map features (lanes, crosswalks, etc.)
        show_trajectories: draw agent trajectory trails
        show_agents: draw agent bounding boxes at current_time_index
        trail_alpha: opacity of trajectory trails
    """
    s = scenario
    sdc = s.tracks[s.sdc_track_index]
    ct = s.current_time_index

    # Center on SDC position at current_time_index
    sdc_st = sdc.states[ct]
    cx, cy = sdc_st.center_x, sdc_st.center_y

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')

    # --- Map features ---
    if show_map:
        _draw_map(ax, s)

    # --- Trajectories and agents ---
    for track in s.tracks:
        is_sdc = (track.id == sdc.id)
        color = SDC_COLOR if is_sdc else AGENT_COLORS.get(track.object_type, '#999999')
        zorder = 10 if is_sdc else 5

        # Collect valid positions
        positions = []
        for st in track.states:
            if st.valid:
                positions.append((st.center_x, st.center_y))
            else:
                positions.append(None)

        # Draw trajectory trail
        if show_trajectories:
            # History (solid)
            hist_pts = [p for p in positions[:ct+1] if p is not None]
            if len(hist_pts) >= 2:
                xs, ys = zip(*hist_pts)
                ax.plot(xs, ys, color=color, linewidth=2 if is_sdc else 1,
                        alpha=trail_alpha * (1.5 if is_sdc else 1), zorder=zorder)

            # Future (dashed)
            fut_pts = [p for p in positions[ct:] if p is not None]
            if len(fut_pts) >= 2:
                xs, ys = zip(*fut_pts)
                ax.plot(xs, ys, color=color, linewidth=2 if is_sdc else 1,
                        linestyle='--', alpha=trail_alpha * 0.7, zorder=zorder)

        # Draw agent box at current time
        if show_agents:
            st = track.states[ct]
            if st.valid:
                _draw_agent_box(ax, st, color, is_sdc, zorder=zorder + 5)

    # Labels
    ax.set_xlabel('x (m)', color='white', fontsize=10)
    ax.set_ylabel('y (m)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.set_title(f'Scenario {s.scenario_id}  (t={ct})',
                 color='white', fontsize=12, pad=10)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=SDC_COLOR, linewidth=2, label='SDC'),
        plt.Line2D([0], [0], color=AGENT_COLORS[1], linewidth=1.5, label='Vehicle'),
        plt.Line2D([0], [0], color=AGENT_COLORS[2], linewidth=1.5, label='Pedestrian'),
        plt.Line2D([0], [0], color=AGENT_COLORS[3], linewidth=1.5, label='Cyclist'),
        plt.Line2D([0], [0], color='#aaaaaa', linewidth=1, linestyle='--', label='Future'),
        plt.Line2D([0], [0], color=CROSSWALK_COLOR, linewidth=4, alpha=0.4, label='Crosswalk'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              facecolor='#333333', edgecolor='white', labelcolor='white')

    fig.tight_layout()
    return fig, ax


def _draw_agent_box(ax, state, color, is_sdc, zorder=15):
    """Draw an oriented rectangle for an agent."""
    w = state.width
    l = state.length
    heading = state.heading

    # Rectangle centered at origin, then rotate and translate
    corners = np.array([
        [-l/2, -w/2],
        [l/2, -w/2],
        [l/2, w/2],
        [-l/2, w/2],
    ])

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T
    corners[:, 0] += state.center_x
    corners[:, 1] += state.center_y

    rect = plt.Polygon(corners, closed=True, facecolor=color,
                        edgecolor='white' if is_sdc else color,
                        linewidth=2 if is_sdc else 0.5,
                        alpha=0.9, zorder=zorder)
    ax.add_patch(rect)

    # Heading arrow
    arrow_len = l * 0.6
    ax.arrow(state.center_x, state.center_y,
             arrow_len * cos_h, arrow_len * sin_h,
             head_width=0.5, head_length=0.3,
             fc='white', ec='white', alpha=0.8, zorder=zorder + 1)


def _draw_map(ax, scenario):
    """Draw map features: lanes, road edges, crosswalks, stop signs."""
    for mf in scenario.map_features:
        if mf.HasField('lane'):
            pts = [(p.x, p.y) for p in mf.lane.polyline]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=LANE_COLOR, linewidth=0.5, alpha=0.4, zorder=1)

        elif mf.HasField('road_line'):
            pts = [(p.x, p.y) for p in mf.road_line.polyline]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=ROAD_EDGE_COLOR, linewidth=0.8, alpha=0.5, zorder=1)

        elif mf.HasField('road_edge'):
            pts = [(p.x, p.y) for p in mf.road_edge.polyline]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=ROAD_EDGE_COLOR, linewidth=1.0, alpha=0.6, zorder=2)

        elif mf.HasField('crosswalk'):
            pts = [(p.x, p.y) for p in mf.crosswalk.polygon]
            if len(pts) >= 3:
                poly = plt.Polygon(pts, closed=True, facecolor=CROSSWALK_COLOR,
                                   alpha=0.2, zorder=1)
                ax.add_patch(poly)

        elif mf.HasField('stop_sign'):
            pos = mf.stop_sign.position
            ax.plot(pos.x, pos.y, 's', color=STOP_SIGN_COLOR, markersize=5,
                    alpha=0.8, zorder=3)


def render_frame(scenario, timestep, radius=80, figsize=(10, 10), show_map=True):
    """
    Render a single timestep of a scenario for animation.

    Shows agent boxes at the given timestep, with trajectory history
    up to that point (no future).
    """
    s = scenario
    sdc = s.tracks[s.sdc_track_index]
    ct = s.current_time_index

    # Center on SDC at this timestep (follow camera)
    sdc_t = sdc.states[timestep]
    if sdc_t.valid:
        cx, cy = sdc_t.center_x, sdc_t.center_y
    else:
        sdc_ct = sdc.states[ct]
        cx, cy = sdc_ct.center_x, sdc_ct.center_y

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')

    if show_map:
        _draw_map(ax, s)

    # Current time marker
    t_sec = s.timestamps_seconds[timestep] - s.timestamps_seconds[0]
    ct_sec = s.timestamps_seconds[ct] - s.timestamps_seconds[0]

    for track in s.tracks:
        is_sdc = (track.id == sdc.id)
        color = SDC_COLOR if is_sdc else AGENT_COLORS.get(track.object_type, '#999999')
        zorder = 10 if is_sdc else 5

        # Draw history trail up to current timestep
        hist_pts = []
        for t in range(timestep + 1):
            st = track.states[t]
            if st.valid:
                hist_pts.append((st.center_x, st.center_y))

        if len(hist_pts) >= 2:
            xs, ys = zip(*hist_pts)
            ax.plot(xs, ys, color=color, linewidth=2 if is_sdc else 0.8,
                    alpha=0.5, zorder=zorder)

        # Draw agent box at this timestep
        st = track.states[timestep]
        if st.valid:
            _draw_agent_box(ax, st, color, is_sdc, zorder=zorder + 5)

    ax.set_xlabel('x (m)', color='white', fontsize=10)
    ax.set_ylabel('y (m)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)

    phase = 'HISTORY' if timestep < ct else 'FUTURE' if timestep > ct else 'NOW'
    ax.set_title(f'Scenario {s.scenario_id}   t={t_sec:.1f}s / {ct_sec:.1f}s   [{phase}]',
                 color='white', fontsize=12, pad=10)

    fig.tight_layout()
    return fig, ax


def render_gif(scenario, path, radius=80, figsize=(10, 10), fps=10,
               step=2, show_map=True):
    """
    Render an animated GIF of a scenario.

    Args:
        step: render every Nth timestep (2 = every other frame for speed)
        fps: frames per second in output GIF
    """
    import tempfile
    from PIL import Image

    n_steps = len(scenario.timestamps_seconds)
    timesteps = list(range(0, n_steps, step))
    frames = []

    for i, t in enumerate(timesteps):
        fig, ax = render_frame(scenario, t, radius=radius, figsize=figsize,
                               show_map=show_map)
        # Render to PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombytes('RGBA', (w, h), buf)
        frames.append(img.convert('RGB'))
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f'  Frame {i+1}/{len(timesteps)}')

    # Save GIF
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps * step),  # adjust for skipped frames
        loop=0,
    )
    print(f'Saved {path} ({len(frames)} frames)')


def render_and_save(scenario, path, **kwargs):
    """Render a scenario and save to file."""
    fig, ax = render_scenario(scenario, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# CLI: render sample scenarios
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'womd_motion')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'bev_renders')
    os.makedirs(out_dir, exist_ok=True)

    # Load features to find interesting scenarios
    features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.json')
    with open(features_path) as f:
        features = json.load(f)

    # Find scenarios by characteristic
    interesting = {}

    # Highest hard brake count
    by_brake = sorted(features, key=lambda r: r.get('hard_brake_count', 0), reverse=True)
    interesting['hard_brake'] = by_brake[0]['scenario_id']

    # Lowest min_ttc (most dangerous)
    by_ttc = sorted(features, key=lambda r: r.get('min_ttc') or 999)
    interesting['low_ttc'] = by_ttc[0]['scenario_id']

    # Most agents
    by_agents = sorted(features, key=lambda r: r.get('num_agents', 0), reverse=True)
    interesting['crowded'] = by_agents[0]['scenario_id']

    # Highest heading change (maneuvering)
    by_heading = sorted(features, key=lambda r: r.get('ego_total_heading_change', 0), reverse=True)
    interesting['maneuvering'] = by_heading[0]['scenario_id']

    # A typical/median scenario
    by_speed = sorted(features, key=lambda r: abs(r.get('ego_mean_speed', 0) - 3.7))
    interesting['typical'] = by_speed[0]['scenario_id']

    target_ids = set(interesting.values())
    print(f'Looking for {len(target_ids)} scenarios: {interesting}')

    # Scan shards to find them
    import glob
    shard_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord*')))
    found = {}

    for shard_path in shard_files:
        if len(found) == len(target_ids):
            break
        records = read_tfrecord(shard_path)
        for raw in records:
            s = parse_scenario(raw)
            if s.scenario_id in target_ids:
                found[s.scenario_id] = s
                print(f'  Found {s.scenario_id} in {os.path.basename(shard_path)}')
                if len(found) == len(target_ids):
                    break

    # Render PNGs and GIFs
    for label, sid in interesting.items():
        if sid in found:
            png_path = os.path.join(out_dir, f'{label}_{sid}.png')
            render_and_save(found[sid], png_path)

            gif_path = os.path.join(out_dir, f'{label}_{sid}.gif')
            print(f'Animating {label}...')
            render_gif(found[sid], gif_path, step=3, fps=10)
        else:
            print(f'  WARNING: {label} scenario {sid} not found')

    print(f'\nRendered {len(found)} scenarios to {out_dir}/')
