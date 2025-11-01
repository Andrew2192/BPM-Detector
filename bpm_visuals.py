import numpy as np
import matplotlib.pyplot as plt


def plot_deviation_heatmap(ax_heatmap, valid_times, valid_bpms, ref_series, sheet_bpm, segment_count=8):
    """
    Render a single-row segment-wise tempo deviation heatmap on the given axes.
    Each cell shows the mean percentage deviation (Mic - Reference) per time segment.

    Parameters:
    - ax_heatmap: matplotlib Axes to render the heatmap into
    - valid_times: list of timestamps (seconds) corresponding to valid BPMs
    - valid_bpms: list or np.array of microphone BPM values (>0 filtered)
    - ref_series: list or np.array of reference BPM values aligned to `valid_times`
    - sheet_bpm: fallback reference BPM when segment has no reference samples
    - segment_count: number of time segments to display (default: 8)

    Returns:
    - im: the matplotlib image object returned by `imshow`, suitable for colorbar
    """
    if not valid_times or not valid_bpms:
        # Nothing to render; create an empty placeholder heatmap
        im = ax_heatmap.imshow([[0] * max(1, segment_count)], cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10,
                               extent=[0, max(1, segment_count), 0, 1])
        ax_heatmap.set_xlabel('Time intervals (s)', fontsize=7)
        ax_heatmap.set_yticks([])
        ax_heatmap.set_title('Tempo Deviation Heatmap (%)', fontsize=7, fontweight='bold', pad=4)
        ax_heatmap.set_xticks(np.arange(max(1, segment_count)) + 0.5)
        ax_heatmap.set_xticklabels(["-"] * max(1, segment_count), fontsize=6)
        return im

    valid_bpms_np = np.array(valid_bpms)
    ref_series_np = np.array(ref_series)

    segment_count = segment_count if len(valid_times) >= segment_count else max(1, len(valid_times))
    t_start, t_end = valid_times[0], valid_times[-1]
    bounds = np.linspace(t_start, t_end, segment_count + 1)
    percent_deviations = []
    times_np = np.array(valid_times)

    for i in range(segment_count):
        left = bounds[i]
        right = bounds[i + 1]
        if i < segment_count - 1:
            mask = (times_np >= left) & (times_np < right)
        else:
            mask = (times_np >= left) & (times_np <= right)
        if np.any(mask):
            seg_mic = valid_bpms_np[mask]
            seg_ref = ref_series_np[mask]
            diff_mean = float(np.mean(seg_mic - seg_ref))
            ref_mean = float(np.mean(seg_ref)) if np.any(seg_ref) else float(sheet_bpm)
            pct = (diff_mean / ref_mean) * 100 if ref_mean > 0 else 0.0
        else:
            pct = 0.0  # no data in segment; neutral color
        percent_deviations.append(pct)

    # Render single-row heatmap with labeled segments
    im = ax_heatmap.imshow([percent_deviations], cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10,
                           extent=[0, segment_count, 0, 1])
    ax_heatmap.set_xlabel('Time intervals (s)', fontsize=7)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title('Tempo Deviation Heatmap (%)', fontsize=7, fontweight='bold', pad=4)
    ax_heatmap.set_xticks(np.arange(segment_count) + 0.5)
    interval_labels = [f"{bounds[i]:.0f}–{bounds[i+1]:.0f}s" for i in range(segment_count)]
    ax_heatmap.set_xticklabels(interval_labels, fontsize=6)

    for i, pct in enumerate(percent_deviations):
        color = 'white' if abs(pct) > 5 else 'black'
        ax_heatmap.text(i + 0.5, 0.5, f'{pct:+.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=6, color=color)

    return im


def plot_bpm_timeseries(ax_ts, valid_times, valid_bpms, sheet_bpm, reference_pairs=None):
    """
    Plot microphone BPM over time with reference series and faster/slower fill.

    Parameters:
    - ax_ts: matplotlib Axes for the time series
    - valid_times: timestamps (seconds)
    - valid_bpms: microphone BPMs aligned with valid_times
    - sheet_bpm: fallback reference BPM
    - reference_pairs: optional list of (time, bpm) pairs for a dynamic reference series

    Returns:
    - ref_series: numpy array aligned to valid_times representing the reference BPM series
    """
    ax_ts.plot(valid_times, valid_bpms, color='#2E86AB', linewidth=2, alpha=0.8, label='Real-time Microphone BPM')
    ax_ts.axhline(y=sheet_bpm, color='#A23B72', linestyle='--', linewidth=2, label=f'Reference BPM: {sheet_bpm:.1f}')
    mean_bpm = np.mean(valid_bpms) if len(valid_bpms) > 0 else sheet_bpm
    ax_ts.axhline(y=mean_bpm, color='#F18F01', linestyle='--', linewidth=2, label=f'Average BPM: {mean_bpm:.1f}')

    # Build reference time series aligned to mic timestamps
    ref_at_times_np = None
    try:
        if reference_pairs:
            ref_pairs_sorted = sorted(reference_pairs, key=lambda x: x[0])
            ref_times = [t for t, _ in ref_pairs_sorted]
            ref_bpms_series = [b for _, b in ref_pairs_sorted]
            if len(ref_times) >= 2:
                ref_times_np = np.array(ref_times)
                ref_bpms_np = np.array(ref_bpms_series)
                ref_at_times_np = np.interp(valid_times, ref_times_np, ref_bpms_np)
            elif len(ref_times) == 1:
                ref_at_times_np = np.array([ref_bpms_series[0]] * len(valid_times))
    except Exception:
        ref_at_times_np = None

    if ref_at_times_np is None or len(ref_at_times_np) != len(valid_times):
        ref_series = np.array([sheet_bpm] * len(valid_times))
    else:
        ref_series = ref_at_times_np

    valid_bpms_np = np.array(valid_bpms)
    ax_ts.plot(valid_times, ref_series, color='#A23B72', linewidth=1, alpha=0.7, label='Reference BPM (time series)')
    ax_ts.fill_between(valid_times, ref_series, valid_bpms_np,
                       where=(valid_bpms_np > ref_series),
                       color='#C44536', alpha=0.2, label='Faster than reference')
    ax_ts.fill_between(valid_times, ref_series, valid_bpms_np,
                       where=(valid_bpms_np <= ref_series),
                       color='#5B8C5A', alpha=0.2, label='Slower than reference')

    ax_ts.set_xlabel('Time (seconds)', fontsize=7)
    ax_ts.set_ylabel('BPM', fontsize=7)
    ax_ts.tick_params(axis='both', labelsize=6)
    ax_ts.set_title('Real-time Microphone BPM vs Reference BPM (Time Series)', fontsize=7, fontweight='bold', pad=6)
    ax_ts.legend(fontsize=6)
    ax_ts.grid(True, alpha=0.3)

    return ref_series


def plot_distributions(ax_violin, ax_box, valid_bpms, ref_series):
    """
    Render violin plot (Mic vs Reference) and box plot of deviations.

    Parameters:
    - ax_violin: axes for violin plot
    - ax_box: axes for box plot
    - valid_bpms: list/np.array of mic BPMs
    - ref_series: np.array reference BPMs aligned to valid_times
    """
    # Violin plot
    violin_parts = ax_violin.violinplot([valid_bpms, ref_series], positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor('#2E86AB' if i == 0 else '#A23B72')
        pc.set_alpha(0.6)
    if 'cbars' in violin_parts:
        violin_parts['cbars'].set_color('#333333')
    if 'cmins' in violin_parts:
        violin_parts['cmins'].set_color('#333333')
    if 'cmaxes' in violin_parts:
        violin_parts['cmaxes'].set_color('#333333')
    if 'cmeans' in violin_parts:
        violin_parts['cmeans'].set_color('#F18F01')
    if 'cmedians' in violin_parts:
        violin_parts['cmedians'].set_color('#333333')
    ax_violin.set_ylabel('BPM', fontsize=7)
    ax_violin.set_title('BPM Distribution (Mic vs Reference)', fontsize=7, fontweight='bold', pad=4)
    ax_violin.set_xticks([1, 2])
    ax_violin.set_xticklabels(['Mic', 'Reference'])

    # Mean/median labels (with description and overlap avoidance)
    mean_mic = float(np.mean(valid_bpms)) if len(valid_bpms) > 0 else float('nan')
    median_mic = float(np.median(valid_bpms)) if len(valid_bpms) > 0 else float('nan')
    mean_ref = float(np.mean(ref_series)) if len(ref_series) > 0 else float('nan')
    median_ref = float(np.median(ref_series)) if len(ref_series) > 0 else float('nan')

    # Compute dynamic vertical offset based on axis range
    ylim_low, ylim_high = ax_violin.get_ylim()
    yrange = max(1e-6, ylim_high - ylim_low)
    dy = max(6.0, yrange * 0.06)

    # Mic labels: avoid overlap if mean and median are close
    mic_mean_y = mean_mic
    mic_median_y = median_mic
    x_mic_mean = 1.12
    x_mic_median = 1.12
    if np.isfinite(mic_mean_y) and np.isfinite(mic_median_y) and abs(mic_mean_y - mic_median_y) < dy * 1.1:
        if mic_mean_y >= mic_median_y:
            mic_mean_y = mean_mic + dy / 2
            mic_median_y = median_mic - dy / 2
        else:
            mic_mean_y = mean_mic - dy / 2
            mic_median_y = median_mic + dy / 2
        # Slight horizontal stagger to further reduce overlap
        x_mic_mean = 1.11
        x_mic_median = 1.18

    ax_violin.text(
        x_mic_mean,
        mic_mean_y,
        f"Mean: {mean_mic:.1f}",
        fontsize=6,
        color='#F18F01',
        va='center',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='#F18F01', alpha=0.6, pad=1.5)
    )
    ax_violin.text(
        x_mic_median,
        mic_median_y,
        f"Median: {median_mic:.1f}",
        fontsize=6,
        color='#333333',
        va='center',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='#333333', alpha=0.6, pad=1.5)
    )

    # Reference labels: avoid overlap if mean and median are close
    ref_mean_y = mean_ref
    ref_median_y = median_ref
    x_ref_mean = 2.12
    x_ref_median = 2.12
    if np.isfinite(ref_mean_y) and np.isfinite(ref_median_y) and abs(ref_mean_y - ref_median_y) < dy * 1.1:
        if ref_mean_y >= ref_median_y:
            ref_mean_y = mean_ref + dy / 2
            ref_median_y = median_ref - dy / 2
        else:
            ref_mean_y = mean_ref - dy / 2
            ref_median_y = median_ref + dy / 2
        # Slight horizontal stagger
        x_ref_mean = 2.11
        x_ref_median = 2.18

    ax_violin.text(
        x_ref_mean,
        ref_mean_y,
        f"Mean: {mean_ref:.1f}",
        fontsize=6,
        color='#F18F01',
        va='center',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='#F18F01', alpha=0.6, pad=1.5)
    )
    ax_violin.text(
        x_ref_median,
        ref_median_y,
        f"Median: {median_ref:.1f}",
        fontsize=6,
        color='#333333',
        va='center',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='#333333', alpha=0.6, pad=1.5)
    )

    ax_violin.set_xlim(0.6, 2.4)
    ax_violin.tick_params(axis='both', labelsize=6)

    # Add legend explaining horizontal lines (mean vs median)
    from matplotlib.lines import Line2D
    mean_handle = Line2D([0], [0], color='#F18F01', linestyle='-', linewidth=2, label='Mean')
    median_handle = Line2D([0], [0], color='#333333', linestyle='-', linewidth=2, label='Median')
    ax_violin.legend(handles=[mean_handle, median_handle], fontsize=6, loc='upper right', framealpha=0.6)

    ax_violin.grid(True, alpha=0.3)

    # Box plot of deviations
    deviations = np.array(valid_bpms) - np.array(ref_series)
    bp = ax_box.boxplot([deviations], positions=[1], patch_artist=True, widths=0.6)
    jitter = np.random.normal(0, 0.05, len(deviations))
    ax_box.scatter(np.ones_like(deviations) + jitter, deviations, alpha=0.3, color='#2E86AB', s=12)
    ax_box.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_box.axhline(np.mean(deviations), color='#A23B72', linestyle='--', linewidth=2, label=f'Mean diff: {np.mean(deviations):.2f}')
    bp['boxes'][0].set_facecolor('#F18F01')
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color('#A23B72')
    bp['medians'][0].set_linewidth(2)
    ax_box.set_ylabel('BPM difference (Mic - Reference)', fontsize=7)
    ax_box.set_title('Deviation Distribution (Mic vs Reference)', fontsize=7, fontweight='bold', pad=4)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels(['Diff'])
    ax_box.tick_params(axis='both', labelsize=6)
    ax_box.legend(fontsize=6)
    ax_box.grid(True, alpha=0.3)