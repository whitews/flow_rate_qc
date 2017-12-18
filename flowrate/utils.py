import os
import numpy as np
import scipy.stats as stats
from statsmodels.robust.scale import mad
import pandas as pd
from matplotlib import pyplot

import flowio
from flowio.create_fcs import create_fcs

diff_roll = 0.01
final_roll = 0.02
k1 = 2.0
k2 = 2.0

default_fig_size = (16, 4)


def find_channel_index(channel_dict, pnn_text):
    index = None

    for k, v in channel_dict.items():
        if v['PnN'] == pnn_text:
            index = int(k) - 1

    return index


def find_channel_label(channel_dict, channel_index):
    label = ''

    for k, v in channel_dict.items():
        if k == str(channel_index + 1):
            if 'PnS' in v:
                label = " ".join([v['PnN'], v['PnS']])
            else:
                label = v['PnN']

    return label


def plot_channel(
        flow_data,
        x_channel_index,
        y_channel_index,
        x_biex=True,
        y_biex=True,
        fig_size=default_fig_size
):
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))

    x_label = find_channel_label(flow_data.channels, x_channel_index)
    y_label = find_channel_label(flow_data.channels, y_channel_index)

    pre_scale = 0.003

    if x_biex:
        x = np.arcsinh(events[:, x_channel_index] * pre_scale)
    else:
        x = events[:, x_channel_index]

    if y_biex:
        y = np.arcsinh(events[:, y_channel_index] * pre_scale)
    else:
        y = events[:, y_channel_index]

    my_cmap = pyplot.cm.get_cmap('jet')
    my_cmap.set_under('w', alpha=0)

    bins = int(np.sqrt(x.shape[0]))

    fig = pyplot.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(flow_data.name, fontsize=16)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.hist2d(
        x,
        y,
        bins=[bins, bins],
        cmap=my_cmap,
        vmin=0.9
    )

    fig.tight_layout()

    pyplot.show()


def plot_flow_rate(
        file_name,
        flow_rate,
        event_idx,
        figure,
        plot_position,
        good_events=None,
        hline=None,
        trendline=None,
        x_lim=None,
        y_lim=None
):
    ax = figure.add_subplot(5, 1, plot_position)

    ax.set_title(file_name, fontsize=16)

    ax.set_xlabel("Event", fontsize=14)
    ax.set_ylabel("Flow rate (events/ms)", fontsize=14)

    if x_lim is None:
        ax.set_xlim([0, max(event_idx)])
    else:
        ax.set_xlim([x_lim[0], x_lim[1]])

    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])

    if good_events is not None:
        ax.scatter(
            event_idx[good_events],
            flow_rate[good_events],
            c='darkslateblue',
            s=1,
            lw=0
        )
    else:
        ax.plot(
            event_idx,
            flow_rate,
            c='darkslateblue'
        )

    if hline is not None:
        ax.axhline(hline, linestyle='-', linewidth=1, c='coral')

    if trendline is not None:
        ax.plot(event_idx, trendline, c='cornflowerblue')


def calculate_flow_rate(events, time_index, roll):
    time_diff = np.diff(events[:, time_index])
    time_diff = np.insert(time_diff, 0, 0)

    time_diff_mean = pd.rolling_mean(time_diff, roll, min_periods=1)
    min_diff = time_diff_mean[time_diff_mean > 0].min()
    time_diff_mean[time_diff_mean == 0] = min_diff

    flow_rate = 1 / time_diff_mean

    return flow_rate


def plot_deviation(
        file_name,
        flow_rate,
        event_indices,
        diff,
        stable_diff,
        smooth_stable_diff,
        threshold,
        figure,
        plot_position
):
    ax = figure.add_subplot(5, 1, plot_position)

    ax.set_title(file_name, fontsize=16)

    ax.set_xlim([0, len(flow_rate)])
    # pyplot.ylim([0, 5])

    ax.set_xlabel("Event", fontsize=14)
    ax.set_ylabel("Deviation (log)", fontsize=14)

    ax.plot(
        event_indices,
        np.log10(1 + diff),
        c='coral',
        alpha=0.6,
        linewidth=1
    )

    ax.plot(
        event_indices,
        np.log10(1 + stable_diff),
        c='cornflowerblue',
        alpha=0.6,
        linewidth=1
    )

    ax.plot(
        event_indices,
        np.log10(1 + smooth_stable_diff),
        c='darkslateblue',
        alpha=1.0,
        linewidth=1
    )

    ax.axhline(np.log10(1 + threshold), linestyle='dashed', linewidth=2, c='crimson')


def get_false_bounds(bool_array):
    diff = np.diff(np.hstack((0, ~bool_array, 0)))

    start = np.where(diff == 1)
    end = np.where(diff == -1)

    return start[0], end[0]


def plot_channel_good_vs_bad(
        file_name,
        channel_data,
        time_data,
        channel_name,
        good_event_map,
        figure,
        sub_plot_count,
        plot_position,
        bi_ex=True,
        drop_negative=False
):
    pre_scale = 0.003

    if bi_ex:
        channel_data = np.arcsinh(channel_data * pre_scale)

    starts, ends = get_false_bounds(good_event_map)

    good_cmap = pyplot.cm.get_cmap('jet')
    good_cmap.set_under('w', alpha=0)

    ax = figure.add_subplot(sub_plot_count, 1, plot_position)

    ax.set_title(file_name, fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(channel_name, fontsize=14)

    bins_good = int(np.sqrt(channel_data.shape[0]))

    ax.hist2d(
        time_data,
        channel_data,
        bins=bins_good,
        cmap=good_cmap,
        vmin=0.9
    )

    for i, s in enumerate(starts):
        ax.axvspan(
            time_data[s],
            time_data[ends[i] - 1],
            facecolor='pink',
            alpha=0.3,
            edgecolor='deeppink'
        )

    if drop_negative:
        ax.set_ylim(ymin=0)


def clean_stage1(flow_data, time_channel):
    """
    Determines the flow rate, median flow rate, and initial good events
    """
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))

    time_index = find_channel_index(flow_data.channels, time_channel)

    diff_roll_count = int(diff_roll * events.shape[0])
    flow_rate = calculate_flow_rate(events, time_index, diff_roll_count)

    median = np.median(flow_rate)
    median_diff = np.abs(flow_rate - median)

    threshold = k1 * mad(median_diff)
    # threshold = 1.0 * np.std(median_diff)

    initial_good_events = median_diff < threshold

    return flow_rate, median, initial_good_events, median_diff


def clean_stage2(flow_rate, event_indices, initial_good_events, fit_method='lin'):
    good_event_indices = event_indices[initial_good_events]

    if fit_method == 'lin':
        line_regress = stats.linregress(good_event_indices, flow_rate[initial_good_events])
        fit = (line_regress.slope * event_indices) + line_regress.intercept
    elif fit_method == 'quad':
        poly = np.polyfit(good_event_indices, flow_rate[initial_good_events], 2)
        p = np.poly1d(poly)
        fit = p(event_indices)
    elif fit_method == 'cubic':
        poly = np.polyfit(good_event_indices, flow_rate[initial_good_events], 3)
        p = np.poly1d(poly)
        fit = p(event_indices)
    else:
        return

    return fit


def clean_stage3(stable_diff):
    final_threshold = k2 * mad(stable_diff)

    final_w = int(final_roll * stable_diff.shape[0])
    smoothed_diff = pd.rolling_mean(stable_diff, window=final_w, min_periods=1, center=True)
    final_good_events = smoothed_diff < final_threshold

    return final_good_events, smoothed_diff, final_threshold


def generate_data_channel_plots(flow_data, good_events, figure, time_channel='Time'):
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))
    time_index = find_channel_index(flow_data.channels, time_channel)

    plot_position = 1

    for channel in sorted([int(k) for k in flow_data.channels.keys()]):
        data_channel = flow_data.channels[str(channel)]['PnN']

        if data_channel.lower() == 'time':
            continue

        plot_channel_good_vs_bad(
            flow_data.name,
            events[:, channel - 1],  # channel is indexed by 1
            events[:, time_index],
            data_channel,
            good_events,
            figure,
            flow_data.channel_count - 1,
            plot_position,
            drop_negative=False
        )

        plot_position += 1


def clean(fcs_file, time_channel="Time", plot_data_channels=False, results_dir=None):
    flow_data = flowio.FlowData(fcs_file)

    flow_rate, median, good_events_stage1, median_diff = clean_stage1(
        flow_data,
        time_channel
    )
    event_indices = np.arange(0, flow_rate.shape[0])
    y_lim = [flow_rate.min(), flow_rate[50:].max()]

    fig_flow_rate = pyplot.figure(1, figsize=(16, 20))
    plot_position = 1

    plot_flow_rate(
        fcs_file,
        flow_rate,
        event_indices,
        fig_flow_rate,
        plot_position,
        y_lim=y_lim
    )

    plot_position += 1

    plot_flow_rate(
        fcs_file,
        flow_rate,
        event_indices,
        fig_flow_rate,
        plot_position,
        good_events=good_events_stage1,
        hline=median,
        y_lim=y_lim
    )

    flow_rate_fit = clean_stage2(flow_rate, event_indices, good_events_stage1, fit_method='lin')
    stable_diff = np.abs(flow_rate - flow_rate_fit)
    good_events_stage3, smoothed_diff, final_threshold = clean_stage3(stable_diff)

    plot_position += 1

    plot_flow_rate(
        fcs_file,
        flow_rate,
        event_indices,
        fig_flow_rate,
        plot_position,
        good_events=good_events_stage3,
        hline=median,
        trendline=flow_rate_fit,
        y_lim=y_lim
    )

    flow_rate_fit2 = clean_stage2(
        flow_rate,
        event_indices,
        good_events_stage3,
        fit_method='cubic'
    )
    final_stable_diff = np.abs(flow_rate - flow_rate_fit2)
    final_good_events, smoothed_diff2, final_threshold2 = clean_stage3(final_stable_diff)

    plot_position += 1

    plot_flow_rate(
        fcs_file,
        flow_rate,
        event_indices,
        fig_flow_rate,
        plot_position,
        good_events=final_good_events,
        hline=median,
        trendline=flow_rate_fit2,
        y_lim=y_lim
    )

    plot_position += 1

    plot_deviation(
        fcs_file,
        flow_rate,
        event_indices,
        median_diff,
        final_stable_diff,
        smoothed_diff2,
        final_threshold2,
        fig_flow_rate,
        plot_position
    )

    if plot_data_channels:
        fig_data_channels = pyplot.figure(2, figsize=(16, 4 * (flow_data.channel_count - 1)))
        generate_data_channel_plots(
            flow_data,
            final_good_events,
            fig_data_channels,
            time_channel
        )
        fig_data_channels.tight_layout(pad=1.0)

    fig_flow_rate.tight_layout(pad=1.0)

    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        events = np.reshape(flow_data.events, (-1, flow_data.channel_count))

        good_events = events[final_good_events]
        bad_events = events[np.logical_not(final_good_events)]

        base_name = os.path.basename(fcs_file)
        good_file_path = os.path.join(results_dir, base_name.replace('.fcs', '_good.fcs'))
        bad_file_path = os.path.join(results_dir, base_name.replace('.fcs', '_bad.fcs'))
        flow_rate_fig_path = os.path.join(
            results_dir,
            base_name.replace('.fcs', '_flow_rate.png')
        )
        channel_fig_path = os.path.join(
            results_dir,
            base_name.replace('.fcs', '_channel_data.png')
        )

        # build channel names
        channel_names = []
        opt_channel_names = []
        for channel in sorted([int(k) for k in flow_data.channels.keys()]):
            channel_names.append(flow_data.channels[str(channel)]['PnN'])

            if 'PnS' in flow_data.channels[str(channel)]:
                opt_channel_names.append(flow_data.channels[str(channel)]['PnS'])
            else:
                opt_channel_names.append(None)

        # build some extra metadata fields
        extra = {}
        acq_date = None
        if 'date' in flow_data.text:
            acq_date = flow_data.text['date']

        if 'timestep' in flow_data.text:
            extra['TIMESTEP'] = flow_data.text['timestep']

        if 'btim' in flow_data.text:
            extra['BTIM'] = flow_data.text['btim']

        if 'etim' in flow_data.text:
            extra['ETIM'] = flow_data.text['etim']

        good_fh = open(good_file_path, 'wb')
        bad_fh = open(bad_file_path, 'wb')

        create_fcs(
            good_events.flatten().tolist(),
            channel_names,
            good_fh,
            date=acq_date,
            extra=extra,
            opt_channel_names=opt_channel_names
        )
        good_fh.close()

        create_fcs(
            bad_events.flatten().tolist(),
            channel_names,
            bad_fh,
            date=acq_date,
            extra=extra,
            opt_channel_names=opt_channel_names
        )
        bad_fh.close()

        # save figures
        fig_flow_rate.savefig(flow_rate_fig_path)
        if plot_data_channels:
            fig_data_channels.savefig(channel_fig_path)

    # close figures
    fig_flow_rate.clf()
    fig_data_channels.clf()

