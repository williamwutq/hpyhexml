import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

'''
Generate comparative box plots for different algorithms based on predefined data. Modify the data as needed.
This should not be a part of a module nor part of the machine learning codebase, but a convienient script to generate graphs for reports.
'''

# Settings
LOG_SCALE = True

turn_infinity = 1000000
score_infinity = 24000000

# Data: (mean, sd, low_extreme, high_extreme)
subgroup_Turn = {
    "5-1": {
        "Random": (14.065, 4.39098793, 7, 36),
        "Density\nIndex": (46.21, 26.35445586, 13, 211),
        "NaiveScore\nIndex": (36.2175, 15.66748843, 11, 108),
        "Score\nIndex": (34.13, 17.13718472, 12, 127),
        "Nrsearch": (84.26, 63.43447328, 15, 469),
    },
    "5-3": {
        "Random": (21.21, 9.968495373, 6, 68),
        "Density\nIndex": (98.4025, 65.62572281, 15, 572),
        "NaiveScore\nIndex": (118.3675, 85.99353141, 17, 649),
        "Score\nIndex": (141.4125, 105.4231111, 17, 621),
        "Nrsearch": (1068.6575, 1035.869176, 27, 6159),
    },
    "8-5": {
        "Random": (50.185, 15.59088756, 28, 136),
        "Density\nIndex": (980.245, 629.1562842, 128, 4463),
        "NaiveScore\nIndex": (513.0625, 367.5751877, 78, 2288),
        "Score\nIndex": (829.6825, 721.5824705, 60, 5034),
        "Nrsearch": (turn_infinity,0,turn_infinity,turn_infinity),
    },
}
subgroup_Score = {
    "5-1": {
        "Random": (113.3475, 83.5116264, 22, 542),
        "Density\nIndex": (789.25, 545.8374415, 107, 4158),
        "NaiveScore\nIndex": (577.915, 318.6184988, 68, 2042),
        "Score\nIndex": (537.285, 355.6116123, 71, 2490),
        "Nrsearch": (1583.6425, 1307.188827, 140, 9403),
    },
    "5-3": {
        "Random": (246.2025, 205.1711761, 28, 1254),
        "Density\nIndex": (1868.3625, 1377.492278, 127, 11861),
        "NaiveScore\nIndex": (2268.175, 1777.562667, 182, 13180),
        "Score\nIndex": (2748.3525, 2184.326775, 189, 12696),
        "Nrsearch": (22038.265, 21512.2781, 370, 127588),
    },
    "8-5": {
        "Random": (409.315, 313.8321459, 99, 2216),
        "Density\nIndex": (19755.5125, 13082.43051, 2073, 92258),
        "NaiveScore\nIndex": (9995.4225, 7614.214309, 1043, 46640),
        "Score\nIndex": (16574.2475, 14944.02899, 605, 103839),
        "Nrsearch": (score_infinity,0,score_infinity,score_infinity),
    },
}

box_width = 0.5

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # second y-axis

def plot_group(ax, data, x_offset, color, meancolor, fontsize=5, labelgroup = True, datafont = "Arial", labelfont = "Times New Roman"):
    lj = len(data)
    for j, (dk, dd) in enumerate(data.items()):
        li = len(dd)
        for i, (k, (mean, sd, low, high)) in enumerate(dd.items()):
            if mean == turn_infinity or mean == score_infinity:
                inf = True
            else: inf = False
            strmean = f"{mean:.1f}"
            strhigh = f"{high:.1f}"
            strlow = f"{low:.1f}"
            x = i + x_offset + j * (li + 1)
            lower_sd = mean - sd
            upper_sd = mean + sd

            if LOG_SCALE:
                mean = np.log10(mean)
                sd = np.log10(sd)
                low = np.log10(low)
                high = np.log10(high)
                lower_sd = np.log10(lower_sd) if lower_sd > 0 else 0
                upper_sd = np.log10(upper_sd)
            
            # whiskers
            ax.vlines(x + box_width/2, low, high, color, linewidth=1.2)
            if inf:
                # arrow for infinity
                ax.add_patch(plt.Rectangle(
                    (x, mean),
                    box_width,
                    mean/12,
                    facecolor=color, edgecolor=color, linewidth = 0
                ))
                # label below box for infinity
                ax.text(
                    x + box_width / 2,
                    low - 0.01,
                    "inf",
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    rotation=0,
                    color=color,
                    font=datafont,
                )
            else:
                # label avove high whisker
                ax.text(
                    x + box_width / 2,
                    high + 0.01,
                    strhigh,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    rotation=0,
                   color=color,
                    font=datafont,
                )
                # label below low whisker
                ax.text(
                    x + box_width / 2,
                    low - 0.01,
                    strlow,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    rotation=0,
                    color=color,
                    font=datafont,
                )
                # rectangle mean ± SD
                ax.add_patch(plt.Rectangle(
                    (x, lower_sd),
                    box_width,
                    upper_sd - lower_sd,
                    facecolor=color, edgecolor=color, linewidth = 0
                ))
                # mean line
                ax.hlines(mean, x, x + box_width, color=meancolor, linewidth=1)
                # Label under mean line
                ax.text(
                    x + box_width / 2,
                    mean - 0.01,
                    strmean,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    rotation=0,
                    color=meancolor,
                    font=datafont,
                )
        
    for j, (dk, dd) in enumerate(data.items()):
        # Label group
        if labelgroup:
            ax.text(
                j * (li + 1) + (li - 1) / 2 + box_width,
                ax.get_ylim()[0] - 0.4,
                dk,
                ha="center",
                va="top",
                fontsize=fontsize*1.6,
                rotation=0,
                color="black",
                fontweight="bold",
                font=labelfont,
            )
            for i, (k, (mean, sd, low, high)) in enumerate(dd.items()):
                x = i + x_offset + j * (li + 1)
                # Label algorithm
                ax.text(
                    x + box_width,
                    ax.get_ylim()[0] - 0.05,
                    k,
                    ha="center",
                    va="top",
                    fontsize=fontsize*1.3,
                    rotation=0,
                    color="black",
                    fontweight="semibold",
                    font=labelfont,
                )


if LOG_SCALE:
    # Find max value for turn
    max_turn = max(v[0] + v[1] for g in subgroup_Turn.values() for v in g.values())
    max_score = max(v[0] + v[1] for g in subgroup_Score.values() for v in g.values())
    log_turn = np.ceil(np.log10(max_turn))
    log_score = np.ceil(np.log10(max_score))
    # Find min value for turn
    min_turn = min(v[0] - v[1] for g in subgroup_Turn.values() for v in g.values() if v[0] - v[1] > 0)
    min_score = min(v[0] - v[1] for g in subgroup_Score.values() for v in g.values() if v[0] - v[1] > 0)
    log_min_turn = np.floor(np.log10(min_turn))
    log_min_score = np.floor(np.log10(min_score))
    # Convert numbers on axis to log scale
    ax1_range = range(int(log_min_turn) + 1, int(log_turn))
    ax1.set_yticks(ax1_range)
    ax1.set_yticklabels([f"$10^{i}$" for i in ax1_range])
    ax2_range = range(int(log_min_score) + 1, int(log_score))
    ax2.set_yticks(ax2_range)
    ax2.set_yticklabels([f"$10^{i}$" for i in ax2_range])

# Styling
ax1.set_ylabel("Game Turn" + (" (log scale)" if LOG_SCALE else ""))
ax2.set_ylabel("Game Score" + (" (log scale)" if LOG_SCALE else ""))
ax1.yaxis.label.set_color("darkblue")
ax2.yaxis.label.set_color("green")
ax1.yaxis.label.set_fontweight("bold")
ax2.yaxis.label.set_fontweight("bold")
ax1.yaxis.label.set_fontname("Times New Roman")
ax2.yaxis.label.set_fontname("Times New Roman")
ax1.yaxis.label.set_fontsize(12)
ax2.yaxis.label.set_fontsize(12)
ax1.set_title("Performance Comparison between DensityIndex, Naive Greedy Score Index,\nGreedy Score Index, and Random Algorithms")
ax1.title.set_fontsize(14)
ax1.title.set_fontweight("bold")
ax1.title.set_fontname("Times New Roman")

# Sync x limits
ax1.set_xlim(-0.5, len(subgroup_Turn) * (len(next(iter(subgroup_Turn.values())).keys()) + 1) - 0.5)
ax2.set_xlim(ax1.get_xlim())
ax1.set_ylim(bottom=0.5)
ax2.set_ylim(bottom=0.5)
ax1.set_ylim(top=log_turn + 0.5)
ax2.set_ylim(top=log_score)

# Plot subgroup A on left axis
plot_group(ax1, subgroup_Turn, x_offset=0, color="darkblue", meancolor="white")

# Plot subgroup B on right axis (shifted in x)
plot_group(ax2, subgroup_Score, x_offset=0.5, color="green", meancolor="black", labelgroup = False)

# Remove x axis ticks
ax1.set_xticks([])
ax2.set_xticks([])

plt.tight_layout()
plt.show()
