import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


def hex_to_rgb(hex_code):
    """
    Converts hex to rgb colours
    ===========================

    Parameters:
    -----------
    hex: 6 characters representing a hex colour (str)

    Returns:
    --------
    rgb: RGB values (list of 3 floats)
    """
    hex_code = hex_code.strip("#")  # removes hash symbol if present
    lv = len(hex_code)
    rgb = tuple(int(hex_code[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return rgb


def rgb_to_dec(rgb):
    """
    Converts rgb to decimal values
    ==============================

    Parameters:
    -----------
    rgb: rgb values (list of 3 floats)

    Returns:
    --------
    dec: decimal values (list of 3 floats)
    """
    return [v / 256 for v in rgb]


def get_continuous_cmap(hex_list, float_list=None):
    """
    Creates and returns a color map that can be used in heat map figures
    ====================================================================
    Note:
    If a float_list is not provided, colour map graduates linearly between each color in hex_list.
    If a float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: Hex codes for each color in the color map (list of str)
    float_list: Floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1. (list of floats)

    Returns
    ----------
    cmap: Colormap (matplotlib.colors.LinearSegmentedColormap)
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def draw_schedule(
    reward_schedule, action_histories=None, save=False, filename="schedule.png", title=None, compare_to=None, figsize=(10, 2)
):
    """
    Plot a reward schedule and optionally action histories and mean bias.
    =====================================================================

    Parameters:
    -----------
    reward_schedule: reward schedule (np.array)
    action_histories: action histories (list of np.arrays)
    mean_bias: mean bias (float)
    save: whether to save the plot (bool)
    filename: filename to save the plot (str)
    """
    cmap = get_continuous_cmap(["058d96", "00a450", "52b448", "8ac341"])

    if action_histories is not None and len(action_histories) > 1:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=(9, 1), wspace=0.05)
        ax = plt.subplot(gs[0, 0])
        ax_ = plt.subplot(gs[0, 1])
        ax.axhline(0.5, 0, 1.14, color="k", linestyle="dotted", clip_on=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 2))
        ax.axhline(0.5, 0, 1, color="k", linestyle="dotted")

    ax.invert_yaxis()

    ax.plot(
        np.arange(reward_schedule.shape[0])[reward_schedule[:, 0] == 1],
        np.zeros(sum(reward_schedule[:, 0] == 1)),
        "o",
        color=cmap(0.5),
    )
    ax.plot(
        np.arange(reward_schedule.shape[0])[reward_schedule[:, 1] == 1],
        np.ones(sum(reward_schedule[:, 1] == 1)),
        "o",
        color=cmap(1.0),
    )
    if action_histories is not None:
        if len(action_histories) > 1:
            ax.plot(np.mean(action_histories, axis=0), color=cmap(0.0))
            ax_.hist(
                np.mean(np.array(action_histories) == 0, axis=1),
                orientation="horizontal",
                density=True,
                histtype="stepfilled",
                color=cmap(0.75),
            )
            ax_.yaxis.set_label_position("right")
            ax_.yaxis.tick_right()
            ax_.set_frame_on(False)
            ax_.set_ylim(0, 1)
            ax_.set_xticks([])
            ax_.axhline(
                np.mean(np.array(action_histories) == 0, axis=1).mean(),
                -8.9,
                1,
                color=cmap(0.0),
                linestyle="dashed",
                linewidth=1.5,
                clip_on=False,
            )

            plt.text(
                1.45, 0.5, "Bias Distribution", color=cmap(0.75), transform=ax_.transAxes, rotation=270, va="center"
            )
            plt.text(1.65, 0.5, "P(Target Odor)", color=cmap(0.0), transform=ax_.transAxes, rotation=270, va="center")

            custom_lines = [Line2D([0], [0], color=cmap(0.0), lw=1.5, ls="dashed")]
            ax.legend(
                custom_lines,
                [f"Net Bias = {np.mean(np.array(action_histories)==0)*100:0.2f} %"],
                frameon=False,
                loc=(0.06, 0.05),
            )
        else:
            ax.plot(action_histories[0], color=cmap(0.0))
            plt.text(1.02, 0.5, "Odor Choice", color=cmap(0.0), transform=ax.transAxes, rotation=270, va="center")

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Odor A\n'Target'", "No Preference", "Odor B"])
    ax.get_yticklabels()[0].set_color(cmap(0.25))
    ax.set_xlabel("Trial")
    ax.set_frame_on(False)

    if title is not None:
        plt.title(title)

    if compare_to is not None:
        assert len(action_histories) == len(
            compare_to
        ), "action_histories and compare_to must have same number of simulations"
        ax.plot(np.mean(compare_to, axis=0), color="grey", alpha=0.5)
        ax_.hist(
            np.mean(np.array(compare_to) == 0, axis=1),
            orientation="horizontal",
            density=True,
            histtype="stepfilled",
            color="grey",
            alpha=0.5,
        )
        ax_.axhline(
            np.mean(np.array(compare_to) == 0, axis=1).mean(),
            -8.9,
            1,
            color="grey",
            linestyle="dashed",
            linewidth=1.5,
            clip_on=False,
        )
        custom_lines = [
            Line2D([0], [0], color=cmap(0.0), lw=1.5, ls="dashed", alpha=0.5),
            Line2D([0], [0], color="grey", lw=1.5, ls="dashed", alpha=0.5),
        ]
        ax.legend(
            custom_lines,
            [
                f"Schedule Bias = {np.mean(np.array(action_histories)==0)*100:0.2f} %",
                f"Naive Bias = {np.mean(np.array(compare_to)==0)*100:0.2f} %",
            ],
            frameon=False,
            loc=(0.06, 0.05),
            ncol=2,
        )

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, transparent=True)
    plt.show()
    plt.close()


def draw_optimization_history(fitnesses, title, save=False, filename="history.png"):
    """
    Plot the optimization history.
    ==============================

    Parameters:
    -----------
    fitnesses: fitnesses (list of list of floats)
    title: title of the plot (str)
    save: whether to save the plot (bool)
    filename: filename to save the plot (str)
    """
    plt.plot(fitnesses, "k.", alpha=0.5)
    plt.plot(np.mean(fitnesses, axis=1), "k--", label="mean")
    plt.plot(np.max(fitnesses, axis=1), "k-o", label="best")
    plt.hlines(0.5, 0, len(fitnesses) - 1, color="k", linestyle="dotted")
    plt.text(0.7 * len(fitnesses), 0.505, "chance level")
    plt.ylabel("bias")
    plt.xlabel("generation")
    plt.xticks(range(0, len(fitnesses)), range(1, len(fitnesses) + 1))
    plt.box(False)
    plt.title(title)
    if save:
        plt.savefig(filename)
    plt.show()
    plt.close()
