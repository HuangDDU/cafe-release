import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

from .._logging import logger
from .._settings import settings


def add_milestone_color(n, palette_name: str = None):
    if palette_name is None:
        palette_name = settings.sns_palette
    palette = sns.color_palette(palette_name)
    if n <= len(palette):
        palette = palette[:n]
    else:
        logger.warning(
            f"The number of colors({n}) is greater than the number of colors in the '{palette_name}' palette({len(palette)}), and the 'husl' palette selection is used."
        )
        palette = sns.color_palette("husl", n_colors=n)
    milestone_color_list = palette
    return milestone_color_list  # rgb


def add_milestone_cell_color(milestone_color_dict, milestone_percentages):
    # color and position fo cell
    milestone_color_df = pd.DataFrame(milestone_color_dict, index=["r", "g", "b"]).T

    def mix_color(mpg):
        # mix related milestone color to get color for a cell
        mpg_color = milestone_color_df.loc[mpg["milestone_id"]]
        mix_color_array = mpg_color.apply(lambda rgb_channel: (rgb_channel.array * mpg["percentage"].array).sum())
        return mcolors.to_hex(mix_color_array)

    cell_color_df = milestone_percentages.groupby("cell_id").apply(lambda mpg: mix_color(mpg))
    return cell_color_df  # hex


def rgb2hex(palette):
    return [mcolors.to_hex(color) for color in palette]
