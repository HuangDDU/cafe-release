from cfe.plot.add_color import add_milestone_color


def test_add_milestone_color():
    n = 5
    palette_name = "tab10"
    milestone_color_list = add_milestone_color(n, palette_name)
    assert len(milestone_color_list) == n  # check color number
    assert all([isinstance(color, list) and len(color) == 3 for color in milestone_color_list])  # check color format [r, g, b]
