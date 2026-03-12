from rich import print as rprint
from rich.tree import Tree


def dict_to_tree(d, tree=None, name="root", level=None, current_level=0):
    """Convert a nested dictionary to a rich Tree

    Parameters
    ----------
    d : dict
        The dictionary to display
    tree : Tree, optional
        The rich Tree object
    name : str
        The root node name
    level : int, optional
        The maximum level to display, None means display all levels
    current_level : int
        The current level (used internally)
    """
    if tree is None:
        tree = Tree(f"[bold]{name}[/bold]")

    for key, value in d.items():
        if isinstance(value, dict):
            # Check if the level limit is exceeded
            if level is not None and current_level >= level:
                tree.add(f"[cyan]{key}[/cyan] → [dim]dict with {len(value)} keys...[/dim]")
            else:
                branch = tree.add(f"[cyan]{key}[/cyan]")
                dict_to_tree(value, branch, key, level=level, current_level=current_level + 1)
        else:
            # Display the type and brief information of the value
            type_name = type(value).__name__
            if hasattr(value, "shape"):
                info = f"[green]{type_name}[/green] shape={value.shape}"
            elif hasattr(value, "__len__") and not isinstance(value, str):
                info = f"[green]{type_name}[/green] len={len(value)}"
            else:
                info = f"[green]{type_name}[/green]: {repr(value)[:50]}"
            tree.add(f"[yellow]{key}[/yellow] → {info}")

    return tree


def print_dict(d, name="root", level=None):
    tree = dict_to_tree(d, name=name, level=level)  # 显示一层结构
    rprint(tree)
