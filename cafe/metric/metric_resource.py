from ..data import FateAnnData


def calculate_resource_usage(fadata: FateAnnData, model_name: str, format_text: bool = True):
    resource_usage = fadata.get_resource_usage(model_name)
    out = {}

    # raw values
    time = resource_usage.get("time", 0)  # seconds
    memory = resource_usage.get("memory", 0)  # KB
    out["time"] = time
    out["memory"] = memory

    # formatted text
    if format_text:
        if time < 60:
            time_text = f"{time:.0f}s"
        elif time < 3600:
            time_text = f"{time/60:.0f}min"
        else:
            time_text = f"{time/3600:.0f}h"
        out["time_text"] = time_text
    if format_text:
        if memory < 1024:
            memory_text = f"{memory:.0f}M"
        else:
            memory_text = f"{memory/1024:.0f}G"
        out["memory_text"] = memory_text

    return out
