from ..data import FateAnnData
from ..util import format_memory, format_time


# TODO: add docs
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
        out["time_text"] = format_time(time)
        out["memory_text"] = format_memory(memory)

    return out
