import json
import re

import pandas as pd


def parse_bash_resource_usage_string(usage_string: str) -> dict:
    # parse usage string generate by "/usr/bin/time -v" to usage dict

    # extract time(s)
    time_match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:]+(?:\.\d+)?)", usage_string)
    time_str = time_match.group(1) if time_match else "0"
    # h:mm:ss or m:ss
    parts = time_str.split(":")
    if len(parts) == 3:
        time_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        time_sec = int(parts[0]) * 60 + float(parts[1])
    else:
        time_sec = float(parts[0])

    # extract memory(MB)
    mem_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", usage_string)
    memory = int(mem_match.group(1)) if mem_match else 0  # KB
    memory = memory / 1024  # MB
    memory = round(memory, 2)

    # extract cpu percentage
    cpu_match = re.search(r"Percent of CPU this job got: (\d+)%", usage_string)
    cpu = float(cpu_match.group(1)) / 100 if cpu_match else 0.0

    usage_dict = {
        "time": time_sec,
        "memory": memory,
        "cpu": cpu,
    }

    return usage_dict


def parse_docker_resource_usage_string_list(usage_string_list: list) -> dict:
    usage_dict_list = []
    for usage_string in usage_string_list:
        usage_string_dict = json.loads(usage_string)
        memory = usage_string_dict["memory_stats"].get("usage", 0)
        memory = memory / 1024 / 1024  # MB
        memory = round(memory, 2)

        # TODO: CPU can't be calculated
        cpu = 0
        tmp_usage_dict = {
            "memory": memory,
            "cpu": cpu,
        }
        usage_dict_list.append(tmp_usage_dict)

    usage_df = pd.DataFrame(usage_dict_list)

    usage_dict = {
        "memory": usage_df["memory"].max(),
        "cpu": usage_df["cpu"].max(),
        "time": 0,
    }

    return usage_dict
