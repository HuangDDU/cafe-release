import re


def parse_resource_useage_string(usage_string: str) -> dict:
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

    # extract memory(KB)
    mem_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", usage_string)
    memory = int(mem_match.group(1)) if mem_match else 0

    # extract cpu percentage
    cpu_match = re.search(r"Percent of CPU this job got: (\d+)%", usage_string)
    cpu = float(cpu_match.group(1)) / 100 if cpu_match else 0.0

    usage_dict = {
        "time": time_sec,
        "memory": memory,
        "cpu": cpu,
    }

    return usage_dict
