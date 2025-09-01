from cfe.util import parse_resource_useage_string


def test_parse_resource_useage_string():
    usage_string = """
    Command being timed: "python cfe/cli.py benchmark --data tmp/input.h5ad --method_list comp1 paga --save_fig tmp/comp1.jpg --save_h5ad tmp/comp1.h5ad --parameter_file tmp/comp1.yaml"
        User time (seconds): 21.87
        System time (seconds): 4.03
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:26.10
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 845320
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 333696
        Voluntary context switches: 1587
        Involuntary context switches: 298128
        Swaps: 0
        File system inputs: 0
        File system outputs: 178648
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
    """
    usage_dict = parse_resource_useage_string(usage_string)
    expected_usage_dict = {
        "time": 26.1,
        "memory": 845320,
        "cpu": 0.99,
    }
    assert usage_dict == expected_usage_dict
