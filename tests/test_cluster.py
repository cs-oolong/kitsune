from kitsune.cluster import parse_inventory, parse_node

SCONTROL = """\
NodeName=dl-01 Arch=x86_64 CoresPerSocket=32 CPUAlloc=40 CPUTot=64 Gres=gpu:L40S:4 GresUsed=gpu:L40S:4(IDX:0-3) NodeAddr=dl-01 State=MIXED+PLANNED Partitions=l40s CfgTRES=cpu=64,mem=512G,billing=64,gres/gpu=4 AllocTRES=cpu=40,mem=300G,gres/gpu=4
NodeName=dl-02 Arch=x86_64 CPUAlloc=8 CPUTot=64 Gres=gpu:L40S:4 State=MIXED Partitions=l40s CfgTRES=cpu=64,mem=512G,billing=64,gres/gpu=4 AllocTRES=cpu=8,mem=20G,gres/gpu=1,gres/gpu:l40s=1
NodeName=dl-04 Arch=x86_64 CPUAlloc=0 CPUTot=128 Gres=gpu:H100:2 State=DOWN+DRAIN Reason=Not responding [slurm@2026-10-27] Partitions=h100 CfgTRES=cpu=128,mem=1500G,billing=128,gres/gpu=2 AllocTRES=
NodeName=dl-06 Arch=x86_64 CPUAlloc=32 CPUTot=80 Gres=gpu:RTXA5000:4 State=MIXED Partitions=a5000 CfgTRES=cpu=80,mem=1031667M,billing=80,gres/gpu=4 AllocTRES=cpu=32,mem=800G,gres/gpu=4
NodeName=dl-09 Arch=x86_64 CPUAlloc=8 CPUTot=64 Gres=gpu:P5000:6 State=MIXED Partitions=p5000 CfgTRES=cpu=64,mem=515617M,billing=64,gres/gpu=6 AllocTRES=cpu=8,mem=200G,gres/gpu=1,gres/gpu:p5000=1
NodeName=cpu-01 Arch=x86_64 CPUAlloc=0 CPUTot=32 Gres=(null) State=IDLE Partitions=cpu CfgTRES=cpu=32,mem=128G,billing=32
@@SQUEUE
l40s
l40s,a5000
a5000
@@SINFO
l40s*
h100
a5000
p5000
cpu
"""


def test_parse_node_reads_gpu_type_count_and_usage():
    node = parse_node(
        "NodeName=dl-02 CPUAlloc=8 Gres=gpu:L40S:4 State=MIXED Partitions=l40s "
        "CfgTRES=cpu=64,gres/gpu=4 AllocTRES=cpu=8,mem=20G,gres/gpu=1,gres/gpu:l40s=1"
    )
    assert node.name == "dl-02"
    assert node.partitions == ("l40s",)
    assert node.gpu_type == "L40S"
    assert node.gpus == 4
    assert node.gpus_used == 1
    assert node.gpus_free == 3
    assert node.usable


def test_parse_node_without_gpus():
    node = parse_node("NodeName=cpu-01 Gres=(null) State=IDLE Partitions=cpu AllocTRES=")
    assert node.gpu_type is None
    assert node.gpus == 0
    assert node.gpus_free == 0


def test_down_node_contributes_no_free_gpus():
    node = parse_node("NodeName=dl-04 Gres=gpu:H100:2 State=DOWN+DRAIN Partitions=h100 AllocTRES=")
    assert not node.usable
    assert node.gpus_free == 0


def test_partition_summary_counts_free_gpus_and_pending_jobs():
    inventory = parse_inventory(SCONTROL)
    by_name = {p.name: p for p in inventory.partitions()}

    l40s = by_name["l40s"]
    assert l40s.is_default
    assert l40s.gpu_type == "L40S"
    assert l40s.gpus_total == 8
    assert l40s.gpus_free == 3  # dl-01: 0 livres, dl-02: 3 livres
    assert l40s.max_node_free == 3
    assert l40s.pending == 2  # "l40s" e "l40s,a5000"

    h100 = by_name["h100"]
    assert h100.nodes == 1
    assert h100.nodes_down == 1
    assert h100.gpus_total == 0
    assert h100.gpus_free == 0


def test_partitions_without_gpus_are_left_out():
    names = {p.name for p in parse_inventory(SCONTROL).partitions()}
    assert "cpu" not in names


def test_partition_lookup_and_defaults():
    inventory = parse_inventory(SCONTROL)
    assert inventory.partition("a5000").gpus_free == 0
    assert inventory.partition("nope") is None
    assert inventory.defaults == {"l40s"}
