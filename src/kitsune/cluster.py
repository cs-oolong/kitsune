"""Visão do cluster: GPUs livres por partição, nós fora do ar e jobs na fila."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .remote import run

UNUSABLE_STATES = ("DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPONDING")
NODE_FIELDS = re.compile(r"(?:^|\s)(NodeName|Partitions|State|Gres|AllocTRES)=(\S*)")
GPU_GRES = re.compile(r"gpu(?::([A-Za-z0-9]+))?:(\d+)")
GPU_ALLOC = re.compile(r"(?:^|,)gres/gpu=(\d+)")
INVENTORY_COMMAND = (
    "scontrol -o show nodes; echo @@SQUEUE; squeue -h -t PD -o '%P'; echo @@SINFO; sinfo -h -o '%P'"
)


@dataclass(frozen=True)
class Node:
    name: str
    partitions: tuple[str, ...]
    state: str
    gpu_type: str | None
    gpus: int
    gpus_used: int

    @property
    def usable(self) -> bool:
        return not any(flag in self.state for flag in UNUSABLE_STATES)

    @property
    def gpus_free(self) -> int:
        if not self.usable:
            return 0
        return max(self.gpus - self.gpus_used, 0)


@dataclass(frozen=True)
class Partition:
    name: str
    gpu_type: str | None
    is_default: bool
    nodes: int
    nodes_down: int
    gpus_total: int
    gpus_free: int
    max_node_gpus: int
    max_node_free: int
    pending: int


@dataclass(frozen=True)
class Inventory:
    nodes: tuple[Node, ...]
    pending: dict[str, int]
    defaults: frozenset[str]

    def partitions(self) -> list[Partition]:
        names = sorted({name for node in self.nodes for name in node.partitions})
        result = []
        for name in names:
            members = [node for node in self.nodes if name in node.partitions]
            if not any(node.gpus for node in members):
                continue
            usable = [node for node in members if node.usable]
            types = Counter(node.gpu_type for node in members if node.gpu_type)
            result.append(
                Partition(
                    name=name,
                    gpu_type=types.most_common(1)[0][0] if types else None,
                    is_default=name in self.defaults,
                    nodes=len(members),
                    nodes_down=len(members) - len(usable),
                    gpus_total=sum(node.gpus for node in usable),
                    gpus_free=sum(node.gpus_free for node in usable),
                    max_node_gpus=max((node.gpus for node in usable), default=0),
                    max_node_free=max((node.gpus_free for node in usable), default=0),
                    pending=self.pending.get(name, 0),
                )
            )
        return result

    def partition(self, name: str) -> Partition | None:
        return next((p for p in self.partitions() if p.name == name), None)


def parse_node(line: str) -> Node:
    fields = dict(NODE_FIELDS.findall(line))
    gres = GPU_GRES.search(fields.get("Gres", ""))
    alloc = GPU_ALLOC.search(fields.get("AllocTRES", ""))
    return Node(
        name=fields.get("NodeName", ""),
        partitions=tuple(p for p in fields.get("Partitions", "").split(",") if p),
        state=fields.get("State", ""),
        gpu_type=gres.group(1) if gres and gres.group(1) else None,
        gpus=int(gres.group(2)) if gres else 0,
        gpus_used=int(alloc.group(1)) if alloc else 0,
    )


def parse_inventory(text: str) -> Inventory:
    nodes_text, _, rest = text.partition("@@SQUEUE")
    queue_text, _, sinfo_text = rest.partition("@@SINFO")
    nodes = tuple(parse_node(line) for line in nodes_text.splitlines() if line.startswith("NodeName="))
    pending: Counter[str] = Counter()
    for line in queue_text.splitlines():
        for name in line.strip().split(","):
            if name:
                pending[name] += 1
    defaults = frozenset(
        line.strip().rstrip("*") for line in sinfo_text.splitlines() if line.strip().endswith("*")
    )
    return Inventory(nodes=nodes, pending=dict(pending), defaults=defaults)


def fetch_inventory() -> Inventory:
    return parse_inventory(run(INVENTORY_COMMAND).stdout)
