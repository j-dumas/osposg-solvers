from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Host:
    name: str
    impact: float


@dataclass(frozen=True)
class Zone:
    name: str
    detection: float
    hosts: list[str]


class Edge(NamedTuple):
    src: str
    dst: str
    vuln: str


@dataclass(frozen=True)
class Control:
    name: str
    cost: float
    mitigations: dict[str, float]


@dataclass(frozen=True)
class Vuln:
    name: str
    success: float


@dataclass(frozen=True)
class Spec:
    hosts: dict[str, Host]
    zones: dict[str, Zone]
    edges: list[Edge]
    vulns: dict[str, Vuln]
    controls: dict[str, Control]
    foothold: dict[str, float]
    gamma: float

class SpecError(ValueError):
    pass
