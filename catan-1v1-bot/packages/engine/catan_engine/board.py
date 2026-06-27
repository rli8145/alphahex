from __future__ import annotations

import math
from dataclasses import dataclass

from catan_engine.resources import Resource


@dataclass(frozen=True)
class Port:
    ratio: int
    resource: Resource | None = None


@dataclass(frozen=True)
class Hex:
    id: int
    q: int
    r: int
    resource: Resource | None
    number: int | None
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class Node:
    id: int
    adjacent_hex_ids: tuple[int, ...]
    adjacent_edge_ids: tuple[int, ...]
    port: Port | None = None
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class Edge:
    id: int
    node_ids: tuple[int, int]


@dataclass(frozen=True)
class Board:
    hexes: dict[int, Hex]
    nodes: dict[int, Node]
    edges: dict[int, Edge]
    robber_hex: int

    def hexes_for_number(self, number: int) -> list[Hex]:
        return [hex_tile for hex_tile in self.hexes.values() if hex_tile.number == number]

    def other_node(self, edge_id: int, node_id: int) -> int:
        a, b = self.edges[edge_id].node_ids
        if node_id == a:
            return b
        if node_id == b:
            return a
        raise ValueError(f"node {node_id} is not on edge {edge_id}")


def standard_board() -> Board:
    """Create a deterministic 19-hex board with explicit topology and ports."""

    coords = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(-q - r)) <= 2
    ]
    coords.sort(key=lambda item: (item[1], item[0]))

    resources = iter(
        [
            Resource.LUMBER,
            Resource.BRICK,
            Resource.WOOL,
            Resource.GRAIN,
            Resource.ORE,
            Resource.WOOL,
            Resource.GRAIN,
            Resource.LUMBER,
            Resource.BRICK,
            Resource.ORE,
            Resource.GRAIN,
            Resource.WOOL,
            Resource.LUMBER,
            Resource.ORE,
            Resource.BRICK,
            Resource.GRAIN,
            Resource.WOOL,
            Resource.LUMBER,
        ]
    )
    numbers = iter([5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11])

    hex_specs: list[tuple[int, int, Resource | None, int | None]] = []
    robber_hex = -1
    for q, r in coords:
        if (q, r) == (0, 0):
            robber_hex = len(hex_specs)
            hex_specs.append((q, r, None, None))
        else:
            hex_specs.append((q, r, next(resources), next(numbers)))

    corner_keys: dict[tuple[float, float], int] = {}
    hex_corner_keys: list[tuple[tuple[float, float], ...]] = []
    for q, r, _resource, _number in hex_specs:
        cx, cy = _hex_center(q, r)
        keys: list[tuple[float, float]] = []
        for corner in range(6):
            angle = math.radians(60 * corner - 30)
            key = (round(cx + math.cos(angle), 6), round(cy + math.sin(angle), 6))
            keys.append(key)
            corner_keys.setdefault(key, -1)
        hex_corner_keys.append(tuple(keys))

    sorted_corners = sorted(corner_keys)
    node_id_by_key = {key: idx for idx, key in enumerate(sorted_corners)}

    hex_node_ids: list[tuple[int, ...]] = [
        tuple(node_id_by_key[key] for key in keys) for keys in hex_corner_keys
    ]

    edge_keys: set[tuple[int, int]] = set()
    for node_ids in hex_node_ids:
        for index, node_id in enumerate(node_ids):
            other = node_ids[(index + 1) % 6]
            edge_keys.add(tuple(sorted((node_id, other))))

    sorted_edge_keys = sorted(edge_keys)
    edge_id_by_nodes = {nodes: idx for idx, nodes in enumerate(sorted_edge_keys)}
    edges = {idx: Edge(id=idx, node_ids=nodes) for nodes, idx in edge_id_by_nodes.items()}

    node_hexes: dict[int, list[int]] = {node_id: [] for node_id in node_id_by_key.values()}
    node_edges: dict[int, list[int]] = {node_id: [] for node_id in node_id_by_key.values()}
    for hex_id, node_ids in enumerate(hex_node_ids):
        for node_id in node_ids:
            node_hexes[node_id].append(hex_id)
    for edge_id, edge in edges.items():
        a, b = edge.node_ids
        node_edges[a].append(edge_id)
        node_edges[b].append(edge_id)

    ports = _assign_ports(sorted_edge_keys, edges, node_hexes, sorted_corners)

    nodes = {
        node_id: Node(
            id=node_id,
            adjacent_hex_ids=tuple(sorted(node_hexes[node_id])),
            adjacent_edge_ids=tuple(sorted(node_edges[node_id])),
            port=ports.get(node_id),
            x=sorted_corners[node_id][0],
            y=sorted_corners[node_id][1],
        )
        for node_id in node_hexes
    }
    hexes = {
        hex_id: Hex(
            id=hex_id,
            q=q,
            r=r,
            resource=resource,
            number=number,
            node_ids=hex_node_ids[hex_id],
        )
        for hex_id, (q, r, resource, number) in enumerate(hex_specs)
    }
    return Board(hexes=hexes, nodes=nodes, edges=edges, robber_hex=robber_hex)


def _hex_center(q: int, r: int) -> tuple[float, float]:
    return (math.sqrt(3) * (q + r / 2), 1.5 * r)


def _assign_ports(
    sorted_edge_keys: list[tuple[int, int]],
    edges: dict[int, Edge],
    node_hexes: dict[int, list[int]],
    sorted_corners: list[tuple[float, float]],
) -> dict[int, Port]:
    boundary_edge_ids = [
        edge_id
        for edge_id, edge in edges.items()
        if len(set(node_hexes[edge.node_ids[0]]).intersection(node_hexes[edge.node_ids[1]])) == 1
    ]

    def edge_angle(edge_id: int) -> float:
        a, b = edges[edge_id].node_ids
        ax, ay = sorted_corners[a]
        bx, by = sorted_corners[b]
        return math.atan2((ay + by) / 2, (ax + bx) / 2)

    boundary_edge_ids.sort(key=edge_angle)
    port_specs = [
        Port(3),
        Port(2, Resource.LUMBER),
        Port(3),
        Port(2, Resource.BRICK),
        Port(2, Resource.WOOL),
        Port(3),
        Port(2, Resource.GRAIN),
        Port(2, Resource.ORE),
        Port(3),
    ]

    ports: dict[int, Port] = {}
    used_nodes: set[int] = set()
    start_indexes = [int(index * len(boundary_edge_ids) / len(port_specs)) for index in range(len(port_specs))]
    for spec, start in zip(port_specs, start_indexes, strict=True):
        for offset in range(len(boundary_edge_ids)):
            edge = edges[boundary_edge_ids[(start + offset) % len(boundary_edge_ids)]]
            if edge.node_ids[0] not in used_nodes and edge.node_ids[1] not in used_nodes:
                ports[edge.node_ids[0]] = spec
                ports[edge.node_ids[1]] = spec
                used_nodes.update(edge.node_ids)
                break
    return ports
