from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum


class Resource(str, Enum):
    LUMBER = "lumber"
    BRICK = "brick"
    WOOL = "wool"
    GRAIN = "grain"
    ORE = "ore"


ALL_RESOURCES: tuple[Resource, ...] = tuple(Resource)
ResourceCount = dict[Resource, int]


def parse_resource(value: Resource | str) -> Resource:
    if isinstance(value, Resource):
        return value
    return Resource(value)


def empty_resource_count() -> ResourceCount:
    return {resource: 0 for resource in ALL_RESOURCES}


def normalize_resource_count(values: Mapping[Resource | str, int] | None = None) -> ResourceCount:
    counts = empty_resource_count()
    if not values:
        return counts
    for resource, amount in values.items():
        parsed = parse_resource(resource)
        counts[parsed] = int(amount)
    return counts


def resource_count_from_iterable(resources: Iterable[Resource | str]) -> ResourceCount:
    counts = empty_resource_count()
    for resource in resources:
        counts[parse_resource(resource)] += 1
    return counts


def total_resources(values: Mapping[Resource | str, int]) -> int:
    return sum(int(amount) for amount in values.values())


def has_resources(resources: Mapping[Resource | str, int], cost: Mapping[Resource | str, int]) -> bool:
    normalized = normalize_resource_count(resources)
    normalized_cost = normalize_resource_count(cost)
    return all(normalized[resource] >= amount for resource, amount in normalized_cost.items())


def add_resources(resources: Mapping[Resource | str, int], delta: Mapping[Resource | str, int]) -> ResourceCount:
    result = normalize_resource_count(resources)
    for resource, amount in normalize_resource_count(delta).items():
        result[resource] += amount
    return result


def subtract_resources(resources: Mapping[Resource | str, int], cost: Mapping[Resource | str, int]) -> ResourceCount:
    if not has_resources(resources, cost):
        raise ValueError("insufficient resources")
    result = normalize_resource_count(resources)
    for resource, amount in normalize_resource_count(cost).items():
        result[resource] -= amount
    return result


def resource_count_to_json(resources: Mapping[Resource | str, int]) -> dict[str, int]:
    normalized = normalize_resource_count(resources)
    return {resource.value: normalized[resource] for resource in ALL_RESOURCES}
