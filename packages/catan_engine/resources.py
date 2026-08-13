from __future__ import annotations

from enum import Enum


class Resource(str, Enum):
    LUMBER = "LUMBER"
    BRICK = "BRICK"
    WOOL = "WOOL"
    GRAIN = "GRAIN"
    ORE = "ORE"


class HexType(str, Enum):
    LUMBER = "LUMBER"
    BRICK = "BRICK"
    WOOL = "WOOL"
    GRAIN = "GRAIN"
    ORE = "ORE"
    DESERT = "DESERT"


ALL_RESOURCES: tuple[Resource, ...] = tuple(Resource)
RESOURCE_TO_HEX = {
    Resource.LUMBER: HexType.LUMBER,
    Resource.BRICK: HexType.BRICK,
    Resource.WOOL: HexType.WOOL,
    Resource.GRAIN: HexType.GRAIN,
    Resource.ORE: HexType.ORE,
}
HEX_TO_RESOURCE = {hex_type: resource for resource, hex_type in RESOURCE_TO_HEX.items()}


def parse_resource(value: Resource | str) -> Resource:
    if isinstance(value, Resource):
        return value
    return Resource(str(value).upper())


def empty_resource_dict() -> dict[Resource, int]:
    return {resource: 0 for resource in ALL_RESOURCES}


def normalize_resources(values: dict[Resource | str, int] | None = None) -> dict[Resource, int]:
    resources = empty_resource_dict()
    if not values:
        return resources
    for resource, amount in values.items():
        resources[parse_resource(resource)] = int(amount)
    return resources


def resource_dict_to_json(values: dict[Resource, int]) -> dict[str, int]:
    normalized = normalize_resources(values)
    return {resource.name: normalized[resource] for resource in ALL_RESOURCES}


def resource_dict_from_json(values: dict[str, int] | None) -> dict[Resource, int]:
    return normalize_resources(values or {})


def total_resources(values: dict[Resource | str, int]) -> int:
    return sum(normalize_resources(values).values())
