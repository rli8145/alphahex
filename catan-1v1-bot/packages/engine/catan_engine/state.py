from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

from catan_engine.actions import Phase
from catan_engine.board import Board, standard_board
from catan_engine.dev_cards import DevCard, shuffled_dev_deck
from catan_engine.resources import ALL_RESOURCES, Resource, normalize_resource_count, resource_count_to_json

RESOURCE_INDEX = {resource: index for index, resource in enumerate(ALL_RESOURCES)}


@dataclass(frozen=True)
class GameConfig:
    players: int = 2
    target_vp: int = 15
    discard_limit: int = 9
    friendly_robber: bool = True
    balanced_dice: bool = True
    max_roads: int = 15
    max_settlements: int = 5
    max_cities: int = 4
    max_turns: int = 500


@dataclass(frozen=True)
class PlayerState:
    id: int
    resources: tuple[int, ...] = (0, 0, 0, 0, 0)
    dev_cards: tuple[DevCard, ...] = ()
    new_dev_cards: tuple[DevCard, ...] = ()
    settlements: frozenset[int] = frozenset()
    cities: frozenset[int] = frozenset()
    roads: frozenset[int] = frozenset()
    played_knights: int = 0
    dev_card_played_this_turn: bool = False

    def resource_count(self, resource: Resource) -> int:
        return self.resources[RESOURCE_INDEX[resource]]

    def resource_dict(self) -> dict[Resource, int]:
        return {resource: self.resources[RESOURCE_INDEX[resource]] for resource in ALL_RESOURCES}

    def total_resources(self) -> int:
        return sum(self.resources)

    def has_resources(self, cost: dict[Resource, int]) -> bool:
        return all(self.resource_count(resource) >= amount for resource, amount in cost.items())

    def add_resources(self, delta: dict[Resource, int]) -> PlayerState:
        values = list(self.resources)
        for resource, amount in normalize_resource_count(delta).items():
            values[RESOURCE_INDEX[resource]] += amount
        return replace(self, resources=tuple(values))

    def subtract_resources(self, cost: dict[Resource, int]) -> PlayerState:
        if not self.has_resources(cost):
            raise ValueError("insufficient resources")
        values = list(self.resources)
        for resource, amount in normalize_resource_count(cost).items():
            values[RESOURCE_INDEX[resource]] -= amount
        return replace(self, resources=tuple(values))

    def add_resource(self, resource: Resource, amount: int = 1) -> PlayerState:
        return self.add_resources({resource: amount})

    def remove_one_resource(self, resource: Resource) -> PlayerState:
        return self.subtract_resources({resource: 1})

    def add_settlement(self, node_id: int) -> PlayerState:
        return replace(self, settlements=self.settlements | {node_id})

    def add_city(self, node_id: int) -> PlayerState:
        if node_id not in self.settlements:
            raise ValueError("city must upgrade an owned settlement")
        return replace(self, settlements=self.settlements - {node_id}, cities=self.cities | {node_id})

    def add_road(self, edge_id: int) -> PlayerState:
        return replace(self, roads=self.roads | {edge_id})

    def add_dev_card(self, dev_card: DevCard, *, new: bool = True) -> PlayerState:
        if new:
            return replace(self, new_dev_cards=self.new_dev_cards + (dev_card,))
        return replace(self, dev_cards=self.dev_cards + (dev_card,))

    def remove_dev_card(self, dev_card: DevCard) -> PlayerState:
        cards = list(self.dev_cards)
        cards.remove(dev_card)
        return replace(self, dev_cards=tuple(cards))

    def mature_new_dev_cards(self) -> PlayerState:
        return replace(
            self,
            dev_cards=self.dev_cards + self.new_dev_cards,
            new_dev_cards=(),
            dev_card_played_this_turn=False,
        )

    def with_dev_played(self) -> PlayerState:
        return replace(self, dev_card_played_this_turn=True)

    def hidden_vp_cards(self) -> int:
        return sum(1 for card in self.dev_cards + self.new_dev_cards if card == DevCard.VICTORY_POINT)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "resources": resource_count_to_json(self.resource_dict()),
            "dev_cards": [card.value for card in self.dev_cards],
            "new_dev_cards": [card.value for card in self.new_dev_cards],
            "settlements": sorted(self.settlements),
            "cities": sorted(self.cities),
            "roads": sorted(self.roads),
            "played_knights": self.played_knights,
        }


@dataclass(frozen=True)
class GameState:
    config: GameConfig
    board: Board
    players: tuple[PlayerState, PlayerState]
    phase: Phase
    current_player: int
    robber_hex: int
    turn: int = 1
    setup_step: int = 0
    setup_order: tuple[int, ...] = (0, 1, 1, 0)
    pending_setup_node: int | None = None
    pending_discard: tuple[int, ...] = ()
    pending_robber_player: int | None = None
    pending_steal_targets: tuple[int, ...] = ()
    pending_road_building: int = 0
    dev_deck: tuple[DevCard, ...] = ()
    longest_road_owner: int | None = None
    largest_army_owner: int | None = None
    dice_seven_history: tuple[int, ...] = ()
    winner: int | None = None
    last_roll: int | None = None

    def player(self, player_id: int) -> PlayerState:
        return self.players[player_id]

    def replace_player(self, player: PlayerState) -> GameState:
        players = list(self.players)
        players[player.id] = player
        return replace(self, players=tuple(players))

    def other_player(self, player_id: int) -> int:
        return 1 - player_id

    def occupied_nodes(self) -> dict[int, int]:
        occupied: dict[int, int] = {}
        for player in self.players:
            for node_id in player.settlements | player.cities:
                occupied[node_id] = player.id
        return occupied

    def occupied_edges(self) -> dict[int, int]:
        occupied: dict[int, int] = {}
        for player in self.players:
            for edge_id in player.roads:
                occupied[edge_id] = player.id
        return occupied

    def clone(self) -> GameState:
        return replace(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "current_player": self.current_player,
            "turn": self.turn,
            "robber_hex": self.robber_hex,
            "winner": self.winner,
            "last_roll": self.last_roll,
            "longest_road_owner": self.longest_road_owner,
            "largest_army_owner": self.largest_army_owner,
            "players": [player.to_json() for player in self.players],
        }


def new_game(seed: int | None = None, config: GameConfig | None = None, board: Board | None = None) -> GameState:
    rng = random.Random(seed)
    board = board or standard_board()
    config = config or GameConfig()
    if config.players != 2:
        raise ValueError("MVP supports exactly two players")
    players = (PlayerState(id=0), PlayerState(id=1))
    return GameState(
        config=config,
        board=board,
        players=players,
        phase=Phase.SETUP_SETTLEMENT,
        current_player=0,
        robber_hex=board.robber_hex,
        dev_deck=shuffled_dev_deck(rng),
    )
