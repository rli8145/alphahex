from __future__ import annotations

from dataclasses import dataclass

from catan_engine.resources import Resource
from catan_engine.state import GameState


@dataclass(frozen=True)
class Observation:
    player_id: int
    state: GameState
    own_resources: dict[Resource, int]
    own_dev_card_count: int
    opponent_resource_count: int
    opponent_dev_card_count: int


def make_observation(state: GameState, player_id: int) -> Observation:
    player = state.player(player_id)
    opponent = state.player(state.other_player(player_id))
    return Observation(
        player_id=player_id,
        state=state,
        own_resources=player.resource_dict(),
        own_dev_card_count=len(player.dev_cards) + len(player.new_dev_cards),
        opponent_resource_count=opponent.total_resources(),
        opponent_dev_card_count=len(opponent.dev_cards) + len(opponent.new_dev_cards),
    )
