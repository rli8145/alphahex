from __future__ import annotations

from catan_engine.scoring import visible_vp
from catan_engine.state import GameState


def create_observation(state: GameState, player_id: int) -> dict:
    opponent_id = state.opponent_id(player_id)
    player = state.players[player_id]
    opponent = state.players[opponent_id]
    return {
        "player_id": player_id,
        "board": state.board.to_dict(),
        "phase": state.phase.name,
        "current_player": state.current_player,
        "turn_number": state.turn_number,
        "dice_roll": state.dice_roll,
        "robber_hex_id": state.board.robber_hex_id,
        "own_resources": {resource.name: amount for resource, amount in player.resources.items()},
        "own_dev_cards": {card.name: amount for card, amount in player.dev_cards.items()},
        "own_new_dev_cards": {card.name: amount for card, amount in player.new_dev_cards.items()},
        "own_settlements": sorted(player.settlements),
        "own_cities": sorted(player.cities),
        "own_roads": sorted(player.roads),
        "opponent_resource_count": opponent.total_resources(),
        "opponent_dev_card_count": sum(opponent.dev_cards.values()) + sum(opponent.new_dev_cards.values()),
        "opponent_visible_vp": visible_vp(state, opponent_id),
        "public_players": [
            {
                "player_id": index,
                "settlements": sorted(public_player.settlements),
                "cities": sorted(public_player.cities),
                "roads": sorted(public_player.roads),
                "played_knights": public_player.played_knights,
                "visible_vp": visible_vp(state, index),
                "resource_count": public_player.total_resources(),
                "dev_card_count": sum(public_player.dev_cards.values()) + sum(public_player.new_dev_cards.values()),
            }
            for index, public_player in enumerate(state.players)
        ],
        "longest_road_owner": state.longest_road_owner,
        "largest_army_owner": state.largest_army_owner,
        "winner": state.winner,
    }
