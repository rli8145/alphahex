from __future__ import annotations

import random
from dataclasses import replace

from catan_bots import RandomBot
from catan_engine.actions import Action, ActionType, Phase
from catan_engine.dev_cards import DevCard
from catan_engine.resources import Resource
from catan_engine.rules import (
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    apply_action,
    can_place_settlement,
    get_legal_actions,
)
from catan_engine.scoring import total_points, update_awards
from catan_engine.simulator import play_game
from catan_engine.state import GameConfig, PlayerState, new_game


def test_setup_placement_order() -> None:
    state = new_game(seed=1)
    expected_order = [0, 1, 1, 0]
    for expected_player in expected_order:
        assert state.phase == Phase.SETUP_SETTLEMENT
        assert state.current_player == expected_player
        settlement = _first_action(state, ActionType.PLACE_SETTLEMENT)
        state = apply_action(state, settlement, random.Random(1))
        assert state.phase == Phase.SETUP_ROAD
        assert state.current_player == expected_player
        road = _first_action(state, ActionType.PLACE_ROAD)
        state = apply_action(state, road, random.Random(1))
    assert state.phase == Phase.ROLL
    assert state.current_player == 0


def test_settlement_distance_rule() -> None:
    state = new_game(seed=2)
    settlement = _first_action(state, ActionType.PLACE_SETTLEMENT)
    node_id = settlement.payload["node_id"]
    state = apply_action(state, settlement)
    for edge_id in state.board.nodes[node_id].adjacent_edge_ids:
        adjacent = state.board.other_node(edge_id, node_id)
        assert not can_place_settlement(state, adjacent, setup=True)


def test_road_placement_legality() -> None:
    state = new_game(seed=3)
    edge_id = next(iter(state.board.edges))
    node_id = state.board.edges[edge_id].node_ids[0]
    player = PlayerState(id=0, settlements=frozenset({node_id})).add_resources(ROAD_COST)
    state = replace(state, phase=Phase.MAIN, current_player=0, players=(player, PlayerState(id=1)))

    legal_edges = {action.payload["edge_id"] for action in get_legal_actions(state) if action.type == ActionType.BUILD_ROAD}
    assert edge_id in legal_edges

    far_edge = next(
        candidate
        for candidate, edge in state.board.edges.items()
        if node_id not in edge.node_ids
        and not set(edge.node_ids).intersection(
            {
                state.board.other_node(adjacent_edge, node_id)
                for adjacent_edge in state.board.nodes[node_id].adjacent_edge_ids
            }
        )
    )
    assert far_edge not in legal_edges


def test_resource_distribution_from_dice() -> None:
    state = new_game(seed=4, config=GameConfig(balanced_dice=False))
    hex_tile = _resource_hex(state)
    node_id = hex_tile.node_ids[0]
    player = PlayerState(id=0, settlements=frozenset({node_id}))
    state = replace(state, phase=Phase.ROLL, current_player=0, players=(player, PlayerState(id=1)))

    state = apply_action(state, Action(ActionType.ROLL_DICE, 0, {"roll": hex_tile.number}))
    assert state.player(0).resource_count(hex_tile.resource) == 1


def test_city_produces_two_resources() -> None:
    state = new_game(seed=5, config=GameConfig(balanced_dice=False))
    hex_tile = _resource_hex(state)
    node_id = hex_tile.node_ids[0]
    player = PlayerState(id=0, cities=frozenset({node_id}))
    state = replace(state, phase=Phase.ROLL, current_player=0, players=(player, PlayerState(id=1)))

    state = apply_action(state, Action(ActionType.ROLL_DICE, 0, {"roll": hex_tile.number}))
    assert state.player(0).resource_count(hex_tile.resource) == 2


def test_robber_blocks_production() -> None:
    state = new_game(seed=6, config=GameConfig(balanced_dice=False))
    hex_tile = _resource_hex(state)
    node_id = hex_tile.node_ids[0]
    player = PlayerState(id=0, settlements=frozenset({node_id}))
    state = replace(
        state,
        phase=Phase.ROLL,
        current_player=0,
        players=(player, PlayerState(id=1)),
        robber_hex=hex_tile.id,
    )

    state = apply_action(state, Action(ActionType.ROLL_DICE, 0, {"roll": hex_tile.number}))
    assert state.player(0).resource_count(hex_tile.resource) == 0


def test_seven_discard_threshold_uses_nine_cards() -> None:
    state = new_game(seed=7, config=GameConfig(balanced_dice=False, discard_limit=9))
    p0 = PlayerState(id=0).add_resources({Resource.LUMBER: 10})
    p1 = PlayerState(id=1).add_resources({Resource.BRICK: 9})
    state = replace(state, phase=Phase.ROLL, current_player=0, players=(p0, p1))

    state = apply_action(state, Action(ActionType.ROLL_DICE, 0, {"roll": 7}))
    assert state.phase == Phase.DISCARD
    assert state.pending_discard == (0,)
    discard = _first_action(state, ActionType.DISCARD)
    assert sum(discard.payload["resources"].values()) == 5


def test_friendly_robber_eligibility() -> None:
    state = new_game(seed=8, config=GameConfig(friendly_robber=True))
    protected_hex = _resource_hex(state)
    protected_node = protected_hex.node_ids[0]
    p1 = PlayerState(id=1, settlements=frozenset({protected_node}))
    state = replace(state, phase=Phase.MOVE_ROBBER, current_player=0, players=(PlayerState(id=0), p1))

    legal_hexes = {action.payload["hex_id"] for action in get_legal_actions(state)}
    assert protected_hex.id not in legal_hexes

    extra_node = _non_adjacent_node(state, protected_node)
    p1 = PlayerState(id=1, settlements=frozenset({extra_node}), cities=frozenset({protected_node}))
    state = replace(state, players=(PlayerState(id=0), p1))
    legal_hexes = {action.payload["hex_id"] for action in get_legal_actions(state)}
    assert protected_hex.id in legal_hexes


def test_build_costs() -> None:
    state = new_game(seed=9)
    edge_id = next(iter(state.board.edges))
    node_id = state.board.edges[edge_id].node_ids[0]
    player = PlayerState(id=0, settlements=frozenset({node_id})).add_resources(ROAD_COST)
    state = replace(state, phase=Phase.MAIN, current_player=0, players=(player, PlayerState(id=1)))

    state = apply_action(state, Action(ActionType.BUILD_ROAD, 0, {"edge_id": edge_id}))
    assert edge_id in state.player(0).roads
    assert state.player(0).total_resources() == 0


def test_dev_card_purchase_and_play() -> None:
    state = new_game(seed=10)
    player = PlayerState(id=0).add_resources(DEV_CARD_COST)
    state = replace(
        state,
        phase=Phase.MAIN,
        current_player=0,
        players=(player, PlayerState(id=1)),
        dev_deck=(DevCard.KNIGHT,),
    )

    state = apply_action(state, Action(ActionType.BUY_DEV_CARD, 0))
    assert state.player(0).new_dev_cards == (DevCard.KNIGHT,)
    state = apply_action(state, Action(ActionType.END_TURN, 0))
    assert state.player(0).dev_cards == (DevCard.KNIGHT,)

    state = replace(state, phase=Phase.MAIN, current_player=0)
    knight = _first_action(state, ActionType.PLAY_KNIGHT)
    state = apply_action(state, knight)
    assert state.player(0).played_knights == 1
    assert DevCard.KNIGHT not in state.player(0).dev_cards


def test_longest_road_minimum_five() -> None:
    state = new_game(seed=11)
    path = _edge_path(state, 5)
    p0 = PlayerState(id=0, roads=frozenset(path[:4]))
    state = update_awards(replace(state, players=(p0, PlayerState(id=1))))
    assert state.longest_road_owner is None

    p0 = replace(p0, roads=frozenset(path))
    state = update_awards(replace(state, players=(p0, PlayerState(id=1))))
    assert state.longest_road_owner == 0


def test_largest_army_minimum_three() -> None:
    state = new_game(seed=12)
    state = update_awards(replace(state, players=(replace(state.player(0), played_knights=2), state.player(1))))
    assert state.largest_army_owner is None

    state = update_awards(replace(state, players=(replace(state.player(0), played_knights=3), state.player(1))))
    assert state.largest_army_owner == 0


def test_win_condition_at_fifteen_vp() -> None:
    state = new_game(seed=13)
    nodes = list(state.board.nodes)
    p0 = PlayerState(
        id=0,
        settlements=frozenset(nodes[:5]),
        cities=frozenset(nodes[5:9]),
        dev_cards=(DevCard.VICTORY_POINT,),
    ).add_resources(DEV_CARD_COST)
    state = replace(
        state,
        phase=Phase.MAIN,
        current_player=0,
        players=(p0, PlayerState(id=1)),
        dev_deck=(DevCard.VICTORY_POINT,),
    )
    assert total_points(0, state) == 14

    state = apply_action(state, Action(ActionType.BUY_DEV_CARD, 0))
    assert state.phase == Phase.GAME_OVER
    assert state.winner == 0


def test_random_bot_can_complete_100_games_without_crashing(tmp_path) -> None:
    config = GameConfig(max_turns=50)
    for seed in range(100):
        result = play_game(
            RandomBot(),
            RandomBot(),
            seed=seed,
            replay_dir=tmp_path,
            game_index=seed,
            config=config,
        )
        assert result.replay_path.exists()
        assert result.turns <= config.max_turns + 1


def _first_action(state, action_type: ActionType) -> Action:
    return next(action for action in get_legal_actions(state) if action.type == action_type)


def _resource_hex(state):
    return next(hex_tile for hex_tile in state.board.hexes.values() if hex_tile.resource is not None)


def _non_adjacent_node(state, node_id: int) -> int:
    adjacent = {
        state.board.other_node(edge_id, node_id)
        for edge_id in state.board.nodes[node_id].adjacent_edge_ids
    }
    blocked = adjacent | {node_id}
    return next(candidate for candidate in state.board.nodes if candidate not in blocked)


def _edge_path(state, length: int) -> list[int]:
    graph: dict[int, list[tuple[int, int]]] = {}
    for edge_id, edge in state.board.edges.items():
        a, b = edge.node_ids
        graph.setdefault(a, []).append((b, edge_id))
        graph.setdefault(b, []).append((a, edge_id))

    def search(node_id: int, used: frozenset[int], path: list[int]) -> list[int] | None:
        if len(path) == length:
            return path
        for next_node, edge_id in graph[node_id]:
            if edge_id in used:
                continue
            found = search(next_node, used | {edge_id}, path + [edge_id])
            if found is not None:
                return found
        return None

    for node_id in graph:
        found = search(node_id, frozenset(), [])
        if found is not None:
            return found
    raise AssertionError("no edge path found")
