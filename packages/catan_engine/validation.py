from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable

from catan_engine.actions import Action, ActionType, IllegalActionError, Phase
from catan_engine.board import Board, hex_resource
from catan_engine.dev_cards import DevCard
from catan_engine.resources import ALL_RESOURCES, Resource, normalize_resources
from catan_engine.rules import apply_action, get_legal_actions, is_legal_action
from catan_engine.scoring import calculate_longest_road
from catan_engine.state import GameConfig, GameState, initialize_game


def validate_action(state: GameState, action: Action) -> None:
    if not is_legal_action(state, action):
        raise IllegalActionError(f"illegal action: {action}")


class EngineSmokeFailure(AssertionError):
    pass


DEV_CARD_ACTION_TYPES = {
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
}


def run_engine_smoke_checks() -> list[str]:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("setup resource payout", _check_setup_resource_payout),
        ("robber and friendly robber", _check_robber_and_friendly_robber),
        ("dev-card timing", _check_dev_card_timing),
        ("ports", _check_ports),
        ("discard flow", _check_discard_flow),
        ("longest road", _check_longest_road),
    ]
    passed: list[str] = []
    for name, check in checks:
        check()
        passed.append(name)
    return passed


def _check_setup_resource_payout() -> None:
    state = initialize_game(GameConfig(starting_player=0), seed=0)

    state = apply_action(state, _best_setup_settlement(state))
    _expect(state.players[0].total_resources() == 0, "first setup settlement should not pay resources")
    state = apply_action(state, _legal_action(state, ActionType.PLACE_ROAD))

    state = apply_action(state, _best_setup_settlement(state))
    _expect(state.players[1].total_resources() == 0, "first opponent setup settlement should not pay resources")
    state = apply_action(state, _legal_action(state, ActionType.PLACE_ROAD))

    action = _best_setup_settlement(state)
    node_id = int(action.payload["node_id"])
    expected = _starting_resource_payout(state, node_id)
    before = _resource_snapshot(state.players[1].resources)
    state = apply_action(state, action)
    _expect(
        _resource_delta(before, state.players[1].resources) == expected,
        "second opponent setup settlement paid the wrong resources",
    )
    state = apply_action(state, _legal_action(state, ActionType.PLACE_ROAD))

    action = _best_setup_settlement(state)
    node_id = int(action.payload["node_id"])
    expected = _starting_resource_payout(state, node_id)
    before = _resource_snapshot(state.players[0].resources)
    state = apply_action(state, action)
    _expect(
        _resource_delta(before, state.players[0].resources) == expected,
        "second setup settlement paid the wrong resources",
    )


def _check_robber_and_friendly_robber() -> None:
    state = initialize_game(GameConfig(starting_player=0, friendly_robber=True), seed=0)
    hex_id, node_id, resource, roll_value = _first_productive_hex_and_node(state)
    state.players[1].settlements = {node_id}
    state.players[1].settlements_remaining = 4
    state.players[1].resources = normalize_resources({resource: 1})
    state.phase = Phase.MOVE_ROBBER
    state.current_player = 0
    state.pending_robber_player = 0

    _expect(
        not _has_legal_action(state, ActionType.MOVE_ROBBER, lambda action: int(action.payload["hex_id"]) == hex_id),
        "friendly robber allowed moving onto a protected player",
    )

    friendly_roll = state.clone()
    friendly_roll.phase = Phase.ROLL
    friendly_roll.current_player = 0
    friendly_roll.pending_robber_player = None
    friendly_roll.board.robber_hex_id = hex_id
    before = friendly_roll.players[1].resources[resource]
    friendly_roll = apply_action(friendly_roll, Action(ActionType.ROLL_DICE, 0, {"roll": roll_value}), random.Random(1))
    _expect(
        friendly_roll.players[1].resources[resource] == before + 1,
        "friendly robber should not block protected player production",
    )

    blocked_roll = state.clone()
    blocked_roll.config.friendly_robber = False
    blocked_roll.phase = Phase.ROLL
    blocked_roll.current_player = 0
    blocked_roll.pending_robber_player = None
    blocked_roll.board.robber_hex_id = hex_id
    before = blocked_roll.players[1].resources[resource]
    blocked_roll = apply_action(blocked_roll, Action(ActionType.ROLL_DICE, 0, {"roll": roll_value}), random.Random(1))
    _expect(
        blocked_roll.players[1].resources[resource] == before,
        "normal robber should block production on its hex",
    )

    normal = state.clone()
    normal.config.friendly_robber = False
    move = _legal_action(normal, ActionType.MOVE_ROBBER, lambda action: int(action.payload["hex_id"]) == hex_id)
    normal = apply_action(normal, move)
    _expect(normal.phase == Phase.STEAL, "normal robber move should require stealing from adjacent resource holder")
    _expect(normal.legal_steal_targets == [1], "normal robber produced the wrong steal targets")
    before_thief_total = normal.players[0].total_resources()
    steal = _legal_action(normal, ActionType.STEAL_RESOURCE, lambda action: int(action.payload["target_player"]) == 1)
    normal = apply_action(normal, steal, random.Random(1))
    _expect(normal.phase == Phase.MAIN, "steal should return the robber player to MAIN phase")
    _expect(normal.players[0].total_resources() == before_thief_total + 1, "steal did not transfer one resource")
    _expect(normal.players[1].total_resources() == 0, "steal did not remove one resource from the target")


def _check_dev_card_timing() -> None:
    state = initialize_game(GameConfig(starting_player=0), seed=1)
    state.phase = Phase.MAIN
    state.current_player = 0
    state.dev_card_deck = [DevCard.KNIGHT]
    state.players[0].resources = normalize_resources({Resource.WOOL: 1, Resource.GRAIN: 1, Resource.ORE: 1})

    state = apply_action(state, _legal_action(state, ActionType.BUY_DEV_CARD))
    _expect(state.players[0].new_dev_cards[DevCard.KNIGHT] == 1, "bought dev card was not marked new")
    _expect(state.players[0].dev_cards[DevCard.KNIGHT] == 0, "bought dev card became playable immediately")
    _expect(not _has_legal_action(state, ActionType.PLAY_KNIGHT), "newly bought knight was playable this turn")

    state = apply_action(state, _legal_action(state, ActionType.END_TURN))
    _expect(state.players[0].new_dev_cards[DevCard.KNIGHT] == 0, "new dev cards were not cleared at end turn")
    _expect(state.players[0].dev_cards[DevCard.KNIGHT] == 1, "new dev cards were not promoted at end turn")

    state.phase = Phase.MAIN
    state.current_player = 0
    state.players[0].played_dev_card_this_turn = False
    state.players[0].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    state = apply_action(state, _legal_action(state, ActionType.PLAY_KNIGHT), random.Random(2))
    _expect(state.players[0].dev_cards[DevCard.KNIGHT] == 0, "played knight was not consumed")
    _expect(state.players[0].played_dev_card_this_turn, "playing a dev card did not mark the turn")
    _expect(
        not any(action.action_type in DEV_CARD_ACTION_TYPES for action in get_legal_actions(state)),
        "second dev card was legal in the same turn",
    )


def _check_ports() -> None:
    state = initialize_game(GameConfig(starting_player=0), seed=0)
    node_id = _port_node(state, kind="resource")
    give = state.board.nodes[node_id].port.resource
    _expect(give is not None, "resource port was missing its resource")
    receive = _other_resource(give)
    state.phase = Phase.MAIN
    state.current_player = 0
    state.players[0].settlements = {node_id}
    state.players[0].resources = normalize_resources({give: 2})

    trade = _legal_action(
        state,
        ActionType.MARITIME_TRADE,
        lambda action: action.payload["give"] == give.name
        and int(action.payload["give_count"]) == 2
        and action.payload["receive"] == receive.name,
    )
    state = apply_action(state, trade)
    _expect(state.players[0].resources[give] == 0, "2:1 resource port did not consume two resources")
    _expect(state.players[0].resources[receive] == 1, "2:1 resource port did not grant the received resource")

    state = initialize_game(GameConfig(starting_player=0), seed=0)
    node_id = _port_node(state, kind="generic")
    give = Resource.LUMBER
    receive = _other_resource(give)
    state.phase = Phase.MAIN
    state.current_player = 0
    state.players[0].settlements = {node_id}
    state.players[0].resources = normalize_resources({give: 3})

    trade = _legal_action(
        state,
        ActionType.MARITIME_TRADE,
        lambda action: action.payload["give"] == give.name
        and int(action.payload["give_count"]) == 3
        and action.payload["receive"] == receive.name,
    )
    state = apply_action(state, trade)
    _expect(state.players[0].resources[give] == 0, "3:1 generic port did not consume three resources")
    _expect(state.players[0].resources[receive] == 1, "3:1 generic port did not grant the received resource")


def _check_discard_flow() -> None:
    state = initialize_game(GameConfig(starting_player=0, discard_limit=9), seed=0)
    state.phase = Phase.ROLL
    state.current_player = 0
    state.players[0].resources = normalize_resources({Resource.LUMBER: 10})
    state.players[1].resources = normalize_resources({Resource.BRICK: 11})

    state = apply_action(state, Action(ActionType.ROLL_DICE, 0, {"roll": 7}), random.Random(3))
    _expect(state.phase == Phase.DISCARD, "rolling seven with oversized hands should enter DISCARD")
    _expect(state.pending_discards == {0, 1}, "rolling seven queued the wrong players for discard")
    _expect(state.current_player == 0, "discard flow should start with the lowest pending player id")

    state = apply_action(state, _legal_action(state, ActionType.DISCARD))
    _expect(state.players[0].total_resources() == 5, "first discard removed the wrong number of resources")
    _expect(state.pending_discards == {1}, "first discard did not leave only the second player pending")
    _expect(state.current_player == 1 and state.phase == Phase.DISCARD, "discard flow did not advance to the second player")

    state = apply_action(state, _legal_action(state, ActionType.DISCARD))
    _expect(state.players[1].total_resources() == 6, "second discard removed the wrong number of resources")
    _expect(not state.pending_discards, "discard flow left stale pending discards")
    _expect(state.current_player == 0 and state.phase == Phase.MOVE_ROBBER, "discard flow did not return to robber movement")

    state = apply_action(state, _legal_action(state, ActionType.MOVE_ROBBER))
    _expect(state.phase == Phase.MAIN, "robber move without steal targets should enter MAIN")


def _check_longest_road() -> None:
    state = initialize_game(GameConfig(starting_player=0, target_vp=15), seed=0)
    path0_nodes, path0_edges = _find_simple_path(state.board, 5)
    path1_nodes, path1_edges = _find_simple_path(state.board, 6, excluded_nodes=set(path0_nodes))

    state.phase = Phase.MAIN
    state.current_player = 0
    state.players[0].settlements = {path0_nodes[0]}
    state.players[0].settlements_remaining = 4
    state.players[0].resources = normalize_resources({Resource.LUMBER: 20, Resource.BRICK: 20})
    state.players[1].settlements = {path1_nodes[0]}
    state.players[1].settlements_remaining = 4
    state.players[1].resources = normalize_resources({Resource.LUMBER: 20, Resource.BRICK: 20})

    for edge_id in path0_edges:
        state.current_player = 0
        state = apply_action(
            state,
            _legal_action(state, ActionType.BUILD_ROAD, lambda action, edge_id=edge_id: int(action.payload["edge_id"]) == edge_id),
        )
    _expect(calculate_longest_road(state.board, state, 0) == 5, "player 0 longest road length should be five")
    _expect(state.longest_road_owner == 0, "first five-road route should claim longest road")

    for edge_id in path1_edges[:5]:
        state.current_player = 1
        state = apply_action(
            state,
            _legal_action(state, ActionType.BUILD_ROAD, lambda action, edge_id=edge_id: int(action.payload["edge_id"]) == edge_id),
        )
    _expect(calculate_longest_road(state.board, state, 1) == 5, "player 1 longest road length should be five")
    _expect(state.longest_road_owner == 0, "equal longest road should not transfer the award")

    state.current_player = 1
    state = apply_action(
        state,
        _legal_action(
            state,
            ActionType.BUILD_ROAD,
            lambda action: int(action.payload["edge_id"]) == path1_edges[5],
        ),
    )
    _expect(calculate_longest_road(state.board, state, 1) == 6, "player 1 longest road length should be six")
    _expect(state.longest_road_owner == 1, "strictly longer road should transfer the award")


def _best_setup_settlement(state: GameState) -> Action:
    actions = [action for action in get_legal_actions(state) if action.action_type == ActionType.PLACE_SETTLEMENT]
    _expect(bool(actions), "expected at least one setup settlement action")
    return max(
        actions,
        key=lambda action: (
            sum(_starting_resource_payout(state, int(action.payload["node_id"])).values()),
            -int(action.payload["node_id"]),
        ),
    )


def _legal_action(
    state: GameState,
    action_type: ActionType,
    predicate: Callable[[Action], bool] | None = None,
) -> Action:
    for action in get_legal_actions(state):
        if action.action_type == action_type and (predicate is None or predicate(action)):
            return action
    raise EngineSmokeFailure(f"expected legal {action_type.name} action in {state.phase.name}")


def _has_legal_action(
    state: GameState,
    action_type: ActionType,
    predicate: Callable[[Action], bool] | None = None,
) -> bool:
    return any(
        action.action_type == action_type and (predicate is None or predicate(action))
        for action in get_legal_actions(state)
    )


def _starting_resource_payout(state: GameState, node_id: int) -> dict[Resource, int]:
    resources = normalize_resources()
    for hex_id in state.board.get_hexes_for_node(node_id):
        resource = hex_resource(state.board.hexes[hex_id].hex_type)
        if resource is not None:
            resources[resource] += 1
    return resources


def _first_productive_hex_and_node(state: GameState) -> tuple[int, int, Resource, int]:
    for hex_id, hex_tile in sorted(state.board.hexes.items()):
        resource = hex_resource(hex_tile.hex_type)
        if resource is not None and hex_tile.number_token is not None:
            return hex_id, hex_tile.node_ids[0], resource, int(hex_tile.number_token)
    raise EngineSmokeFailure("expected a productive hex")


def _port_node(state: GameState, *, kind: str) -> int:
    for node_id, node in sorted(state.board.nodes.items()):
        if node.port is not None and node.port.kind == kind:
            return node_id
    raise EngineSmokeFailure(f"expected a {kind} port")


def _other_resource(resource: Resource) -> Resource:
    for candidate in ALL_RESOURCES:
        if candidate != resource:
            return candidate
    raise EngineSmokeFailure("expected another resource")


def _find_simple_path(
    board: Board,
    edge_count: int,
    excluded_nodes: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    excluded = excluded_nodes or set()

    def search(path_nodes: list[int], path_edges: list[int]) -> tuple[list[int], list[int]] | None:
        if len(path_edges) == edge_count:
            return path_nodes, path_edges
        node_id = path_nodes[-1]
        for edge_id in board.get_edges_for_node(node_id):
            if edge_id in path_edges:
                continue
            next_node = board.get_opposite_node(edge_id, node_id)
            if next_node in excluded or next_node in path_nodes:
                continue
            result = search([*path_nodes, next_node], [*path_edges, edge_id])
            if result is not None:
                return result
        return None

    for node_id in sorted(board.nodes):
        if node_id in excluded:
            continue
        result = search([node_id], [])
        if result is not None:
            return result
    raise EngineSmokeFailure(f"expected a simple path with {edge_count} edges")


def _resource_snapshot(resources: dict[Resource, int]) -> dict[Resource, int]:
    return {resource: resources.get(resource, 0) for resource in ALL_RESOURCES}


def _resource_delta(before: dict[Resource, int], after: dict[Resource, int]) -> dict[Resource, int]:
    return {resource: after.get(resource, 0) - before.get(resource, 0) for resource in ALL_RESOURCES}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise EngineSmokeFailure(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic engine correctness smoke checks.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        passed = run_engine_smoke_checks()
    except EngineSmokeFailure as exc:
        print(f"engine correctness smoke check failed: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("engine correctness smoke checks passed:")
        for name in passed:
            print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
