from __future__ import annotations

import random
from dataclasses import replace

from catan_engine.actions import Action, ActionType, IllegalActionError, Phase, action_key
from catan_engine.dev_cards import DevCard
from catan_engine.dice import dice_for_config
from catan_engine.resources import ALL_RESOURCES, Resource, normalize_resource_count
from catan_engine.scoring import total_points, update_awards, visible_points
from catan_engine.state import GameState, PlayerState

ROAD_COST = {Resource.LUMBER: 1, Resource.BRICK: 1}
SETTLEMENT_COST = {
    Resource.LUMBER: 1,
    Resource.BRICK: 1,
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
}
CITY_COST = {Resource.GRAIN: 2, Resource.ORE: 3}
DEV_CARD_COST = {Resource.WOOL: 1, Resource.GRAIN: 1, Resource.ORE: 1}


def get_legal_actions(state: GameState) -> list[Action]:
    if state.phase == Phase.GAME_OVER:
        return []
    if state.phase == Phase.SETUP_SETTLEMENT:
        return _setup_settlement_actions(state)
    if state.phase == Phase.SETUP_ROAD:
        return _setup_road_actions(state)
    if state.phase == Phase.ROLL:
        return [Action(ActionType.ROLL_DICE, state.current_player)]
    if state.phase == Phase.DISCARD:
        return _discard_actions(state)
    if state.phase == Phase.MOVE_ROBBER:
        return _move_robber_actions(state, state.current_player)
    if state.phase == Phase.STEAL:
        return [
            Action(ActionType.STEAL_RESOURCE, state.current_player, {"target_player": target})
            for target in state.pending_steal_targets
        ]
    if state.phase == Phase.MAIN:
        return _main_actions(state)
    raise ValueError(f"unknown phase {state.phase}")


def apply_action(state: GameState, action: Action, rng: random.Random | None = None) -> GameState:
    rng = rng or random.Random(0)
    if not is_legal_action(state, action):
        raise IllegalActionError(f"illegal action: {action}")

    match action.type:
        case ActionType.PLACE_SETTLEMENT:
            return _apply_place_settlement(state, action)
        case ActionType.PLACE_ROAD:
            return _apply_place_setup_road(state, action)
        case ActionType.ROLL_DICE:
            return _apply_roll_dice(state, action, rng)
        case ActionType.DISCARD:
            return _apply_discard(state, action)
        case ActionType.MOVE_ROBBER:
            return _apply_move_robber(state, action)
        case ActionType.STEAL_RESOURCE:
            return _apply_steal(state, action, rng)
        case ActionType.BUILD_ROAD:
            return _apply_build_road(state, action)
        case ActionType.BUILD_SETTLEMENT:
            return _apply_build_settlement(state, action)
        case ActionType.BUILD_CITY:
            return _apply_build_city(state, action)
        case ActionType.BUY_DEV_CARD:
            return _apply_buy_dev_card(state, action)
        case ActionType.PLAY_KNIGHT:
            return _apply_play_knight(state, action)
        case ActionType.PLAY_ROAD_BUILDING:
            return _apply_play_road_building(state, action)
        case ActionType.PLAY_YEAR_OF_PLENTY:
            return _apply_play_year_of_plenty(state, action)
        case ActionType.PLAY_MONOPOLY:
            return _apply_play_monopoly(state, action)
        case ActionType.MARITIME_TRADE:
            return _apply_maritime_trade(state, action)
        case ActionType.END_TURN:
            return _apply_end_turn(state)
    raise ValueError(f"unhandled action {action.type}")


def is_legal_action(state: GameState, action: Action) -> bool:
    legal_actions = get_legal_actions(state)
    if action_key(action) in {action_key(legal_action) for legal_action in legal_actions}:
        return True
    if action.type == ActionType.ROLL_DICE:
        return any(
            legal_action.type == ActionType.ROLL_DICE and legal_action.player_id == action.player_id
            for legal_action in legal_actions
        )
    if action.type == ActionType.STEAL_RESOURCE:
        return any(
            legal_action.type == ActionType.STEAL_RESOURCE
            and legal_action.player_id == action.player_id
            and legal_action.payload.get("target_player") == action.payload.get("target_player")
            for legal_action in legal_actions
        )
    return False


def _setup_settlement_actions(state: GameState) -> list[Action]:
    player_id = state.current_player
    return [
        Action(ActionType.PLACE_SETTLEMENT, player_id, {"node_id": node_id})
        for node_id in sorted(state.board.nodes)
        if can_place_settlement(state, node_id, setup=True)
    ]


def _setup_road_actions(state: GameState) -> list[Action]:
    if state.pending_setup_node is None:
        return []
    occupied_edges = state.occupied_edges()
    return [
        Action(ActionType.PLACE_ROAD, state.current_player, {"edge_id": edge_id})
        for edge_id in state.board.nodes[state.pending_setup_node].adjacent_edge_ids
        if edge_id not in occupied_edges
    ]


def _discard_actions(state: GameState) -> list[Action]:
    if not state.pending_discard:
        return []
    player_id = state.pending_discard[0]
    player = state.player(player_id)
    discard_amount = player.total_resources() // 2
    return [
        Action(ActionType.DISCARD, player_id, {"resources": {resource.value: amount for resource, amount in combo.items()}})
        for combo in _discard_combinations(player, discard_amount)
    ]


def _move_robber_actions(state: GameState, player_id: int) -> list[Action]:
    actions = [
        Action(ActionType.MOVE_ROBBER, player_id, {"hex_id": hex_id})
        for hex_id in sorted(state.board.hexes)
        if hex_id != state.robber_hex and _robber_hex_allowed(state, player_id, hex_id)
    ]
    return actions


def _main_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    actions: list[Action] = []

    if state.pending_road_building > 0:
        road_actions = _build_road_actions(state, free=True)
        if road_actions:
            return road_actions
        return [Action(ActionType.END_TURN, state.current_player)]

    actions.extend(_build_city_actions(state))
    actions.extend(_build_settlement_actions(state))
    actions.extend(_build_road_actions(state, free=False))
    actions.extend(_buy_dev_card_actions(state))
    actions.extend(_dev_card_actions(state))
    actions.extend(_maritime_trade_actions(state))
    actions.append(Action(ActionType.END_TURN, player.id))
    return actions


def _build_road_actions(state: GameState, *, free: bool) -> list[Action]:
    player = state.player(state.current_player)
    if not free and not player.has_resources(ROAD_COST):
        return []
    if len(player.roads) >= state.config.max_roads:
        return []
    return [
        Action(ActionType.BUILD_ROAD, player.id, {"edge_id": edge_id})
        for edge_id in sorted(state.board.edges)
        if can_build_road(state, player.id, edge_id)
    ]


def _build_settlement_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    if not player.has_resources(SETTLEMENT_COST):
        return []
    if len(player.settlements) >= state.config.max_settlements:
        return []
    return [
        Action(ActionType.BUILD_SETTLEMENT, player.id, {"node_id": node_id})
        for node_id in sorted(state.board.nodes)
        if can_place_settlement(state, node_id, setup=False, player_id=player.id)
    ]


def _build_city_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    if not player.has_resources(CITY_COST):
        return []
    if len(player.cities) >= state.config.max_cities:
        return []
    return [
        Action(ActionType.BUILD_CITY, player.id, {"node_id": node_id})
        for node_id in sorted(player.settlements)
    ]


def _buy_dev_card_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    if state.dev_deck and player.has_resources(DEV_CARD_COST):
        return [Action(ActionType.BUY_DEV_CARD, player.id)]
    return []


def _dev_card_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    if player.dev_card_played_this_turn:
        return []
    actions: list[Action] = []
    cards = set(player.dev_cards)
    if DevCard.KNIGHT in cards:
        for robber_action in _move_robber_actions(state, player.id):
            actions.append(
                Action(ActionType.PLAY_KNIGHT, player.id, {"hex_id": robber_action.payload["hex_id"]})
            )
    if DevCard.ROAD_BUILDING in cards:
        actions.append(Action(ActionType.PLAY_ROAD_BUILDING, player.id))
    if DevCard.YEAR_OF_PLENTY in cards:
        for first in ALL_RESOURCES:
            for second in ALL_RESOURCES:
                actions.append(
                    Action(
                        ActionType.PLAY_YEAR_OF_PLENTY,
                        player.id,
                        {"resources": [first.value, second.value]},
                    )
                )
    if DevCard.MONOPOLY in cards:
        for resource in ALL_RESOURCES:
            actions.append(Action(ActionType.PLAY_MONOPOLY, player.id, {"resource": resource.value}))
    return actions


def _maritime_trade_actions(state: GameState) -> list[Action]:
    player = state.player(state.current_player)
    actions: list[Action] = []
    for give in ALL_RESOURCES:
        ratio = maritime_trade_ratio(state, player.id, give)
        if player.resource_count(give) < ratio:
            continue
        for receive in ALL_RESOURCES:
            if receive != give:
                actions.append(
                    Action(
                        ActionType.MARITIME_TRADE,
                        player.id,
                        {"give": give.value, "receive": receive.value, "ratio": ratio},
                    )
                )
    return actions


def can_place_settlement(
    state: GameState,
    node_id: int,
    *,
    setup: bool,
    player_id: int | None = None,
) -> bool:
    if node_id in state.occupied_nodes():
        return False
    for edge_id in state.board.nodes[node_id].adjacent_edge_ids:
        other = state.board.other_node(edge_id, node_id)
        if other in state.occupied_nodes():
            return False
    if setup:
        return True
    if player_id is None:
        raise ValueError("player_id is required outside setup")
    return any(edge_id in state.player(player_id).roads for edge_id in state.board.nodes[node_id].adjacent_edge_ids)


def can_build_road(state: GameState, player_id: int, edge_id: int) -> bool:
    if edge_id in state.occupied_edges():
        return False
    player = state.player(player_id)
    occupied_nodes = state.occupied_nodes()
    edge = state.board.edges[edge_id]
    for node_id in edge.node_ids:
        owner = occupied_nodes.get(node_id)
        if owner == player_id:
            return True
        if owner is not None and owner != player_id:
            continue
        if any(adjacent_edge in player.roads for adjacent_edge in state.board.nodes[node_id].adjacent_edge_ids):
            return True
    return False


def maritime_trade_ratio(state: GameState, player_id: int, resource: Resource) -> int:
    player = state.player(player_id)
    ratio = 4
    for node_id in player.settlements | player.cities:
        port = state.board.nodes[node_id].port
        if port is None:
            continue
        if port.resource is None:
            ratio = min(ratio, port.ratio)
        elif port.resource == resource:
            ratio = min(ratio, port.ratio)
    return ratio


def _apply_place_settlement(state: GameState, action: Action) -> GameState:
    node_id = int(action.payload["node_id"])
    player = state.player(action.player_id).add_settlement(node_id)
    if state.setup_step >= 2:
        player = player.add_resources(_starting_resources(state, node_id))
    state = state.replace_player(player)
    return replace(state, phase=Phase.SETUP_ROAD, pending_setup_node=node_id)


def _apply_place_setup_road(state: GameState, action: Action) -> GameState:
    edge_id = int(action.payload["edge_id"])
    player = state.player(action.player_id).add_road(edge_id)
    state = state.replace_player(player)
    next_setup_step = state.setup_step + 1
    if next_setup_step >= len(state.setup_order):
        return replace(
            state,
            phase=Phase.ROLL,
            current_player=0,
            setup_step=next_setup_step,
            pending_setup_node=None,
        )
    return replace(
        state,
        phase=Phase.SETUP_SETTLEMENT,
        current_player=state.setup_order[next_setup_step],
        setup_step=next_setup_step,
        pending_setup_node=None,
    )


def _apply_roll_dice(state: GameState, action: Action, rng: random.Random) -> GameState:
    roll = int(action.payload.get("roll", dice_for_config(state).roll(action.player_id, state, rng)))
    state = replace(state, last_roll=roll)
    if roll == 7:
        pending = tuple(player.id for player in state.players if player.total_resources() > state.config.discard_limit)
        state = replace(
            state,
            dice_seven_history=(state.dice_seven_history + (action.player_id,))[-24:],
            pending_discard=pending,
            pending_robber_player=action.player_id,
        )
        if pending:
            return replace(state, phase=Phase.DISCARD, current_player=pending[0])
        return replace(state, phase=Phase.MOVE_ROBBER, current_player=action.player_id)
    state = _distribute_resources(state, roll)
    return _maybe_win(replace(state, phase=Phase.MAIN, current_player=action.player_id), action.player_id)


def _apply_discard(state: GameState, action: Action) -> GameState:
    resources = normalize_resource_count(action.payload["resources"])
    player = state.player(action.player_id).subtract_resources(resources)
    state = state.replace_player(player)
    pending = tuple(player_id for player_id in state.pending_discard if player_id != action.player_id)
    if pending:
        return replace(state, pending_discard=pending, current_player=pending[0])
    robber_player = state.pending_robber_player
    if robber_player is None:
        raise ValueError("missing robber player after discard")
    return replace(
        state,
        pending_discard=(),
        phase=Phase.MOVE_ROBBER,
        current_player=robber_player,
    )


def _apply_move_robber(state: GameState, action: Action) -> GameState:
    hex_id = int(action.payload["hex_id"])
    mover = action.player_id
    targets = _steal_targets_for_hex(state, mover, hex_id)
    state = replace(
        state,
        robber_hex=hex_id,
        pending_robber_player=None,
        pending_steal_targets=targets,
        current_player=mover,
    )
    if targets:
        return replace(state, phase=Phase.STEAL)
    return _maybe_win(replace(state, phase=Phase.MAIN), mover)


def _apply_steal(state: GameState, action: Action, rng: random.Random) -> GameState:
    target_id = int(action.payload["target_player"])
    target = state.player(target_id)
    thief = state.player(action.player_id)
    stolen_resource = action.payload.get("resource")
    if stolen_resource is None:
        pool = [
            resource
            for resource in ALL_RESOURCES
            for _ in range(target.resource_count(resource))
        ]
        if pool:
            stolen = rng.choice(pool)
            target = target.remove_one_resource(stolen)
            thief = thief.add_resource(stolen)
    else:
        stolen = Resource(stolen_resource)
        target = target.remove_one_resource(stolen)
        thief = thief.add_resource(stolen)
    state = state.replace_player(target).replace_player(thief)
    return _maybe_win(replace(state, phase=Phase.MAIN, pending_steal_targets=()), action.player_id)


def _apply_build_road(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id)
    if state.pending_road_building == 0:
        player = player.subtract_resources(ROAD_COST)
    player = player.add_road(int(action.payload["edge_id"]))
    state = state.replace_player(player)
    pending = max(0, state.pending_road_building - 1)
    state = replace(state, pending_road_building=pending)
    state = update_awards(state)
    return _maybe_win(state, action.player_id)


def _apply_build_settlement(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id).subtract_resources(SETTLEMENT_COST)
    player = player.add_settlement(int(action.payload["node_id"]))
    state = update_awards(state.replace_player(player))
    return _maybe_win(state, action.player_id)


def _apply_build_city(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id).subtract_resources(CITY_COST)
    player = player.add_city(int(action.payload["node_id"]))
    state = state.replace_player(player)
    return _maybe_win(state, action.player_id)


def _apply_buy_dev_card(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id).subtract_resources(DEV_CARD_COST)
    dev_card = state.dev_deck[0]
    player = player.add_dev_card(dev_card, new=True)
    state = replace(state.replace_player(player), dev_deck=state.dev_deck[1:])
    return _maybe_win(state, action.player_id)


def _apply_play_knight(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id).remove_dev_card(DevCard.KNIGHT)
    player = replace(player, played_knights=player.played_knights + 1).with_dev_played()
    state = update_awards(state.replace_player(player))
    hex_id = int(action.payload["hex_id"])
    targets = _steal_targets_for_hex(state, action.player_id, hex_id)
    state = replace(state, robber_hex=hex_id, pending_steal_targets=targets)
    if targets:
        return _maybe_win(replace(state, phase=Phase.STEAL), action.player_id)
    return _maybe_win(state, action.player_id)


def _apply_play_road_building(state: GameState, action: Action) -> GameState:
    player = state.player(action.player_id).remove_dev_card(DevCard.ROAD_BUILDING).with_dev_played()
    state = state.replace_player(player)
    road_actions = _build_road_actions(replace(state, pending_road_building=2), free=True)
    pending = 2 if road_actions else 0
    return replace(state, pending_road_building=pending)


def _apply_play_year_of_plenty(state: GameState, action: Action) -> GameState:
    resources = [Resource(value) for value in action.payload["resources"]]
    player = state.player(action.player_id).remove_dev_card(DevCard.YEAR_OF_PLENTY).with_dev_played()
    for resource in resources:
        player = player.add_resource(resource)
    state = state.replace_player(player)
    return _maybe_win(state, action.player_id)


def _apply_play_monopoly(state: GameState, action: Action) -> GameState:
    resource = Resource(action.payload["resource"])
    player = state.player(action.player_id).remove_dev_card(DevCard.MONOPOLY).with_dev_played()
    opponent = state.player(state.other_player(action.player_id))
    amount = opponent.resource_count(resource)
    if amount:
        opponent = opponent.subtract_resources({resource: amount})
        player = player.add_resources({resource: amount})
    state = state.replace_player(opponent).replace_player(player)
    return _maybe_win(state, action.player_id)


def _apply_maritime_trade(state: GameState, action: Action) -> GameState:
    give = Resource(action.payload["give"])
    receive = Resource(action.payload["receive"])
    ratio = int(action.payload["ratio"])
    player = state.player(action.player_id).subtract_resources({give: ratio}).add_resource(receive)
    return state.replace_player(player)


def _apply_end_turn(state: GameState) -> GameState:
    if total_points(state.current_player, state) >= state.config.target_vp:
        return replace(state, phase=Phase.GAME_OVER, winner=state.current_player)
    player = state.player(state.current_player).mature_new_dev_cards()
    state = state.replace_player(player)
    next_player = state.other_player(state.current_player)
    return replace(
        state,
        phase=Phase.ROLL,
        current_player=next_player,
        pending_road_building=0,
        turn=state.turn + 1,
    )


def _starting_resources(state: GameState, node_id: int) -> dict[Resource, int]:
    resources: dict[Resource, int] = {}
    for hex_id in state.board.nodes[node_id].adjacent_hex_ids:
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.resource is not None:
            resources[hex_tile.resource] = resources.get(hex_tile.resource, 0) + 1
    return resources


def _distribute_resources(state: GameState, roll: int) -> GameState:
    for hex_tile in state.board.hexes_for_number(roll):
        if hex_tile.id == state.robber_hex or hex_tile.resource is None:
            continue
        for player in state.players:
            amount = 0
            for node_id in hex_tile.node_ids:
                if node_id in player.settlements:
                    amount += 1
                elif node_id in player.cities:
                    amount += 2
            if amount:
                state = state.replace_player(player.add_resources({hex_tile.resource: amount}))
    return state


def _discard_combinations(player: PlayerState, amount: int) -> list[dict[Resource, int]]:
    resources = player.resource_dict()
    combos: list[dict[Resource, int]] = []

    def search(index: int, remaining: int, current: dict[Resource, int]) -> None:
        if index == len(ALL_RESOURCES):
            if remaining == 0:
                combos.append(dict(current))
            return
        resource = ALL_RESOURCES[index]
        for count in range(min(resources[resource], remaining) + 1):
            current[resource] = count
            search(index + 1, remaining - count, current)
        current.pop(resource, None)

    search(0, amount, {})
    return combos


def _robber_hex_allowed(state: GameState, player_id: int, hex_id: int) -> bool:
    if not state.config.friendly_robber:
        return True
    for node_id in state.board.hexes[hex_id].node_ids:
        owner = state.occupied_nodes().get(node_id)
        if owner is not None and owner != player_id and visible_points(owner, state) <= 2:
            return False
    return True


def _steal_targets_for_hex(state: GameState, player_id: int, hex_id: int) -> tuple[int, ...]:
    targets: set[int] = set()
    occupied = state.occupied_nodes()
    for node_id in state.board.hexes[hex_id].node_ids:
        owner = occupied.get(node_id)
        if owner is None or owner == player_id:
            continue
        if state.config.friendly_robber and visible_points(owner, state) <= 2:
            continue
        if state.player(owner).total_resources() > 0:
            targets.add(owner)
    return tuple(sorted(targets))


def _maybe_win(state: GameState, player_id: int) -> GameState:
    if state.phase == Phase.GAME_OVER:
        return state
    if total_points(player_id, state) >= state.config.target_vp:
        return replace(state, phase=Phase.GAME_OVER, winner=player_id)
    return state
