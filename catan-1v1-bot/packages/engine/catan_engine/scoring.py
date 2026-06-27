from __future__ import annotations

from catan_engine.state import GameState


def visible_points(player_id: int, state: GameState) -> int:
    player = state.player(player_id)
    points = len(player.settlements) + 2 * len(player.cities)
    if state.longest_road_owner == player_id:
        points += 2
    if state.largest_army_owner == player_id:
        points += 2
    return points


def total_points(player_id: int, state: GameState) -> int:
    return visible_points(player_id, state) + state.player(player_id).hidden_vp_cards()


def road_length(player_id: int, state: GameState) -> int:
    player = state.player(player_id)
    if not player.roads:
        return 0

    road_graph: dict[int, list[tuple[int, int]]] = {}
    for edge_id in player.roads:
        a, b = state.board.edges[edge_id].node_ids
        road_graph.setdefault(a, []).append((b, edge_id))
        road_graph.setdefault(b, []).append((a, edge_id))

    blocked_nodes = {
        node_id
        for opponent in state.players
        if opponent.id != player_id
        for node_id in (opponent.settlements | opponent.cities)
    }

    best = 0

    def dfs(node_id: int, used_edges: frozenset[int], start_node: int) -> int:
        longest = 0
        if node_id in blocked_nodes and node_id != start_node:
            return 0
        for next_node, edge_id in road_graph.get(node_id, []):
            if edge_id in used_edges:
                continue
            length = 1 + dfs(next_node, used_edges | {edge_id}, start_node)
            longest = max(longest, length)
        return longest

    for node_id in road_graph:
        best = max(best, dfs(node_id, frozenset(), node_id))
    return best


def update_awards(state: GameState) -> GameState:
    state = _update_longest_road(state)
    return _update_largest_army(state)


def _update_longest_road(state: GameState) -> GameState:
    lengths = {player.id: road_length(player.id, state) for player in state.players}
    current_owner = state.longest_road_owner
    current_length = lengths.get(current_owner, 0) if current_owner is not None else 0
    best_player, best_length = max(lengths.items(), key=lambda item: item[1])

    if best_length < 5:
        owner = None
    elif current_owner is None:
        tied = sum(1 for length in lengths.values() if length == best_length)
        owner = best_player if tied == 1 else None
    elif best_player != current_owner and best_length > current_length:
        tied = sum(1 for length in lengths.values() if length == best_length)
        owner = best_player if tied == 1 else current_owner
    else:
        owner = current_owner

    if owner == state.longest_road_owner:
        return state
    from dataclasses import replace

    return replace(state, longest_road_owner=owner)


def _update_largest_army(state: GameState) -> GameState:
    counts = {player.id: player.played_knights for player in state.players}
    current_owner = state.largest_army_owner
    current_count = counts.get(current_owner, 0) if current_owner is not None else 0
    best_player, best_count = max(counts.items(), key=lambda item: item[1])

    if best_count < 3:
        owner = None
    elif current_owner is None:
        tied = sum(1 for count in counts.values() if count == best_count)
        owner = best_player if tied == 1 else None
    elif best_player != current_owner and best_count > current_count:
        tied = sum(1 for count in counts.values() if count == best_count)
        owner = best_player if tied == 1 else current_owner
    else:
        owner = current_owner

    if owner == state.largest_army_owner:
        return state
    from dataclasses import replace

    return replace(state, largest_army_owner=owner)
