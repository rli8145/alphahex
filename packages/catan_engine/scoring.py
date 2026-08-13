from __future__ import annotations

from catan_engine.dev_cards import DevCard
from catan_engine.state import GameState


def visible_vp(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    score = len(player.settlements) + 2 * len(player.cities)
    if state.longest_road_owner == player_id:
        score += 2
    if state.largest_army_owner == player_id:
        score += 2
    return score


def total_vp(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    return visible_vp(state, player_id) + player.dev_cards[DevCard.VICTORY_POINT] + player.new_dev_cards[DevCard.VICTORY_POINT]


def calculate_longest_road(board, state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    if not player.roads:
        return 0

    graph: dict[int, list[tuple[int, int]]] = {}
    for edge_id in player.roads:
        edge = board.edges[edge_id]
        graph.setdefault(edge.node_a, []).append((edge.node_b, edge_id))
        graph.setdefault(edge.node_b, []).append((edge.node_a, edge_id))

    blocked_nodes = {
        node_id
        for opponent_id, opponent in enumerate(state.players)
        if opponent_id != player_id
        for node_id in opponent.settlements | opponent.cities
    }

    def dfs(node_id: int, used_edges: set[int], start_node: int) -> int:
        if node_id in blocked_nodes and node_id != start_node:
            return 0
        best = 0
        for next_node, edge_id in graph.get(node_id, []):
            if edge_id in used_edges:
                continue
            used_edges.add(edge_id)
            best = max(best, 1 + dfs(next_node, used_edges, start_node))
            used_edges.remove(edge_id)
        return best

    return max((dfs(node_id, set(), node_id) for node_id in graph), default=0)


def update_longest_road(state: GameState) -> None:
    lengths = {player_id: calculate_longest_road(state.board, state, player_id) for player_id in (0, 1)}
    current = state.longest_road_owner
    if current is None:
        best_player, best_length = max(lengths.items(), key=lambda item: item[1])
        if best_length >= 5 and list(lengths.values()).count(best_length) == 1:
            state.longest_road_owner = best_player
        return

    current_length = lengths[current]
    challenger = 1 - current
    if lengths[challenger] >= 5 and lengths[challenger] > current_length:
        state.longest_road_owner = challenger


def update_largest_army(state: GameState) -> None:
    counts = {player_id: state.players[player_id].played_knights for player_id in (0, 1)}
    current = state.largest_army_owner
    if current is None:
        best_player, best_count = max(counts.items(), key=lambda item: item[1])
        if best_count >= 3 and list(counts.values()).count(best_count) == 1:
            state.largest_army_owner = best_player
        return

    challenger = 1 - current
    if counts[challenger] >= 3 and counts[challenger] > counts[current]:
        state.largest_army_owner = challenger


def update_awards(state: GameState) -> None:
    update_longest_road(state)
    update_largest_army(state)
