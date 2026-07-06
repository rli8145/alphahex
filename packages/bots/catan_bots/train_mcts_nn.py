from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catan_bots.mcts_bot import MCTSBot
from catan_bots.value_network import (
    DEFAULT_VALUE_NETWORK_PATH,
    FEATURE_NAMES,
    TrainingExample,
    ValueNetwork,
    action_policy_label,
    checkpoint_serving_ready,
    extract_state_features,
    load_value_network,
    save_value_network,
)
from catan_engine.actions import IllegalActionError, Phase
from catan_engine.rules import apply_action, get_legal_actions, is_legal_action
from catan_engine.scoring import total_vp
from catan_engine.state import initialize_game

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = REPO_ROOT / "data" / "training"
DEFAULT_DATASET_PATH = TRAINING_DIR / "selfplay.jsonl"
DEFAULT_LEADERBOARD_PATH = TRAINING_DIR / "leaderboard.json"
DEFAULT_HISTORY_DIR = TRAINING_DIR / "checkpoints"
DEFAULT_LOG_PATH = TRAINING_DIR / "logs" / "train.out.log"
DEFAULT_METRICS_CSV_PATH = TRAINING_DIR / "logs" / "train_metrics.csv"
BOARD_RULE_VERSION = "random_start_balanced_board_random_ports_friendly_robber_exact_policy_tactical_v2"
_LOG_FILE_PATH: Path | None = None

TRAINABLE_PHASES = {Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.MOVE_ROBBER, Phase.MAIN}
PROFILE_DEFAULTS = {
    "quick": {
        "games": 1,
        "hidden_size": 32,
        "epochs": 1,
        "learning_rate": 0.02,
        "l2": 0.0001,
        "iterations": 1,
        "rollout_depth": 1,
        "branch_limit": 3,
        "max_turns": 120,
        "eval_games": 0,
        "buffer_samples": 0,
        "history_opponent_rate": 0.0,
        "history_opponents": 0,
    },
    "offline": {
        "games": 12,
        "hidden_size": 64,
        "epochs": 4,
        "learning_rate": 0.015,
        "l2": 0.0001,
        "iterations": 12,
        "rollout_depth": 5,
        "branch_limit": 8,
        "max_turns": 260,
        "eval_games": 20,
        "buffer_samples": 3000,
        "history_opponent_rate": 0.25,
        "history_opponents": 6,
    },
}


@dataclass
class RecordedPosition:
    features: list[float]
    player_id: int
    policy_target: str | None


@dataclass
class GameExamples:
    examples: list[TrainingExample]
    winner: int | None
    final_score: list[int]
    turn_count: int
    illegal_actions: int
    crashes: int


def _log_progress(enabled: bool, **payload: Any) -> None:
    if not enabled:
        return
    payload = {"time": int(time.time()), **payload}
    _emit_json(payload)


def _configure_log_file(path: Path | None) -> None:
    global _LOG_FILE_PATH
    _LOG_FILE_PATH = path
    if _LOG_FILE_PATH is not None:
        _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _emit_json(payload: dict[str, Any]) -> None:
    payload = {"time": int(time.time()), **payload}
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    if _LOG_FILE_PATH is not None:
        with _LOG_FILE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def train(
    *,
    games: int,
    seed: int,
    output: Path,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    resume: bool,
    dataset_path: Path | None,
    buffer_samples: int,
    eval_games: int,
    history_dir: Path,
    leaderboard_path: Path,
    history_opponent_rate: float,
    history_opponents: int,
    accept_vp_margin: float,
    workers: int = 1,
    metrics_csv: Path | None = None,
    dataset_max_games: int = 0,
    cycle: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    incumbent = load_value_network(output) if resume else None
    incumbent_serving_ready = checkpoint_serving_ready(output) if resume else False
    base_network = incumbent or ValueNetwork.create(len(FEATURE_NAMES), hidden_size, rng)
    history_networks = _load_history_networks(history_dir, limit=history_opponents)

    latest_examples: list[TrainingExample] = []
    illegal_actions = 0
    crashes = 0
    total_turns = 0
    wins = {0: 0, 1: 0, "draw": 0}

    _log_progress(progress, cycle=cycle, stage="selfplay", status="started", games=games, workers=workers)
    jobs = [
        (
            game_index,
            seed + game_index,
            base_network,
            iterations,
            rollout_depth,
            branch_limit,
            max_turns,
            history_networks,
            history_opponent_rate,
        )
        for game_index in range(games)
    ]
    game_results: list[tuple[int, GameExamples]] = []
    if workers > 1 and games > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            job_outputs = executor.map(_selfplay_job, jobs)
            for game_index, result in job_outputs:
                game_results.append((game_index, result))
                _log_selfplay_game(progress, cycle, game_index, games, result)
    else:
        for job in jobs:
            game_index, result = _selfplay_job(job)
            game_results.append((game_index, result))
            _log_selfplay_game(progress, cycle, game_index, games, result)

    game_results.sort(key=lambda item: item[0])
    for game_index, result in game_results:
        latest_examples.extend(result.examples)
        illegal_actions += result.illegal_actions
        crashes += result.crashes
        total_turns += result.turn_count
        if result.winner in (0, 1):
            wins[result.winner] += 1
        else:
            wins["draw"] += 1
        if dataset_path is not None:
            _append_game_dataset(dataset_path, seed + game_index, result)
    if dataset_path is not None:
        _compact_dataset(dataset_path, dataset_max_games)

    buffer_examples = _load_replay_examples(dataset_path, rng, buffer_samples) if dataset_path is not None else []
    training_examples = latest_examples + buffer_examples

    candidate = ValueNetwork.from_dict(base_network.to_dict())
    _log_progress(
        progress,
        cycle=cycle,
        stage="sgd",
        status="started",
        latest_examples=len(latest_examples),
        buffer_examples=len(buffer_examples),
        total_examples=len(training_examples),
        epochs=epochs,
        learning_rate=learning_rate,
    )
    training_stats = candidate.train(
        training_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        rng=rng,
    )
    _log_progress(progress, cycle=cycle, stage="sgd", status="finished", **training_stats)

    _log_progress(progress, cycle=cycle, stage="evaluation", status="started", games=eval_games, workers=workers)
    evaluation = evaluate_candidate(
        candidate=candidate,
        current=incumbent,
        seed=seed + 500_000,
        games=eval_games,
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        max_turns=max_turns,
        workers=workers,
    )
    _log_progress(progress, cycle=cycle, stage="evaluation", status="finished", evaluation=evaluation)
    if incumbent is None:
        baseline_evaluation = evaluation
    else:
        _log_progress(
            progress,
            cycle=cycle,
            stage="baseline_evaluation",
            status="started",
            games=eval_games,
            workers=workers,
        )
        baseline_evaluation = evaluate_candidate(
            candidate=candidate,
            current=None,
            seed=seed + 625_000,
            games=eval_games,
            iterations=iterations,
            rollout_depth=rollout_depth,
            branch_limit=branch_limit,
            max_turns=max_turns,
            workers=workers,
        )
        _log_progress(
            progress,
            cycle=cycle,
            stage="baseline_evaluation",
            status="finished",
            evaluation=baseline_evaluation,
        )
    _log_progress(
        progress,
        cycle=cycle,
        stage="history_evaluation",
        status="started",
        opponents=len(history_networks[:3]),
    )
    history_evaluations = _evaluate_history(
        candidate=candidate,
        history_networks=history_networks,
        seed=seed + 750_000,
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        max_turns=max_turns,
        workers=workers,
    )
    _log_progress(
        progress,
        cycle=cycle,
        stage="history_evaluation",
        status="finished",
        evaluations=history_evaluations,
    )
    candidate_accepted_vs_incumbent = _should_accept_candidate(incumbent, evaluation, accept_vp_margin)
    baseline_gate_passed = _evaluation_passes(baseline_evaluation, accept_vp_margin)
    accepted = candidate_accepted_vs_incumbent and baseline_gate_passed
    network_to_save = candidate if accepted else base_network
    saved_network = "candidate" if accepted else "incumbent" if incumbent is not None else "bootstrap"
    saved_serving_ready = accepted or (incumbent is not None and incumbent_serving_ready)

    metadata = {
        "seed": seed,
        "games": games,
        "examples": len(training_examples),
        "latest_examples": len(latest_examples),
        "buffer_examples": len(buffer_examples),
        "hidden_size": network_to_save.hidden_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "iterations": iterations,
        "rollout_depth": rollout_depth,
        "branch_limit": branch_limit,
        "max_turns": max_turns,
        "workers": workers,
        "randomized_boards": True,
        "board_rule_version": BOARD_RULE_VERSION,
        "policy_head": True,
        "replay_dataset": str(dataset_path) if dataset_path is not None else None,
        "history_opponents": len(history_networks),
        "wins_by_player": {"0": wins[0], "1": wins[1], "draw": wins["draw"]},
        "average_turns": round(total_turns / games, 2) if games else 0.0,
        "illegal_actions": illegal_actions,
        "crashes": crashes,
        "accepted": accepted,
        "candidate_accepted_vs_incumbent": candidate_accepted_vs_incumbent,
        "baseline_gate_passed": baseline_gate_passed,
        "serving_ready": saved_serving_ready,
        "saved_network": saved_network,
        "evaluation": evaluation,
        "baseline_evaluation": baseline_evaluation,
        "history_evaluations": history_evaluations,
        **training_stats,
    }
    checkpoint = save_value_network(network_to_save, output, metadata=metadata)
    history_checkpoint = _save_history_checkpoint(candidate, history_dir, seed=seed, metadata=metadata) if accepted else None
    _update_leaderboard(
        leaderboard_path,
        {
            "time": int(time.time()),
            "checkpoint": str(history_checkpoint or checkpoint),
            "accepted": accepted,
            "score": _leaderboard_score(evaluation),
            "evaluation": evaluation,
            "baseline_evaluation": baseline_evaluation,
            "examples": len(training_examples),
            "loss_after": metadata["loss_after"],
            "policy_loss_after": metadata.get("policy_loss_after", 0.0),
        },
    )
    if metrics_csv is not None:
        _append_metrics_csv(
            metrics_csv,
            {
                "time": int(time.time()),
                "cycle": cycle if cycle is not None else 0,
                "seed": seed,
                "games": games,
                "latest_examples": len(latest_examples),
                "buffer_examples": len(buffer_examples),
                "learning_rate": learning_rate,
                "loss_before": metadata["loss_before"],
                "loss_after": metadata["loss_after"],
                "value_loss_after": metadata.get("value_loss_after", 0.0),
                "policy_loss_after": metadata.get("policy_loss_after", 0.0),
                "eval_games": evaluation["games"],
                "candidate_wins": evaluation["candidate_wins"],
                "current_wins": evaluation["current_wins"],
                "draws": evaluation["draws"],
                "average_vp_margin": evaluation["average_vp_margin"],
                "eval_early_stopped": evaluation.get("early_stopped", False),
                "accepted": accepted,
                "average_turns": metadata["average_turns"],
                "illegal_actions": illegal_actions,
                "crashes": crashes,
            },
        )
    _log_progress(
        progress,
        cycle=cycle,
        stage="checkpoint",
        status="saved",
        accepted=accepted,
        baseline_gate_passed=baseline_gate_passed,
        serving_ready=saved_serving_ready,
        checkpoint=str(checkpoint),
        history_checkpoint=str(history_checkpoint) if history_checkpoint else None,
        wins_by_player=metadata["wins_by_player"],
    )
    return {"checkpoint": str(checkpoint), "history_checkpoint": str(history_checkpoint) if history_checkpoint else None, **metadata}


def train_continuously(
    *,
    games: int,
    seed: int,
    output: Path,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    resume: bool,
    sleep_seconds: float,
    dataset_path: Path | None,
    buffer_samples: int,
    eval_games: int,
    history_dir: Path,
    leaderboard_path: Path,
    history_opponent_rate: float,
    history_opponents: int,
    accept_vp_margin: float,
    workers: int = 1,
    lr_decay: float = 1.0,
    metrics_csv: Path | None = None,
    dataset_max_games: int = 0,
) -> None:
    cycle = 0
    examples_total = 0
    try:
        while True:
            cycle += 1
            # Decay the learning rate across cycles so late training makes
            # smaller, more stable updates.
            cycle_learning_rate = learning_rate * (lr_decay ** (cycle - 1))
            _emit_json(
                {
                    "cycle": cycle,
                    "status": "started",
                    "profile_games": games,
                    "iterations": iterations,
                    "rollout_depth": rollout_depth,
                    "branch_limit": branch_limit,
                    "eval_games": eval_games,
                    "workers": workers,
                    "learning_rate": round(cycle_learning_rate, 6),
                }
            )
            summary = train(
                games=games,
                seed=seed + cycle * 1_000_000,
                output=output,
                hidden_size=hidden_size,
                epochs=epochs,
                learning_rate=cycle_learning_rate,
                l2=l2,
                iterations=iterations,
                rollout_depth=rollout_depth,
                branch_limit=branch_limit,
                max_turns=max_turns,
                resume=resume or cycle > 1,
                dataset_path=dataset_path,
                buffer_samples=buffer_samples,
                eval_games=eval_games,
                history_dir=history_dir,
                leaderboard_path=leaderboard_path,
                history_opponent_rate=history_opponent_rate,
                history_opponents=history_opponents,
                accept_vp_margin=accept_vp_margin,
                workers=workers,
                metrics_csv=metrics_csv,
                dataset_max_games=dataset_max_games,
                cycle=cycle,
                progress=True,
            )
            examples_total += int(summary["latest_examples"])
            _emit_json(
                {
                    "cycle": cycle,
                    "accepted": summary["accepted"],
                    "checkpoint": summary["checkpoint"],
                    "history_checkpoint": summary["history_checkpoint"],
                    "examples_total": examples_total,
                    "latest_examples": summary["latest_examples"],
                    "buffer_examples": summary["buffer_examples"],
                    "loss_before": summary["loss_before"],
                    "loss_after": summary["loss_after"],
                    "policy_loss_after": summary.get("policy_loss_after"),
                    "evaluation": summary["evaluation"],
                    "baseline_evaluation": summary["baseline_evaluation"],
                    "baseline_gate_passed": summary["baseline_gate_passed"],
                    "wins_by_player": summary["wins_by_player"],
                    "illegal_actions": summary["illegal_actions"],
                    "crashes": summary["crashes"],
                }
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        _emit_json({"stopped": True, "cycles": cycle, "examples_total": examples_total})


def collect_game_examples(
    *,
    seed: int,
    network: ValueNetwork,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    opponent_pool: list[ValueNetwork] | None = None,
    history_opponent_rate: float = 0.0,
) -> GameExamples:
    rng = random.Random(seed)
    state = initialize_game(seed=seed)
    networks: list[ValueNetwork] = [network, network]
    if opponent_pool and rng.random() < history_opponent_rate:
        history_player = rng.randrange(2)
        networks[history_player] = rng.choice(opponent_pool)
    bots = [
        MCTSBot(iterations=iterations, rollout_depth=rollout_depth, branch_limit=branch_limit, value_network=networks[0]),
        MCTSBot(iterations=iterations, rollout_depth=rollout_depth, branch_limit=branch_limit, value_network=networks[1]),
    ]
    positions: list[RecordedPosition] = []
    illegal_actions = 0
    crashes = 0
    max_actions = max_turns * 200
    action_count = 0

    while state.phase != Phase.GAME_OVER and state.turn_number <= max_turns and action_count <= max_actions:
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            break
        player_id = state.current_player
        record_features = (
            extract_state_features(state, player_id)
            if state.phase in TRAINABLE_PHASES and networks[player_id] is network
            else None
        )
        try:
            observation = {"player_id": player_id, "phase": state.phase.name, "_state": state}
            action = bots[player_id].choose_action(observation, legal_actions, rng)
            if not is_legal_action(state, action):
                illegal_actions += 1
                action = rng.choice(legal_actions)
            if record_features is not None:
                positions.append(RecordedPosition(record_features, player_id, action_policy_label(action)))
            state = apply_action(state, action, rng)
        except Exception as exc:
            crashes += 1
            fallback = rng.choice(legal_actions)
            if record_features is not None:
                positions.append(RecordedPosition(record_features, player_id, action_policy_label(fallback)))
            try:
                state = apply_action(state, fallback, rng)
            except IllegalActionError:
                raise RuntimeError("fallback legal action failed") from exc
        action_count += 1

    final_score = [total_vp(state, 0), total_vp(state, 1)]
    winner = state.winner
    if state.phase != Phase.GAME_OVER:
        if final_score[0] > final_score[1]:
            winner = 0
        elif final_score[1] > final_score[0]:
            winner = 1
        else:
            winner = None

    examples: list[TrainingExample] = [
        (position.features, _target_for_position(winner, final_score, position.player_id), position.policy_target)
        for position in positions
    ]
    return GameExamples(
        examples=examples,
        winner=winner,
        final_score=final_score,
        turn_count=state.turn_number,
        illegal_actions=illegal_actions,
        crashes=crashes,
    )


def _selfplay_job(args: tuple) -> tuple[int, GameExamples]:
    (
        game_index,
        game_seed,
        network,
        iterations,
        rollout_depth,
        branch_limit,
        max_turns,
        opponent_pool,
        history_opponent_rate,
    ) = args
    result = collect_game_examples(
        seed=game_seed,
        network=network,
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        max_turns=max_turns,
        opponent_pool=opponent_pool,
        history_opponent_rate=history_opponent_rate,
    )
    return game_index, result


def _log_selfplay_game(progress: bool, cycle: int | None, game_index: int, games: int, result: GameExamples) -> None:
    _log_progress(
        progress,
        cycle=cycle,
        stage="selfplay",
        status="finished",
        game=game_index + 1,
        games=games,
        examples=len(result.examples),
        winner=result.winner,
        final_score=result.final_score,
        turn_count=result.turn_count,
        illegal_actions=result.illegal_actions,
        crashes=result.crashes,
    )


def _evaluation_job(args: tuple) -> tuple[int, dict[str, Any]]:
    (game_index, game_seed, candidate, current, candidate_player, iterations, rollout_depth, branch_limit, max_turns) = args
    result = _play_evaluation_game(
        seed=game_seed,
        candidate=candidate,
        current=current,
        candidate_player=candidate_player,
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        max_turns=max_turns,
    )
    return game_index, result


def evaluate_candidate(
    *,
    candidate: ValueNetwork,
    current: ValueNetwork | None,
    seed: int,
    games: int,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    workers: int = 1,
) -> dict[str, Any]:
    if games <= 0:
        return {
            "games": 0,
            "candidate_wins": 0,
            "current_wins": 0,
            "draws": 0,
            "average_vp_margin": 0.0,
            "illegal_actions": 0,
            "crashes": 0,
            "average_turns": 0.0,
            "early_stopped": False,
        }
    candidate_wins = 0
    current_wins = 0
    draws = 0
    illegal_actions = 0
    crashes = 0
    total_turns = 0
    total_margin = 0
    played = 0
    early_stopped = False
    chunk_size = max(1, workers)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 and games > 1 else None
    try:
        while played < games:
            chunk = range(played, min(games, played + chunk_size))
            jobs = [
                (index, seed + index, candidate, current, index % 2, iterations, rollout_depth, branch_limit, max_turns)
                for index in chunk
            ]
            outcomes = list(executor.map(_evaluation_job, jobs)) if executor else [_evaluation_job(job) for job in jobs]
            for game_index, result in outcomes:
                candidate_player = game_index % 2
                winner = result["winner"]
                if winner == candidate_player:
                    candidate_wins += 1
                elif winner is None:
                    draws += 1
                else:
                    current_wins += 1
                score = result["final_score"]
                total_margin += score[candidate_player] - score[1 - candidate_player]
                illegal_actions += result["illegal_actions"]
                crashes += result["crashes"]
                total_turns += result["turn_count"]
            played += len(jobs)
            # Stop early once the remaining games can no longer change which
            # side has more wins.
            if played < games and abs(candidate_wins - current_wins) > games - played:
                early_stopped = True
                break
    finally:
        if executor is not None:
            executor.shutdown()
    return {
        "games": played,
        "candidate_wins": candidate_wins,
        "current_wins": current_wins,
        "draws": draws,
        "average_vp_margin": round(total_margin / played, 3),
        "illegal_actions": illegal_actions,
        "crashes": crashes,
        "average_turns": round(total_turns / played, 2),
        "early_stopped": early_stopped,
    }


def build_eval_report(
    *,
    candidate_path: Path,
    opponent_checkpoint: Path | None,
    history_dir: Path,
    seed: int,
    games: int,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    workers: int = 1,
) -> dict[str, Any]:
    candidate = load_value_network(candidate_path)
    if candidate is None:
        raise SystemExit(f"could not load candidate checkpoint: {candidate_path}")

    opponent_path = opponent_checkpoint or _select_report_opponent(candidate_path, history_dir)
    opponent = load_value_network(opponent_path) if opponent_path is not None else None
    if opponent_path is not None and opponent is None:
        raise SystemExit(f"could not load opponent checkpoint: {opponent_path}")

    result = evaluate_candidate(
        candidate=candidate,
        current=opponent,
        seed=seed,
        games=games,
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        max_turns=max_turns,
        workers=workers,
    )
    return {
        "candidate": str(candidate_path),
        "opponent": str(opponent_path) if opponent_path is not None else "heuristic_baseline",
        "board_rule_version": BOARD_RULE_VERSION,
        "settings": {
            "games": games,
            "iterations": iterations,
            "rollout_depth": rollout_depth,
            "branch_limit": branch_limit,
            "max_turns": max_turns,
            "seed": seed,
        },
        "result": result,
    }


def run_smoke(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    network = ValueNetwork.create(len(FEATURE_NAMES), 16, rng)
    for offset in range(5):
        initialize_game(seed=seed + offset).board.validate()
    result = collect_game_examples(
        seed=seed,
        network=network,
        iterations=1,
        rollout_depth=1,
        branch_limit=3,
        max_turns=80,
    )
    return {
        "boards_validated": 5,
        "examples": len(result.examples),
        "winner": result.winner,
        "final_score": result.final_score,
        "illegal_actions": result.illegal_actions,
        "crashes": result.crashes,
    }


def _play_evaluation_game(
    *,
    seed: int,
    candidate: ValueNetwork,
    current: ValueNetwork | None,
    candidate_player: int,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    state = initialize_game(seed=seed)
    bots = []
    for player_id in range(2):
        network = candidate if player_id == candidate_player else current
        bots.append(
            MCTSBot(
                iterations=iterations,
                rollout_depth=rollout_depth,
                branch_limit=branch_limit,
                value_network=network,
                use_value_network=network is not None,
            )
        )
    illegal_actions = 0
    crashes = 0
    max_actions = max_turns * 200
    action_count = 0
    while state.phase != Phase.GAME_OVER and state.turn_number <= max_turns and action_count <= max_actions:
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            break
        player_id = state.current_player
        try:
            action = bots[player_id].choose_action({"player_id": player_id, "_state": state}, legal_actions, rng)
            if not is_legal_action(state, action):
                illegal_actions += 1
                action = rng.choice(legal_actions)
            state = apply_action(state, action, rng)
        except Exception:
            crashes += 1
            state = apply_action(state, rng.choice(legal_actions), rng)
        action_count += 1
    final_score = [total_vp(state, 0), total_vp(state, 1)]
    winner = state.winner
    if state.phase != Phase.GAME_OVER:
        if final_score[0] > final_score[1]:
            winner = 0
        elif final_score[1] > final_score[0]:
            winner = 1
        else:
            winner = None
    return {
        "winner": winner,
        "final_score": final_score,
        "turn_count": state.turn_number,
        "illegal_actions": illegal_actions,
        "crashes": crashes,
    }


def _target_for_position(winner: int | None, final_score: list[int], player_id: int) -> float:
    opponent_id = 1 - player_id
    if winner == player_id:
        return 1.0
    if winner == opponent_id:
        return 0.0
    margin = final_score[player_id] - final_score[opponent_id]
    return max(0.05, min(0.95, 0.5 + margin / 30.0))


def _should_accept_candidate(incumbent: ValueNetwork | None, evaluation: dict[str, Any], accept_vp_margin: float) -> bool:
    if incumbent is None:
        return True
    return _evaluation_passes(evaluation, accept_vp_margin)


def _evaluation_passes(evaluation: dict[str, Any], accept_vp_margin: float) -> bool:
    if evaluation["games"] == 0:
        return False
    if evaluation["crashes"] > 0:
        return False
    if evaluation["candidate_wins"] > evaluation["current_wins"]:
        return True
    if evaluation["candidate_wins"] == evaluation["current_wins"]:
        return float(evaluation["average_vp_margin"]) >= accept_vp_margin
    return False


def _append_game_dataset(path: Path, seed: int, result: GameExamples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "board_rule_version": BOARD_RULE_VERSION,
        "seed": seed,
        "winner": result.winner,
        "final_score": result.final_score,
        "turn_count": result.turn_count,
        "illegal_actions": result.illegal_actions,
        "crashes": result.crashes,
        "examples": [_example_to_json(example) for example in result.examples],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _example_to_json(example: TrainingExample) -> dict[str, Any]:
    features, target, policy_target = _unpack_example(example)
    return {"features": features, "target": target, "policy_target": policy_target}


def _load_replay_examples(path: Path, rng: random.Random, limit: int) -> list[TrainingExample]:
    if limit <= 0 or not path.exists():
        return []
    examples: list[TrainingExample] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("board_rule_version") != BOARD_RULE_VERSION:
                    continue
                for item in payload.get("examples", []):
                    features = [float(value) for value in item.get("features", [])]
                    if len(features) != len(FEATURE_NAMES):
                        continue
                    examples.append((features, float(item.get("target", 0.5)), item.get("policy_target")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    if len(examples) <= limit:
        return examples
    return rng.sample(examples, limit)


def _unpack_example(example: TrainingExample) -> tuple[list[float], float, str | None]:
    if len(example) == 2:
        features, target = example
        return features, float(target), None
    features, target, policy_target = example
    return features, float(target), policy_target


def _select_report_opponent(candidate_path: Path, history_dir: Path) -> Path | None:
    current_rule_paths = _history_checkpoint_paths(history_dir, matching_rules=True)
    fallback_paths = _history_checkpoint_paths(history_dir, matching_rules=False)
    candidate_resolved = _safe_resolve(candidate_path)
    for path in current_rule_paths or fallback_paths:
        if _safe_resolve(path) != candidate_resolved:
            return path
    return None


def _history_checkpoint_paths(history_dir: Path, *, matching_rules: bool) -> list[Path]:
    if not history_dir.exists():
        return []
    paths = sorted(history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matching_rules:
        return paths
    return [path for path in paths if _checkpoint_matches_board_rules(path)]


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _load_history_networks(history_dir: Path, *, limit: int) -> list[ValueNetwork]:
    if limit <= 0 or not history_dir.exists():
        return []
    networks: list[ValueNetwork] = []
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        if not _checkpoint_matches_board_rules(path):
            continue
        network = load_value_network(path)
        if network is not None:
            networks.append(network)
        if len(networks) >= limit:
            break
    return networks


def _checkpoint_matches_board_rules(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    training = data.get("training", {})
    return isinstance(training, dict) and training.get("board_rule_version") == BOARD_RULE_VERSION


def _save_history_checkpoint(network: ValueNetwork, history_dir: Path, *, seed: int, metadata: dict[str, Any]) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_dir / f"value_network_{int(time.time())}_{seed}.json"
    save_value_network(network, target, metadata=metadata)
    _trim_history(history_dir, keep=12)
    return target


def _trim_history(history_dir: Path, *, keep: int) -> None:
    checkpoints = sorted(history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in checkpoints[keep:]:
        try:
            path.unlink()
        except OSError:
            pass


def _evaluate_history(
    *,
    candidate: ValueNetwork,
    history_networks: list[ValueNetwork],
    seed: int,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    workers: int = 1,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for index, network in enumerate(history_networks[:3]):
        summary = evaluate_candidate(
            candidate=candidate,
            current=network,
            seed=seed + index * 100,
            games=2,
            iterations=iterations,
            rollout_depth=rollout_depth,
            branch_limit=branch_limit,
            max_turns=max_turns,
            workers=workers,
        )
        summary["history_index"] = index
        evaluations.append(summary)
    return evaluations


METRICS_CSV_FIELDS = [
    "time",
    "cycle",
    "seed",
    "games",
    "latest_examples",
    "buffer_examples",
    "learning_rate",
    "loss_before",
    "loss_after",
    "value_loss_after",
    "policy_loss_after",
    "eval_games",
    "candidate_wins",
    "current_wins",
    "draws",
    "average_vp_margin",
    "eval_early_stopped",
    "accepted",
    "average_turns",
    "illegal_actions",
    "crashes",
]


def _append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRICS_CSV_FIELDS})


def _compact_dataset(path: Path, max_games: int) -> None:
    """Keep only the newest max_games self-play games (one JSONL line each)."""
    if max_games <= 0 or not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError:
        return
    if len(lines) <= max_games:
        return
    temp_target = path.with_name(f".{path.name}.tmp")
    temp_target.write_text("".join(lines[-max_games:]), encoding="utf-8")
    temp_target.replace(path)


def _update_leaderboard(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(data, list):
            data = []
    except (OSError, json.JSONDecodeError):
        data = []
    data.append(entry)
    data = sorted(data, key=lambda item: int(item.get("time", 0)), reverse=True)[:20]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _leaderboard_score(evaluation: dict[str, Any]) -> float:
    games = max(1, int(evaluation.get("games", 0)))
    win_delta = (int(evaluation.get("candidate_wins", 0)) - int(evaluation.get("current_wins", 0))) / games
    return round(win_delta + float(evaluation.get("average_vp_margin", 0.0)) / 15.0, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the MCTS neural value/policy network from self-play.")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="offline")
    parser.add_argument("--games", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALUE_NETWORK_PATH)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rollout-depth", type=int)
    parser.add_argument("--branch-limit", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--buffer-samples", type=int)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--no-dataset", action="store_true", help="do not append or sample the self-play JSONL dataset")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="continuous-training JSON log path")
    parser.add_argument("--history-opponent-rate", type=float)
    parser.add_argument("--history-opponents", type=int)
    parser.add_argument("--accept-vp-margin", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=0, help="parallel self-play/eval processes; 0 = auto (cpu count - 1)")
    parser.add_argument("--lr-decay", type=float, default=0.995, help="learning-rate multiplier applied per continuous cycle")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV_PATH, help="chart-friendly per-cycle CSV path")
    parser.add_argument("--no-metrics-csv", action="store_true", help="do not append per-cycle metrics CSV rows")
    parser.add_argument(
        "--dataset-max-games",
        type=int,
        default=4000,
        help="compact the self-play JSONL to at most this many newest games (0 disables)",
    )
    parser.add_argument("--fresh", action="store_true", help="ignore the existing NN checkpoint and start a new network")
    parser.add_argument("--continuous", action="store_true", help="keep training and checkpointing until the process is stopped")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="pause between continuous training cycles")
    parser.add_argument("--evaluate-candidate", type=Path, help="evaluate this checkpoint against --output/current and exit")
    parser.add_argument("--eval-report", action="store_true", help="evaluate --output against the previous/history checkpoint and exit")
    parser.add_argument("--opponent-checkpoint", type=Path, help="checkpoint to use as the --eval-report opponent")
    parser.add_argument("--smoke", action="store_true", help="run random-board plus MCTS-NN smoke checks and exit")
    args = parser.parse_args(argv)

    if args.smoke:
        print(json.dumps(run_smoke(args.seed), indent=2))
        return 0

    defaults = PROFILE_DEFAULTS[args.profile]
    settings = {
        "games": args.games if args.games is not None else defaults["games"],
        "hidden_size": args.hidden_size if args.hidden_size is not None else defaults["hidden_size"],
        "epochs": args.epochs if args.epochs is not None else defaults["epochs"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else defaults["learning_rate"],
        "l2": args.l2 if args.l2 is not None else defaults["l2"],
        "iterations": args.iterations if args.iterations is not None else defaults["iterations"],
        "rollout_depth": args.rollout_depth if args.rollout_depth is not None else defaults["rollout_depth"],
        "branch_limit": args.branch_limit if args.branch_limit is not None else defaults["branch_limit"],
        "max_turns": args.max_turns if args.max_turns is not None else defaults["max_turns"],
        "eval_games": args.eval_games if args.eval_games is not None else defaults["eval_games"],
        "buffer_samples": args.buffer_samples if args.buffer_samples is not None else defaults["buffer_samples"],
        "history_opponent_rate": (
            args.history_opponent_rate
            if args.history_opponent_rate is not None
            else defaults["history_opponent_rate"]
        ),
        "history_opponents": args.history_opponents if args.history_opponents is not None else defaults["history_opponents"],
    }
    dataset_path = None if args.no_dataset else args.dataset
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    metrics_csv = None if args.no_metrics_csv else args.metrics_csv

    if args.eval_report:
        report_games = args.eval_games if args.eval_games is not None else 50
        report = build_eval_report(
            candidate_path=args.output,
            opponent_checkpoint=args.opponent_checkpoint,
            history_dir=args.history_dir,
            seed=args.seed,
            games=report_games,
            iterations=settings["iterations"],
            rollout_depth=settings["rollout_depth"],
            branch_limit=settings["branch_limit"],
            max_turns=settings["max_turns"],
            workers=workers,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.evaluate_candidate is not None:
        candidate = load_value_network(args.evaluate_candidate)
        if candidate is None:
            raise SystemExit(f"could not load candidate checkpoint: {args.evaluate_candidate}")
        current = load_value_network(args.output)
        summary = evaluate_candidate(
            candidate=candidate,
            current=current,
            seed=args.seed,
            games=settings["eval_games"] or 2,
            iterations=settings["iterations"],
            rollout_depth=settings["rollout_depth"],
            branch_limit=settings["branch_limit"],
            max_turns=settings["max_turns"],
            workers=workers,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.continuous:
        _configure_log_file(args.log_file)
        train_continuously(
            seed=args.seed,
            output=args.output,
            resume=not args.fresh,
            sleep_seconds=args.sleep_seconds,
            dataset_path=dataset_path,
            history_dir=args.history_dir,
            leaderboard_path=args.leaderboard,
            accept_vp_margin=args.accept_vp_margin,
            workers=workers,
            lr_decay=args.lr_decay,
            metrics_csv=metrics_csv,
            dataset_max_games=args.dataset_max_games,
            **settings,
        )
        return 0

    summary = train(
        seed=args.seed,
        output=args.output,
        resume=not args.fresh,
        dataset_path=dataset_path,
        history_dir=args.history_dir,
        leaderboard_path=args.leaderboard,
        accept_vp_margin=args.accept_vp_margin,
        workers=workers,
        metrics_csv=metrics_csv,
        dataset_max_games=args.dataset_max_games,
        **settings,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
