from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catan_bots.mcts_bot import (
    DEFAULT_WEIGHTS,
    DEFAULT_WEIGHTS_PATH,
    EvaluationWeights,
    MCTSBot,
    load_trained_weights,
    save_trained_weights,
)
from catan_engine.simulator import run_game

POSITIVE_WEIGHTS = {
    "own_vp",
    "own_resources",
    "resource_diversity",
    "production",
    "port",
    "expansion",
    "road_length",
    "own_knights",
    "dev_cards",
    "new_dev_cards",
}
NEGATIVE_WEIGHTS = {"opponent_vp", "opponent_resources", "opponent_knights", "visible_vp_deficit"}


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    vp_delta: float = 0.0
    total_turns: int = 0
    illegal_actions: int = 0
    crashes: int = 0

    def to_dict(self, games: int) -> dict[str, Any]:
        return {
            "games": games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "vp_delta": round(self.vp_delta, 2),
            "average_vp_delta": round(self.vp_delta / games, 3) if games else 0.0,
            "average_turns": round(self.total_turns / games, 2) if games else 0.0,
            "illegal_actions": self.illegal_actions,
            "crashes": self.crashes,
        }


def train(
    *,
    generations: int,
    candidates: int,
    games_per_candidate: int,
    seed: int,
    output: Path,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    mutation_rate: float,
    mutation_scale: float,
    resume: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    incumbent = load_trained_weights(output) if resume and output.exists() else DEFAULT_WEIGHTS
    accepted = 0
    generation_summaries: list[dict[str, Any]] = []

    for generation in range(1, generations + 1):
        best_weights = incumbent
        best_match: MatchResult | None = None
        best_candidate_index = 0

        for candidate_index in range(1, candidates + 1):
            candidate = mutate_weights(incumbent, rng, rate=mutation_rate, scale=mutation_scale)
            match = evaluate_candidate(
                candidate,
                incumbent,
                games=games_per_candidate,
                seed=seed + generation * 10_000 + candidate_index * 100,
                iterations=iterations,
                rollout_depth=rollout_depth,
                branch_limit=branch_limit,
                max_turns=max_turns,
            )
            if best_match is None or _match_score(match) > _match_score(best_match):
                best_weights = candidate
                best_match = match
                best_candidate_index = candidate_index

        accepted_generation = (
            best_match is not None
            and best_match.crashes == 0
            and best_match.illegal_actions == 0
            and _match_score(best_match) > (0, 0.0, 0, 0)
        )
        if accepted_generation:
            incumbent = best_weights
            accepted += 1

        generation_summaries.append(
            {
                "generation": generation,
                "accepted": accepted_generation,
                "candidate": best_candidate_index,
                "match": best_match.to_dict(games_per_candidate) if best_match else {},
            }
        )

    metadata = {
        "seed": seed,
        "generations": generations,
        "accepted_generations": accepted,
        "candidates_per_generation": candidates,
        "games_per_candidate": games_per_candidate,
        "iterations": iterations,
        "rollout_depth": rollout_depth,
        "branch_limit": branch_limit,
        "max_turns": max_turns,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
        "randomized_boards": True,
        "history": generation_summaries,
    }
    checkpoint = save_trained_weights(incumbent, output, metadata=metadata)
    return {
        "checkpoint": str(checkpoint),
        "accepted_generations": accepted,
        "weights": incumbent.to_dict(),
        "history": generation_summaries,
    }


def train_continuously(
    *,
    generations: int,
    candidates: int,
    games_per_candidate: int,
    seed: int,
    output: Path,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
    mutation_rate: float,
    mutation_scale: float,
    resume: bool,
    sleep_seconds: float,
) -> None:
    cycle = 0
    accepted_total = 0
    try:
        while True:
            cycle += 1
            summary = train(
                generations=generations,
                candidates=candidates,
                games_per_candidate=games_per_candidate,
                seed=seed + cycle * 1_000_000,
                output=output,
                iterations=iterations,
                rollout_depth=rollout_depth,
                branch_limit=branch_limit,
                max_turns=max_turns,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
                resume=resume or cycle > 1,
            )
            accepted_total += int(summary["accepted_generations"])
            print(
                json.dumps(
                    {
                        "cycle": cycle,
                        "accepted_generations_total": accepted_total,
                        "checkpoint": summary["checkpoint"],
                        "accepted_generations": summary["accepted_generations"],
                        "history": summary["history"],
                    }
                ),
                flush=True,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"stopped": True, "cycles": cycle, "accepted_generations_total": accepted_total}), flush=True)


def evaluate_candidate(
    candidate: EvaluationWeights,
    incumbent: EvaluationWeights,
    *,
    games: int,
    seed: int,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
    max_turns: int,
) -> MatchResult:
    result = MatchResult()
    candidate_bot = MCTSBot(
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        weights=candidate,
        use_value_network=False,
    )
    incumbent_bot = MCTSBot(
        iterations=iterations,
        rollout_depth=rollout_depth,
        branch_limit=branch_limit,
        weights=incumbent,
        use_value_network=False,
    )

    for game_index in range(games):
        candidate_player = game_index % 2
        bots = (candidate_bot, incumbent_bot) if candidate_player == 0 else (incumbent_bot, candidate_bot)
        game = run_game(
            bots[0],
            bots[1],
            seed=seed + game_index,
            max_turns=max_turns,
            save_replay_file=False,
        )
        opponent_player = 1 - candidate_player
        winner = game["winner"]
        if winner == candidate_player:
            result.wins += 1
        elif winner == opponent_player:
            result.losses += 1
        else:
            result.draws += 1
        result.vp_delta += game["final_score"][candidate_player] - game["final_score"][opponent_player]
        result.total_turns += game["turn_count"]
        result.illegal_actions += game["illegal_action_count"]
        result.crashes += game["crash_count"]
    return result


def mutate_weights(weights: EvaluationWeights, rng: random.Random, *, rate: float, scale: float) -> EvaluationWeights:
    values = weights.to_dict()
    changed = False
    names = list(values)
    for name in names:
        if rng.random() >= rate:
            continue
        values[name] = _mutated_value(name, values[name], rng, scale)
        changed = True

    if not changed:
        name = rng.choice(names)
        values[name] = _mutated_value(name, values[name], rng, scale)

    return EvaluationWeights.from_dict(values)


def _mutated_value(name: str, value: float, rng: random.Random, scale: float) -> float:
    spread = max(abs(value) * scale, 0.05)
    updated = value + rng.gauss(0.0, spread)
    updated = max(-50.0, min(50.0, updated))
    if name in POSITIVE_WEIGHTS:
        updated = max(0.01, updated)
    elif name in NEGATIVE_WEIGHTS:
        updated = min(-0.01, updated)
    return round(updated, 4)


def _match_score(match: MatchResult) -> tuple[int, float, int, int]:
    return (
        match.wins - match.losses,
        match.vp_delta,
        -match.crashes,
        -match.illegal_actions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train MCTS evaluation weights with self-play.")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--games-per-candidate", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--branch-limit", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-scale", type=float, default=0.18)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start from defaults")
    parser.add_argument("--continuous", action="store_true", help="keep training and checkpointing until the process is stopped")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="pause between continuous training cycles")
    args = parser.parse_args(argv)

    if args.continuous:
        train_continuously(
            generations=args.generations,
            candidates=args.candidates,
            games_per_candidate=args.games_per_candidate,
            seed=args.seed,
            output=args.output,
            iterations=args.iterations,
            rollout_depth=args.rollout_depth,
            branch_limit=args.branch_limit,
            max_turns=args.max_turns,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            resume=not args.fresh,
            sleep_seconds=args.sleep_seconds,
        )
        return 0

    summary = train(
        generations=args.generations,
        candidates=args.candidates,
        games_per_candidate=args.games_per_candidate,
        seed=args.seed,
        output=args.output,
        iterations=args.iterations,
        rollout_depth=args.rollout_depth,
        branch_limit=args.branch_limit,
        max_turns=args.max_turns,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        resume=not args.fresh,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
