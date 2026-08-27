#!/usr/bin/env python3
"""Generate a non-valid capture-line layer sensitivity PNG.

This is a standalone, headless version of the relevant cells in
`demos/Othello_GPT_Jacobian_Lens.ipynb`:

- load Neel Nanda's Othello-GPT TransformerLens checkpoint
- sample mid-game Othello prefixes
- choose illegal empty targets with opponent runs that lack a friendly terminator
- train lightweight board probes for each layer
- score non-valid line sensitivity with an invalid-target-vs-legal-logit contrast
- render 10 sampled boards, each with its own 8-layer white-to-green sensitivity rows
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-transformerlens")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.utilities as utils


BOARD_FILES = list("ABCDEFGH")
BOARD_RANKS = [str(i) for i in range(1, 9)]
CENTER_SQUARES = {27, 28, 35, 36}
DIRECTIONS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PROBE_PREFIX_LENGTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
PROBE_NUM_GAMES = 150
PROBE_BATCH_SIZE = 64
PROBE_TRAIN_FRAC = 0.8
PROBE_TRAIN_BATCH_SIZE = 128
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
PROBE_RANDOM_SEED = 0
LEGALITY_RANDOM_SEED = PROBE_RANDOM_SEED + 7
RATIO_EPSILON = 1e-9

TOKEN_TO_SQUARE = {0: None}
SQUARE_TO_TOKEN = {}
for square_index in range(64):
    if square_index in CENTER_SQUARES:
        continue
    token_id = len(SQUARE_TO_TOKEN) + 1
    TOKEN_TO_SQUARE[token_id] = square_index
    SQUARE_TO_TOKEN[square_index] = token_id


def square_to_rc(square_index: int) -> tuple[int, int]:
    return divmod(square_index, 8)


def rc_to_square(row: int, col: int) -> int:
    return row * 8 + col


def square_label(square_index: int) -> str:
    row, col = square_to_rc(square_index)
    return f"{BOARD_FILES[col]}{row + 1}"


def move_token_label(token_id: int) -> str:
    if token_id == 0:
        return "pass"
    square_index = TOKEN_TO_SQUARE[token_id]
    assert square_index is not None
    return square_label(square_index)


class CompactOthelloGame:
    """Tiny explicit Othello engine from the Chapter 7 notebook."""

    def __init__(self) -> None:
        self.board = [0] * 64
        self.board[27] = -1
        self.board[28] = 1
        self.board[35] = 1
        self.board[36] = -1
        self.to_play = 1

    def capture_lines_for(self, square_index: int | None, color: int | None = None) -> list[list[int]]:
        color = self.to_play if color is None else color
        if square_index is None or self.board[square_index] != 0:
            return []

        row0, col0 = square_to_rc(square_index)
        capture_lines = []
        for d_row, d_col in DIRECTIONS_8:
            row, col = row0 + d_row, col0 + d_col
            line = []
            while 0 <= row < 8 and 0 <= col < 8:
                idx = rc_to_square(row, col)
                value = self.board[idx]
                if value == -color:
                    line.append(idx)
                elif value == color:
                    if line:
                        capture_lines.append(line)
                    break
                else:
                    break
                row += d_row
                col += d_col
        return capture_lines

    def flips_for(self, square_index: int | None, color: int | None = None) -> list[int]:
        return [idx for line in self.capture_lines_for(square_index, color=color) for idx in line]

    def legal_squares(self, color: int | None = None) -> list[int]:
        color = self.to_play if color is None else color
        return [square_index for square_index in range(64) if self.flips_for(square_index, color)]

    def legal_moves_with_metadata(self, color: int | None = None) -> list[dict]:
        color = self.to_play if color is None else color
        legal_moves = []
        for square_index in self.legal_squares(color):
            capture_lines = self.capture_lines_for(square_index, color=color)
            legal_moves.append(
                {
                    "square_index": square_index,
                    "token_id": SQUARE_TO_TOKEN[square_index],
                    "label": square_label(square_index),
                    "num_flipped": sum(len(line) for line in capture_lines),
                    "num_capture_lines": len(capture_lines),
                    "capture_lines": capture_lines,
                    "capture_line_labels": [
                        [square_label(captured_square) for captured_square in line]
                        for line in capture_lines
                    ],
                }
            )
        return legal_moves

    def apply_move_token(self, token_id: int) -> None:
        if token_id == 0:
            assert not self.legal_squares(self.to_play)
            assert self.legal_squares(-self.to_play)
            self.to_play *= -1
            return

        square_index = TOKEN_TO_SQUARE[token_id]
        flips = self.flips_for(square_index, self.to_play)
        if not flips:
            raise ValueError(f"Illegal move token {token_id} -> {move_token_label(token_id)}")

        self.board[square_index] = self.to_play
        for idx in flips:
            self.board[idx] = self.to_play
        self.to_play *= -1

    def board_state_targets(self) -> torch.Tensor:
        targets = []
        for value in self.board:
            if value == 0:
                targets.append(0)
            elif value == self.to_play:
                targets.append(1)
            else:
                targets.append(2)
        return torch.tensor(targets, dtype=torch.long)


def generate_random_othello_game(max_len: int, rng: random.Random) -> list[int]:
    game = CompactOthelloGame()
    tokens_out = []
    while len(tokens_out) < max_len:
        legal_squares = game.legal_squares()
        if legal_squares:
            chosen_square = rng.choice(legal_squares)
            token_id = SQUARE_TO_TOKEN[chosen_square]
            game.apply_move_token(token_id)
            tokens_out.append(token_id)
        else:
            opponent_legal = game.legal_squares(-game.to_play)
            if opponent_legal:
                game.apply_move_token(0)
                tokens_out.append(0)
            else:
                break
    return tokens_out


def decode_tokens_into_game(token_ids: tuple[int, ...] | list[int]) -> CompactOthelloGame:
    game = CompactOthelloGame()
    for token_id in token_ids:
        game.apply_move_token(int(token_id))
    return game


def generate_unique_othello_games(
    num_games: int,
    max_len: int,
    rng: random.Random,
) -> list[list[int]]:
    games = []
    seen_games = set()
    while len(games) < num_games:
        game_tokens = tuple(generate_random_othello_game(max_len=max_len, rng=rng))
        if game_tokens in seen_games:
            continue
        seen_games.add(game_tokens)
        games.append(list(game_tokens))
    return games


def split_games_by_whole_game(
    games: list[list[int]],
    train_frac: float = PROBE_TRAIN_FRAC,
    seed: int = PROBE_RANDOM_SEED,
) -> tuple[list[list[int]], list[list[int]]]:
    game_indices = list(range(len(games)))
    split_rng = random.Random(seed)
    split_rng.shuffle(game_indices)
    train_size = int(train_frac * len(game_indices))
    train_indices = sorted(game_indices[:train_size])
    val_indices = sorted(game_indices[train_size:])
    return [games[idx] for idx in train_indices], [games[idx] for idx in val_indices]


def extract_prefix_samples_from_games(
    games: list[list[int]],
    prefix_lengths: list[int] = PROBE_PREFIX_LENGTHS,
) -> tuple[defaultdict[int, list], set[tuple[int, ...]]]:
    samples_by_length = defaultdict(list)
    prefix_keys = set()
    for full_game in games:
        replay = CompactOthelloGame()
        targets_after_move = []
        for token_id in full_game:
            replay.apply_move_token(token_id)
            targets_after_move.append(replay.board_state_targets())
        for prefix_len in prefix_lengths:
            if prefix_len <= len(full_game):
                prefix_tokens = torch.tensor(full_game[:prefix_len], dtype=torch.long)
                board_targets = targets_after_move[prefix_len - 1]
                prefix_key = tuple(prefix_tokens.tolist())
                samples_by_length[prefix_len].append((prefix_tokens, board_targets, prefix_key))
                prefix_keys.add(prefix_key)
    return samples_by_length, prefix_keys


def filter_cross_split_prefix_overlap(
    samples_by_length: defaultdict[int, list],
    forbidden_prefixes: set[tuple[int, ...]],
) -> tuple[defaultdict[int, list], int]:
    filtered_samples = defaultdict(list)
    removed = 0
    for prefix_len, samples in samples_by_length.items():
        for prefix_tokens, board_targets, prefix_key in samples:
            if prefix_key in forbidden_prefixes:
                removed += 1
                continue
            filtered_samples[prefix_len].append((prefix_tokens, board_targets, prefix_key))
    return filtered_samples, removed


def strip_prefix_keys(samples_by_length: defaultdict[int, list]) -> defaultdict[int, list]:
    stripped = defaultdict(list)
    for prefix_len, samples in samples_by_length.items():
        for prefix_tokens, board_targets, _prefix_key in samples:
            stripped[prefix_len].append((prefix_tokens, board_targets))
    return stripped


def normalize_last_dim(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return tensor / (tensor.norm(dim=-1, keepdim=True) + eps)


def choose_legality_analysis_move(game: CompactOthelloGame) -> dict | None:
    candidates = [
        move_info
        for move_info in game.legal_moves_with_metadata()
        if move_info["num_capture_lines"] >= 1 and move_info["num_flipped"] >= 2
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda move_info: (move_info["num_flipped"], move_info["num_capture_lines"], -move_info["token_id"]),
    )


def opponent_runs_without_friendly_terminator(
    game: CompactOthelloGame,
    square_index: int,
    color: int | None = None,
) -> list[dict]:
    color = game.to_play if color is None else color
    if square_index is None or game.board[square_index] != 0:
        return []

    row0, col0 = square_to_rc(square_index)
    runs = []
    for d_row, d_col in DIRECTIONS_8:
        row, col = row0 + d_row, col0 + d_col
        line = []
        while 0 <= row < 8 and 0 <= col < 8:
            idx = rc_to_square(row, col)
            value = game.board[idx]
            if value == -color:
                line.append(idx)
                row += d_row
                col += d_col
                continue
            if line and value != color:
                runs.append(
                    {
                        "line": line,
                        "failure_square": idx,
                        "failure_kind": "empty",
                    }
                )
            break
        else:
            if line:
                runs.append(
                    {
                        "line": line,
                        "failure_square": None,
                        "failure_kind": "edge",
                    }
                )
    return runs


def choose_nonvalid_capture_analysis_target(game: CompactOthelloGame) -> dict | None:
    legal_square_ids = {move_info["square_index"] for move_info in game.legal_moves_with_metadata()}
    candidates = []
    for square_index in sorted(SQUARE_TO_TOKEN):
        if game.board[square_index] != 0 or square_index in legal_square_ids:
            continue
        failed_runs = opponent_runs_without_friendly_terminator(game, square_index)
        if not failed_runs:
            continue
        candidates.append(
            {
                "square_index": square_index,
                "token_id": SQUARE_TO_TOKEN[square_index],
                "label": square_label(square_index),
                "num_flipped": sum(len(run["line"]) for run in failed_runs),
                "num_capture_lines": len(failed_runs),
                "capture_lines": [run["line"] for run in failed_runs],
                "capture_line_labels": [
                    [square_label(captured_square) for captured_square in run["line"]]
                    for run in failed_runs
                ],
                "failed_runs": failed_runs,
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda move_info: (move_info["num_flipped"], move_info["num_capture_lines"], -move_info["token_id"]),
    )


def sample_legality_positions(
    num_positions: int,
    min_prefix_len: int,
    max_prefix_len: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    positions = []
    seen_prefixes = set()
    while len(positions) < num_positions:
        game_tokens = generate_random_othello_game(max_len=max(PROBE_PREFIX_LENGTHS), rng=rng)
        if len(game_tokens) <= min_prefix_len:
            continue
        prefix_len = rng.randint(min_prefix_len, min(max_prefix_len, len(game_tokens) - 1))
        prefix_token_ids = tuple(game_tokens[:prefix_len])
        if prefix_token_ids in seen_prefixes:
            continue
        game = decode_tokens_into_game(prefix_token_ids)
        move_info = choose_legality_analysis_move(game)
        if move_info is None:
            continue
        seen_prefixes.add(prefix_token_ids)
        positions.append(
            {
                "prefix_len": prefix_len,
                "prefix_token_ids": prefix_token_ids,
                "game": game,
                "move_info": move_info,
            }
        )
    return positions


def sample_nonvalid_capture_positions(
    num_positions: int,
    min_prefix_len: int,
    max_prefix_len: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    positions = []
    seen_position_targets = set()
    while len(positions) < num_positions:
        game_tokens = generate_random_othello_game(max_len=max(PROBE_PREFIX_LENGTHS), rng=rng)
        if len(game_tokens) <= min_prefix_len:
            continue
        prefix_len = rng.randint(min_prefix_len, min(max_prefix_len, len(game_tokens) - 1))
        prefix_token_ids = tuple(game_tokens[:prefix_len])
        game = decode_tokens_into_game(prefix_token_ids)
        move_info = choose_nonvalid_capture_analysis_target(game)
        if move_info is None:
            continue
        key = (prefix_token_ids, int(move_info["square_index"]))
        if key in seen_position_targets:
            continue
        seen_position_targets.add(key)
        positions.append(
            {
                "prefix_len": prefix_len,
                "prefix_token_ids": prefix_token_ids,
                "game": game,
                "move_info": move_info,
            }
        )
    return positions


def line_direction_from_squares(start_square: int, next_square: int) -> tuple[int, int]:
    start_row, start_col = square_to_rc(start_square)
    next_row, next_col = square_to_rc(next_square)
    d_row = next_row - start_row
    d_col = next_col - start_col
    d_row = 0 if d_row == 0 else d_row // abs(d_row)
    d_col = 0 if d_col == 0 else d_col // abs(d_col)
    return d_row, d_col


def terminator_square_for_line(move_square_index: int, capture_line: list[int]) -> int | None:
    d_row, d_col = line_direction_from_squares(move_square_index, capture_line[0])
    row, col = square_to_rc(capture_line[-1])
    row += d_row
    col += d_col
    if 0 <= row < 8 and 0 <= col < 8:
        return rc_to_square(row, col)
    return None


def legal_move_token_ids(game: CompactOthelloGame) -> list[int]:
    return sorted(move_info["token_id"] for move_info in game.legal_moves_with_metadata())


def illegal_empty_move_token_ids(game: CompactOthelloGame) -> list[int]:
    legal_square_ids = {move_info["square_index"] for move_info in game.legal_moves_with_metadata()}
    illegal_tokens = []
    for square_index in sorted(SQUARE_TO_TOKEN):
        if game.board[square_index] != 0:
            continue
        if square_index in legal_square_ids:
            continue
        illegal_tokens.append(SQUARE_TO_TOKEN[square_index])
    return illegal_tokens


def rank_within_subset(logits: torch.Tensor, move_id: int, subset_ids: list[int]) -> int:
    subset = logits[subset_ids]
    move_score = logits[move_id]
    return int((subset > move_score).sum().item()) + 1


def compute_legality_metrics_from_logits(
    logits_1d: torch.Tensor,
    move_id: int,
    legal_ids: list[int],
    illegal_ids: list[int],
) -> dict:
    z_m = logits_1d[move_id]
    illegal_mean = logits_1d[illegal_ids].mean()
    other_legal_ids = [token_id for token_id in legal_ids if token_id != move_id]
    other_legal_mean = logits_1d[other_legal_ids].mean() if other_legal_ids else torch.full_like(z_m, float("nan"))
    return {
        "raw_move_logit": z_m,
        "mean_illegal_logit": illegal_mean,
        "mean_other_legal_logit": other_legal_mean,
        "legality_contrast": z_m - illegal_mean,
        "legal_preference_contrast": z_m - other_legal_mean,
        "move_rank_all_tokens": int((logits_1d > z_m).sum().item()) + 1,
        "move_rank_legal_moves": rank_within_subset(logits_1d, move_id, legal_ids),
    }


def compute_invalid_target_metrics_from_logits(
    logits_1d: torch.Tensor,
    target_id: int,
    legal_ids: list[int],
) -> dict:
    z_target = logits_1d[target_id]
    legal_mean = logits_1d[legal_ids].mean()
    return {
        "raw_target_logit": z_target,
        "mean_legal_logit": legal_mean,
        "invalid_vs_legal_contrast": z_target - legal_mean,
        "target_rank_all_tokens": int((logits_1d > z_target).sum().item()) + 1,
        "target_rank_legal_moves": rank_within_subset(logits_1d, target_id, legal_ids),
    }


def chebyshev_distance(square_a: int, square_b: int) -> int:
    row_a, col_a = square_to_rc(square_a)
    row_b, col_b = square_to_rc(square_b)
    return max(abs(row_a - row_b), abs(col_a - col_b))


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def load_model(device: torch.device) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=str(device),
    )
    model = HookedTransformer(cfg)
    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens",
        "synthetic_model.pth",
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


class Chapter7Scorer:
    def __init__(
        self,
        model: HookedTransformer,
        device: torch.device,
        probe_epochs: int,
    ) -> None:
        self.model = model
        self.device = device
        self.d_model = model.cfg.d_model
        self.probe_epochs = probe_epochs
        self.layer_probe_cache: dict[int, dict] = {}
        self.layer_activation_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        probe_rng = random.Random(PROBE_RANDOM_SEED)
        torch.manual_seed(PROBE_RANDOM_SEED)
        games = generate_unique_othello_games(
            num_games=PROBE_NUM_GAMES,
            max_len=max(PROBE_PREFIX_LENGTHS),
            rng=probe_rng,
        )
        train_games, val_games = split_games_by_whole_game(games)
        train_samples_keyed, train_prefix_keys = extract_prefix_samples_from_games(train_games)
        val_samples_keyed, _ = extract_prefix_samples_from_games(val_games)
        val_samples_filtered, _ = filter_cross_split_prefix_overlap(val_samples_keyed, train_prefix_keys)
        self.probe_train_samples_by_length = strip_prefix_keys(train_samples_keyed)
        self.probe_val_samples_by_length = strip_prefix_keys(val_samples_filtered)

    def collect_probe_activations(
        self,
        samples_by_length: defaultdict[int, list],
        hook_name: str,
        batch_size: int = PROBE_BATCH_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activations = []
        targets = []
        prefix_lengths = []
        with torch.no_grad():
            for prefix_len in sorted(samples_by_length):
                samples = samples_by_length[prefix_len]
                if not samples:
                    continue
                token_batch = torch.stack([tokens_i for tokens_i, _ in samples]).to(self.device)
                target_batch = torch.stack([targets_i for _, targets_i in samples])
                for start in range(0, token_batch.shape[0], batch_size):
                    token_slice = token_batch[start : start + batch_size]
                    target_slice = target_batch[start : start + batch_size]
                    _, cache = self.model.run_with_cache(
                        token_slice,
                        names_filter=lambda name: name == hook_name,
                        return_type="logits",
                    )
                    acts = cache[hook_name][:, -1, :].detach().cpu()
                    activations.append(acts)
                    targets.append(target_slice)
                    prefix_lengths.append(torch.full((acts.shape[0],), prefix_len, dtype=torch.long))
        return torch.cat(activations), torch.cat(targets), torch.cat(prefix_lengths)

    def get_probe_data_for_layer(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer not in self.layer_activation_cache:
            hook_name = f"blocks.{layer}.hook_resid_post"
            X_train_layer, Y_train_layer, _ = self.collect_probe_activations(
                self.probe_train_samples_by_length,
                hook_name=hook_name,
            )
            X_val_layer, Y_val_layer, _ = self.collect_probe_activations(
                self.probe_val_samples_by_length,
                hook_name=hook_name,
            )
            self.layer_activation_cache[layer] = (X_train_layer, Y_train_layer, X_val_layer, Y_val_layer)
        return self.layer_activation_cache[layer]

    def train_probe_for_layer(self, layer: int) -> dict:
        if layer in self.layer_probe_cache:
            return self.layer_probe_cache[layer]

        X_train_layer, Y_train_layer, X_val_layer, Y_val_layer = self.get_probe_data_for_layer(layer)
        torch.manual_seed(PROBE_RANDOM_SEED + layer)
        probe = nn.Linear(self.d_model, 64 * 3).to(self.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        permutation = torch.randperm(X_train_layer.shape[0])
        X_train_local = X_train_layer[permutation]
        Y_train_local = Y_train_layer[permutation]

        for _ in range(self.probe_epochs):
            probe.train()
            for start in range(0, X_train_local.shape[0], PROBE_TRAIN_BATCH_SIZE):
                x_batch = X_train_local[start : start + PROBE_TRAIN_BATCH_SIZE].to(self.device)
                y_batch = Y_train_local[start : start + PROBE_TRAIN_BATCH_SIZE].to(self.device)
                logits_batch = probe(x_batch).view(-1, 64, 3)
                loss = F.cross_entropy(logits_batch.view(-1, 3), y_batch.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_layer.to(self.device)).view(-1, 64, 3)
            val_pred = val_logits.argmax(dim=-1).cpu()
        val_acc = float((val_pred == Y_val_layer).float().mean())
        probe_weight = probe.weight.detach().view(64, 3, self.d_model)
        mine_dirs = normalize_last_dim(probe_weight[:, 1, :] - probe_weight[:, 2, :])
        occ_dirs = normalize_last_dim(0.5 * (probe_weight[:, 1, :] + probe_weight[:, 2, :]) - probe_weight[:, 0, :])
        self.layer_probe_cache[layer] = {
            "probe": probe,
            "val_accuracy": val_acc,
            "mine_dirs": mine_dirs.detach(),
            "occ_dirs": occ_dirs.detach(),
        }
        return self.layer_probe_cache[layer]

    def prepare_position_context(self, prefix_token_ids: tuple[int, ...], layer: int) -> dict:
        prefix_tokens = torch.tensor(prefix_token_ids, dtype=torch.long, device=self.device)[None, :]
        source_pos = prefix_tokens.shape[1] - 1
        target_pos = source_pos
        hook_name = f"blocks.{layer}.hook_resid_post"
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                prefix_tokens,
                names_filter=lambda name: name == hook_name,
                return_type="logits",
            )
        source_resid_local = cache[hook_name].detach()

        def logits_fn(delta: torch.Tensor) -> torch.Tensor:
            resid = source_resid_local.clone()
            resid[:, source_pos, :] = resid[:, source_pos, :] + delta
            logits = self.model(resid, start_at_layer=layer + 1, return_type="logits")
            return logits[:, target_pos, :]

        return {"logits_fn": logits_fn}

    def legality_gradient_for_move(
        self,
        prefix_token_ids: tuple[int, ...],
        move_info: dict,
        layer: int,
    ) -> dict:
        game = decode_tokens_into_game(prefix_token_ids)
        legal_ids = legal_move_token_ids(game)
        illegal_ids = illegal_empty_move_token_ids(game)
        context = self.prepare_position_context(prefix_token_ids, layer=layer)
        delta = torch.zeros(self.d_model, device=self.device, requires_grad=True)
        logits = context["logits_fn"](delta)[0]
        metrics = compute_legality_metrics_from_logits(logits, move_info["token_id"], legal_ids, illegal_ids)
        legality_grad = torch.autograd.grad(metrics["legality_contrast"], delta, retain_graph=False)[0].detach()
        metric_values = {
            key: float(value.detach()) if torch.is_tensor(value) else value
            for key, value in metrics.items()
        }
        return {
            "game": game,
            "logits": logits.detach(),
            "legal_ids": legal_ids,
            "illegal_ids": illegal_ids,
            "metrics": metric_values,
            "legality_grad": legality_grad,
        }

    def invalid_target_gradient_for_move(
        self,
        prefix_token_ids: tuple[int, ...],
        move_info: dict,
        layer: int,
    ) -> dict:
        game = decode_tokens_into_game(prefix_token_ids)
        legal_ids = legal_move_token_ids(game)
        context = self.prepare_position_context(prefix_token_ids, layer=layer)
        delta = torch.zeros(self.d_model, device=self.device, requires_grad=True)
        logits = context["logits_fn"](delta)[0]
        metrics = compute_invalid_target_metrics_from_logits(logits, move_info["token_id"], legal_ids)
        invalid_grad = torch.autograd.grad(metrics["invalid_vs_legal_contrast"], delta, retain_graph=False)[0].detach()
        metric_values = {
            key: float(value.detach()) if torch.is_tensor(value) else value
            for key, value in metrics.items()
        }
        return {
            "game": game,
            "logits": logits.detach(),
            "legal_ids": legal_ids,
            "metrics": metric_values,
            "invalid_grad": invalid_grad,
        }

    def per_position_legality_statistics(
        self,
        position: dict,
        layer: int,
        mine_dirs: torch.Tensor,
        occ_dirs: torch.Tensor,
    ) -> dict:
        result = self.legality_gradient_for_move(position["prefix_token_ids"], position["move_info"], layer=layer)
        mine_scores = mine_dirs @ result["legality_grad"]
        occ_scores = occ_dirs @ result["legality_grad"]

        capture_squares = set(square for line in position["move_info"]["capture_lines"] for square in line)
        terminators = {
            terminator_square_for_line(position["move_info"]["square_index"], line)
            for line in position["move_info"]["capture_lines"]
        }
        terminators.discard(None)
        excluded = capture_squares | terminators | {position["move_info"]["square_index"]}
        unrelated_occupied = [
            square_index
            for square_index in sorted(SQUARE_TO_TOKEN)
            if square_index not in excluded and result["game"].board[square_index] != 0
        ]
        unrelated_empty = [
            square_index
            for square_index in sorted(SQUARE_TO_TOKEN)
            if square_index not in excluded and result["game"].board[square_index] == 0
        ]
        nearby_irrelevant = [
            square_index
            for square_index in unrelated_occupied + unrelated_empty
            if chebyshev_distance(square_index, position["move_info"]["square_index"]) <= 2
        ]
        capture_union = sorted(capture_squares | terminators)
        capture_union_values = [float(mine_scores[square_index].abs()) for square_index in capture_union]
        unrelated_occupied_values = [float(mine_scores[square_index].abs()) for square_index in unrelated_occupied]

        return {
            "capture_opponent": [float(mine_scores[square_index].abs()) for square_index in sorted(capture_squares)],
            "capture_terminator": [float(mine_scores[square_index].abs()) for square_index in sorted(terminators)],
            "unrelated_occupied": unrelated_occupied_values,
            "unrelated_empty": [float(occ_scores[square_index].abs()) for square_index in unrelated_empty],
            "nearby_irrelevant": [
                float((mine_scores if result["game"].board[square_index] != 0 else occ_scores)[square_index].abs())
                for square_index in nearby_irrelevant
            ],
            "capture_union_mean": safe_mean(capture_union_values),
            "unrelated_occupied_mean": safe_mean(unrelated_occupied_values),
            "capture_union_ratio": safe_mean(capture_union_values) / (safe_mean(unrelated_occupied_values) + RATIO_EPSILON),
            "capture_squares": capture_squares,
            "terminators": terminators,
        }

    def per_position_nonvalid_statistics(
        self,
        position: dict,
        layer: int,
        mine_dirs: torch.Tensor,
        occ_dirs: torch.Tensor,
    ) -> dict:
        result = self.invalid_target_gradient_for_move(position["prefix_token_ids"], position["move_info"], layer=layer)
        mine_scores = mine_dirs @ result["invalid_grad"]
        occ_scores = occ_dirs @ result["invalid_grad"]

        opponent_run_squares = set(square for line in position["move_info"]["capture_lines"] for square in line)
        failure_squares = {
            run["failure_square"]
            for run in position["move_info"]["failed_runs"]
            if run["failure_square"] is not None
        }
        excluded = opponent_run_squares | failure_squares | {position["move_info"]["square_index"]}

        support_values = [float(mine_scores[square_index].abs()) for square_index in sorted(opponent_run_squares)]
        for square_index in sorted(failure_squares):
            score_source = mine_scores if result["game"].board[square_index] != 0 else occ_scores
            support_values.append(float(score_source[square_index].abs()))

        control_values = []
        for square_index in sorted(SQUARE_TO_TOKEN):
            if square_index in excluded:
                continue
            score_source = mine_scores if result["game"].board[square_index] != 0 else occ_scores
            control_values.append(float(score_source[square_index].abs()))

        unrelated_occupied = [
            square_index
            for square_index in sorted(SQUARE_TO_TOKEN)
            if square_index not in excluded and result["game"].board[square_index] != 0
        ]
        return {
            "opponent_run": [float(mine_scores[square_index].abs()) for square_index in sorted(opponent_run_squares)],
            "failure_square": [
                float((mine_scores if result["game"].board[square_index] != 0 else occ_scores)[square_index].abs())
                for square_index in sorted(failure_squares)
            ],
            "control": control_values,
            "support_mean": safe_mean(support_values),
            "control_mean": safe_mean(control_values),
            "capture_union_mean": safe_mean(support_values),
            "unrelated_occupied_mean": safe_mean(control_values),
            "capture_union_ratio": safe_mean(support_values) / (safe_mean(control_values) + RATIO_EPSILON),
            "opponent_run_squares": opponent_run_squares,
            "failure_squares": failure_squares,
            "capture_squares": opponent_run_squares,
            "terminators": failure_squares,
            "unrelated_occupied_squares": unrelated_occupied,
        }

    def layer_summary(self, positions: list[dict], layers: list[int]) -> tuple[list[dict], dict[int, list[dict]]]:
        layer_rows = []
        stats_by_layer = {}
        for layer in layers:
            print(f"Scoring layer {layer}...", flush=True)
            probe_info = self.train_probe_for_layer(layer)
            layer_position_stats = [
                self.per_position_legality_statistics(
                    position,
                    layer=layer,
                    mine_dirs=probe_info["mine_dirs"],
                    occ_dirs=probe_info["occ_dirs"],
                )
                for position in positions
            ]
            stats_by_layer[layer] = layer_position_stats
            layer_rows.append(
                {
                    "layer": layer,
                    "probe_val_accuracy": probe_info["val_accuracy"],
                    "capture_vs_unrelated_ratio": float(np.mean([stats["capture_union_ratio"] for stats in layer_position_stats])),
                    "capture_minus_unrelated": float(
                        np.mean([
                            stats["capture_union_mean"] - stats["unrelated_occupied_mean"]
                            for stats in layer_position_stats
                        ])
                    ),
                }
            )
        return sorted(layer_rows, key=lambda row: row["layer"]), stats_by_layer

    def nonvalid_layer_summary(self, positions: list[dict], layers: list[int]) -> tuple[list[dict], dict[int, list[dict]]]:
        layer_rows = []
        stats_by_layer = {}
        for layer in layers:
            print(f"Scoring layer {layer}...", flush=True)
            probe_info = self.train_probe_for_layer(layer)
            layer_position_stats = [
                self.per_position_nonvalid_statistics(
                    position,
                    layer=layer,
                    mine_dirs=probe_info["mine_dirs"],
                    occ_dirs=probe_info["occ_dirs"],
                )
                for position in positions
            ]
            stats_by_layer[layer] = layer_position_stats
            layer_rows.append(
                {
                    "layer": layer,
                    "probe_val_accuracy": probe_info["val_accuracy"],
                    "nonvalid_line_vs_control_ratio": float(np.mean([stats["capture_union_ratio"] for stats in layer_position_stats])),
                    "nonvalid_line_minus_control": float(
                        np.mean([
                            stats["capture_union_mean"] - stats["unrelated_occupied_mean"]
                            for stats in layer_position_stats
                        ])
                    ),
                }
            )
        return sorted(layer_rows, key=lambda row: row["layer"]), stats_by_layer


def layer_color(value: float, min_value: float, max_value: float) -> tuple[float, float, float]:
    if max_value <= min_value:
        t = 1.0
    else:
        t = (value - min_value) / (max_value - min_value)
    white = np.array([1.0, 1.0, 1.0])
    green = np.array([0.02, 0.52, 0.22])
    return tuple(white * (1.0 - t) + green * t)


def draw_board(ax, position: dict, index: int) -> None:
    game = position["game"]
    move_info = position["move_info"]
    move_square = move_info["square_index"]
    capture_squares = set(square for line in move_info["capture_lines"] for square in line)
    if "failed_runs" in move_info:
        terminators = set()
        failure_squares = {
            run["failure_square"]
            for run in move_info["failed_runs"]
            if run["failure_square"] is not None
        }
    else:
        terminators = {
            terminator_square_for_line(move_square, line)
            for line in move_info["capture_lines"]
        }
        terminators.discard(None)
        failure_squares = set()

    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    for row in range(8):
        for col in range(8):
            square = rc_to_square(row, col)
            face = "#f4f7ef" if (row + col) % 2 == 0 else "#e8efdf"
            edge = "#c7d1be"
            linewidth = 0.5
            if square in capture_squares:
                face = "#f5c253"
                edge = "#9a6b00"
                linewidth = 1.2
            if square in terminators:
                face = "#6cc07c"
                edge = "#1f6f3a"
                linewidth = 1.2
            if square in failure_squares:
                face = "#f6b0a6"
                edge = "#b42318"
                linewidth = 1.2
            if square == move_square:
                face = "#8fc5ff"
                edge = "#1a5fb4"
                linewidth = 1.6
            rect = plt.Rectangle((col, row), 1, 1, facecolor=face, edgecolor=edge, linewidth=linewidth)
            ax.add_patch(rect)

            value = game.board[square]
            if value != 0:
                fill = "#181f2a" if value == game.to_play else "#fafafa"
                stroke = "#111827" if value == game.to_play else "#6b7280"
                disc = plt.Circle((col + 0.5, row + 0.5), 0.31, facecolor=fill, edgecolor=stroke, linewidth=1.1)
                ax.add_patch(disc)
            if square == move_square:
                ax.text(col + 0.5, row + 0.54, "x", ha="center", va="center", fontsize=8, color="#083b7a", weight="bold")

    for line in move_info["capture_lines"]:
        if not line:
            continue
        start_row, start_col = square_to_rc(move_square)
        if "failed_runs" in move_info:
            matching_runs = [run for run in move_info["failed_runs"] if run["line"] == line]
            end_square = matching_runs[0]["failure_square"] if matching_runs else line[-1]
            if end_square is None:
                end_square = line[-1]
        else:
            end_square = terminator_square_for_line(move_square, line) or line[-1]
        end_row, end_col = square_to_rc(end_square)
        ax.annotate(
            "",
            xy=(end_col + 0.5, end_row + 0.5),
            xytext=(start_col + 0.5, start_row + 0.5),
            arrowprops={"arrowstyle": "->", "color": "#0f5132", "lw": 1.5},
        )

    ax.set_title(
        f"{index + 1}. {move_info['label']}  run={move_info['num_flipped']}  "
        f"failed lines={move_info['num_capture_lines']}  prefix={position['prefix_len']}",
        loc="left",
        pad=2,
        fontsize=8.5,
        color="#172033",
        fontweight="bold",
    )


def draw_layer_rows_for_position(
    ax,
    layer_rows: list[dict],
    stats_by_layer: dict[int, list[dict]],
    position_index: int,
    min_ratio: float,
    max_ratio: float,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(8, 0)
    ax.set_anchor("N")
    ax.axis("off")

    for display_row, layer_row in enumerate(layer_rows):
        layer = layer_row["layer"]
        stats = stats_by_layer[layer][position_index]
        ratio = stats["capture_union_ratio"]
        color = layer_color(ratio, min_ratio, max_ratio)
        rect = plt.Rectangle(
            (0.02, display_row + 0.08),
            0.92,
            0.78,
            facecolor=color,
            edgecolor="#b9c2b4",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        text_color = "#ffffff" if ratio > (min_ratio + 0.65 * (max_ratio - min_ratio)) else "#172033"
        ax.text(
            0.08,
            display_row + 0.47,
            f"L{layer}",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            0.88,
            display_row + 0.47,
            f"{ratio:.2f}",
            ha="right",
            va="center",
            fontsize=7.7,
            color=text_color,
        )


def render_png(
    output_path: Path,
    positions: list[dict],
    layer_rows: list[dict],
    stats_by_layer: dict[int, list[dict]],
    num_boards: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    displayed_count = min(num_boards, len(positions))
    fig = plt.figure(figsize=(8.2, 19), dpi=180)
    grid = fig.add_gridspec(displayed_count + 1, 2, width_ratios=[1.0, 0.9], height_ratios=[0.34] + [1.0] * displayed_count, hspace=0.24, wspace=0.18)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.035)

    ratios = [
        stats_by_layer[row["layer"]][position_index]["capture_union_ratio"]
        for position_index in range(displayed_count)
        for row in layer_rows
    ]
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    ax_left_header = fig.add_subplot(grid[0, 0])
    ax_left_header.axis("off")
    ax_left_header.text(0, 0.75, "Non-valid capture-like board", ha="left", va="top", fontsize=11, weight="bold", color="#172033")
    ax_left_header.text(0, 0.25, "blue = illegal target, gold = opponent run, red = missing terminator", ha="left", va="top", fontsize=7.5, color="#5c667a")

    ax_right_header = fig.add_subplot(grid[0, 1])
    ax_right_header.axis("off")
    ax_right_header.text(0, 0.75, "Per-board layer sensitivity", ha="left", va="top", fontsize=11, weight="bold", color="#172033")
    ax_right_header.text(
        0.0,
        0.25,
        "Each row is that board's invalid-line ratio; colors share one global scale.",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#5c667a",
        wrap=True,
    )

    for idx, position in enumerate(positions[:displayed_count]):
        draw_board(fig.add_subplot(grid[idx + 1, 0]), position, idx)
        draw_layer_rows_for_position(
            fig.add_subplot(grid[idx + 1, 1]),
            layer_rows=layer_rows,
            stats_by_layer=stats_by_layer,
            position_index=idx,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )

    fig.suptitle("Othello-GPT: non-valid capture-like line sensitivity", x=0.03, y=0.996, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.03,
        0.982,
        "Every board row has its own 8-layer heatmap for an illegal target with an opponent run but no friendly terminator.",
        ha="left",
        va="top",
        fontsize=8,
        color="#5c667a",
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def serialize_position(position: dict) -> dict:
    move_info = position["move_info"]
    return {
        "prefix_len": int(position["prefix_len"]),
        "prefix_token_ids": [int(token_id) for token_id in position["prefix_token_ids"]],
        "to_play": int(position["game"].to_play),
        "board": [int(value) for value in position["game"].board],
        "chosen_move": {
            "label": move_info["label"],
            "square_index": int(move_info["square_index"]),
            "token_id": int(move_info["token_id"]),
            "opponent_run_squares": int(move_info["num_flipped"]),
            "num_failed_lines": int(move_info["num_capture_lines"]),
            "capture_line_labels": move_info["capture_line_labels"],
            "failed_runs": [
                {
                    "line": [square_label(square) for square in run["line"]],
                    "failure_square": None if run["failure_square"] is None else square_label(run["failure_square"]),
                    "failure_kind": run["failure_kind"],
                }
                for run in move_info.get("failed_runs", [])
            ],
        },
    }


def serialize_position_layer_scores(
    positions: list[dict],
    layer_rows: list[dict],
    stats_by_layer: dict[int, list[dict]],
    num_boards: int,
) -> list[dict]:
    displayed_count = min(num_boards, len(positions))
    rows = []
    for position_index, position in enumerate(positions[:displayed_count]):
        rows.append(
            {
                "position_index": position_index,
                "illegal_target": position["move_info"]["label"],
                "prefix_len": int(position["prefix_len"]),
                "layers": [
                    {
                        "layer": layer_row["layer"],
                        "capture_vs_unrelated_ratio": float(
                            stats_by_layer[layer_row["layer"]][position_index]["capture_union_ratio"]
                        ),
                        "capture_minus_unrelated": float(
                            stats_by_layer[layer_row["layer"]][position_index]["capture_union_mean"]
                            - stats_by_layer[layer_row["layer"]][position_index]["unrelated_occupied_mean"]
                        ),
                    }
                    for layer_row in layer_rows
                ],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("chapter07_nonvalid_capture_line_layer_sensitivity.png"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--num-positions", type=int, default=50)
    parser.add_argument("--num-boards", type=int, default=10)
    parser.add_argument("--min-prefix-len", type=int, default=12)
    parser.add_argument("--max-prefix-len", type=int, default=45)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(True)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    model = load_model(device)

    positions = sample_nonvalid_capture_positions(
        num_positions=args.num_positions,
        min_prefix_len=args.min_prefix_len,
        max_prefix_len=args.max_prefix_len,
        seed=LEGALITY_RANDOM_SEED,
    )
    print(f"Sampled {len(positions)} positions with non-valid capture-like lines.", flush=True)

    scorer = Chapter7Scorer(model=model, device=device, probe_epochs=args.probe_epochs)
    layer_rows, stats_by_layer = scorer.nonvalid_layer_summary(positions=positions, layers=args.layers)

    render_png(
        args.output,
        positions=positions,
        layer_rows=layer_rows,
        stats_by_layer=stats_by_layer,
        num_boards=args.num_boards,
    )
    print(f"Wrote {args.output}", flush=True)

    json_output = args.json_output
    if json_output is None:
        json_output = args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_notebook": "demos/Othello_GPT_Jacobian_Lens.ipynb",
        "source_sections": [
            "Chapter 7 helpers plus invalid opponent-run sampling inspired by Chapter 9 condition dataset",
        ],
        "configuration": {
            "num_positions": args.num_positions,
            "num_boards": args.num_boards,
            "layers": args.layers,
            "probe_epochs": args.probe_epochs,
            "probe_num_games": PROBE_NUM_GAMES,
            "legality_random_seed": LEGALITY_RANDOM_SEED,
            "score": "For each illegal target with an opponent run but no friendly terminator, mean absolute invalid-target-vs-legal-gradient projection on opponent-run plus failure squares divided by non-support controls.",
        },
        "layer_summary": layer_rows,
        "displayed_position_layer_scores": serialize_position_layer_scores(
            positions=positions,
            layer_rows=layer_rows,
            stats_by_layer=stats_by_layer,
            num_boards=args.num_boards,
        ),
        "displayed_positions": [serialize_position(position) for position in positions[: args.num_boards]],
    }
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
