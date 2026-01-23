#!/usr/bin/env python3
"""
Advanced Chess AI Training with Tactical Awareness

This script trains a chess AI that understands:
- Material value and when to capture pieces
- King safety and avoiding unsafe king moves
- Pins, forks, skewers, and other tactics
- Positional understanding (center control, piece activity)
- Opening principles and endgame basics
"""

import torch
import numpy as np
import chess
import chess.engine
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import os

# ========== PIECE VALUES AND TABLES ==========

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables (from white's perspective)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_MIDDLEGAME_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]


def is_endgame(board):
    """Determine if position is endgame based on material"""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minor_pieces = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                   len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))

    # Endgame if no queens, or queen + minor piece or less
    if queens == 0:
        return True
    if queens <= 2 and minor_pieces <= 2 and rooks <= 2:
        return True
    return False


def get_piece_square_value(piece, square, endgame=False):
    """Get piece-square table value for a piece at a given square"""
    if piece.color == chess.BLACK:
        square = chess.square_mirror(square)

    if piece.piece_type == chess.PAWN:
        return PAWN_TABLE[square]
    elif piece.piece_type == chess.KNIGHT:
        return KNIGHT_TABLE[square]
    elif piece.piece_type == chess.BISHOP:
        return BISHOP_TABLE[square]
    elif piece.piece_type == chess.ROOK:
        return ROOK_TABLE[square]
    elif piece.piece_type == chess.QUEEN:
        return QUEEN_TABLE[square]
    elif piece.piece_type == chess.KING:
        if endgame:
            return KING_ENDGAME_TABLE[square]
        return KING_MIDDLEGAME_TABLE[square]
    return 0


# ========== TACTICAL DETECTION ==========

def detect_pins(board, color):
    """Detect pinned pieces for a color"""
    pinned_count = 0
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    # Check all pieces that could be pinned
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                pinned_count += 1

    return pinned_count


def detect_hanging_pieces(board, color):
    """Detect pieces that are attacked but not defended"""
    hanging = []
    opponent = not color

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            attackers = board.attackers(opponent, sq)
            defenders = board.attackers(color, sq)

            if attackers and not defenders:
                hanging.append((sq, piece, PIECE_VALUES[piece.piece_type]))
            elif attackers:
                # Check if lowest value attacker < piece value
                min_attacker_value = min(
                    PIECE_VALUES[board.piece_at(a).piece_type]
                    for a in attackers if board.piece_at(a)
                )
                if min_attacker_value < PIECE_VALUES[piece.piece_type]:
                    hanging.append((sq, piece, PIECE_VALUES[piece.piece_type] - min_attacker_value))

    return hanging


def count_attackers_near_king(board, color):
    """Count enemy pieces attacking squares near the king"""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    opponent = not color
    attack_count = 0

    # Check king and adjacent squares
    king_zone = [king_sq] + list(board.attacks(king_sq))

    for sq in king_zone:
        attackers = board.attackers(opponent, sq)
        attack_count += len(attackers)

    return attack_count


def has_castling_available(board, color):
    """Check if castling is still available"""
    if color == chess.WHITE:
        return board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE)
    return board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)


def king_on_open_file(board, color):
    """Check if king is on an open or semi-open file"""
    king_sq = board.king(color)
    if king_sq is None:
        return False

    king_file = chess.square_file(king_sq)

    # Check for pawns on the king's file
    own_pawns = 0
    enemy_pawns = 0

    for rank in range(8):
        sq = chess.square(king_file, rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == color:
                own_pawns += 1
            else:
                enemy_pawns += 1

    return own_pawns == 0  # Open or semi-open


def detect_forks(board, color):
    """Detect potential knight/pawn forks"""
    fork_count = 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in [chess.KNIGHT, chess.PAWN]:
                attacks = board.attacks(sq)
                valuable_targets = 0
                for target_sq in attacks:
                    target = board.piece_at(target_sq)
                    if target and target.color != color:
                        if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            valuable_targets += 1

                if valuable_targets >= 2:
                    fork_count += 1

    return fork_count


# ========== ADVANCED ENCODING ==========

def board_to_tactical_tensor(board: chess.Board):
    """
    Advanced board encoding with heavy emphasis on tactical features.
    Total features: 768 + 30 = 798
    """
    features = []
    endgame = is_endgame(board)

    # 1. Piece positions (12 channels * 64 squares = 768 features)
    piece_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        piece_tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0
    features.extend(piece_tensor.flatten().tolist())

    # 2. Material balance (1 feature)
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = (white_material - black_material) / 3000.0  # Normalized to ~[-1, 1]
    features.append(material_advantage)

    # 3. Piece-square evaluation (1 feature)
    white_psq = sum(get_piece_square_value(p, sq, endgame)
                    for sq, p in board.piece_map().items() if p.color == chess.WHITE)
    black_psq = sum(get_piece_square_value(p, sq, endgame)
                    for sq, p in board.piece_map().items() if p.color == chess.BLACK)
    psq_advantage = (white_psq - black_psq) / 500.0
    features.append(psq_advantage)

    # 4. Mobility (1 feature)
    current_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    board.turn = current_turn
    mobility_advantage = (white_mobility - black_mobility) / 30.0
    features.append(mobility_advantage)

    # 5. Center control (2 features)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    extended_center = [chess.C3, chess.D3, chess.E3, chess.F3,
                       chess.C4, chess.F4, chess.C5, chess.F5,
                       chess.C6, chess.D6, chess.E6, chess.F6]

    white_center = sum(1 for sq in center_squares
                       if len(board.attackers(chess.WHITE, sq)) > len(board.attackers(chess.BLACK, sq)))
    black_center = sum(1 for sq in center_squares
                       if len(board.attackers(chess.BLACK, sq)) > len(board.attackers(chess.WHITE, sq)))
    features.append((white_center - black_center) / 4.0)

    white_ext = sum(1 for sq in extended_center if len(board.attackers(chess.WHITE, sq)) > 0)
    black_ext = sum(1 for sq in extended_center if len(board.attackers(chess.BLACK, sq)) > 0)
    features.append((white_ext - black_ext) / 12.0)

    # 6. KING SAFETY (6 features) - CRITICAL
    # 6a. Attackers near king
    white_king_attackers = count_attackers_near_king(board, chess.WHITE)
    black_king_attackers = count_attackers_near_king(board, chess.BLACK)
    features.append((black_king_attackers - white_king_attackers) / 20.0)  # More attackers = worse

    # 6b. Pawn shield (pawns in front of king)
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    white_pawn_shield = 0
    black_pawn_shield = 0

    if white_king_sq:
        king_file = chess.square_file(white_king_sq)
        king_rank = chess.square_rank(white_king_sq)
        for f in [max(0, king_file-1), king_file, min(7, king_file+1)]:
            for r in [king_rank+1, king_rank+2]:
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        white_pawn_shield += 1

    if black_king_sq:
        king_file = chess.square_file(black_king_sq)
        king_rank = chess.square_rank(black_king_sq)
        for f in [max(0, king_file-1), king_file, min(7, king_file+1)]:
            for r in [king_rank-1, king_rank-2]:
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        black_pawn_shield += 1

    features.append((white_pawn_shield - black_pawn_shield) / 6.0)

    # 6c. King on open file (dangerous)
    white_king_open = 1.0 if king_on_open_file(board, chess.WHITE) else 0.0
    black_king_open = 1.0 if king_on_open_file(board, chess.BLACK) else 0.0
    features.append(black_king_open - white_king_open)  # Negative if white king exposed

    # 6d. Castling rights
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # 7. TACTICAL FEATURES (8 features) - CRITICAL
    # 7a. Pins
    white_pins = detect_pins(board, chess.WHITE)
    black_pins = detect_pins(board, chess.BLACK)
    features.append((black_pins - white_pins) / 4.0)  # More pins on opponent = better

    # 7b. Hanging pieces
    white_hanging = detect_hanging_pieces(board, chess.WHITE)
    black_hanging = detect_hanging_pieces(board, chess.BLACK)
    white_hanging_value = sum(h[2] for h in white_hanging) / 1000.0
    black_hanging_value = sum(h[2] for h in black_hanging) / 1000.0
    features.append(black_hanging_value - white_hanging_value)

    # 7c. Number of hanging pieces
    features.append((len(black_hanging) - len(white_hanging)) / 4.0)

    # 7d. Forks
    white_forks = detect_forks(board, chess.WHITE)
    black_forks = detect_forks(board, chess.BLACK)
    features.append((white_forks - black_forks) / 3.0)

    # 7e. Check status
    features.append(1.0 if board.is_check() else 0.0)

    # 7f. Checks available
    board.turn = chess.WHITE
    white_checks = sum(1 for m in board.legal_moves if board.gives_check(m))
    board.turn = chess.BLACK
    black_checks = sum(1 for m in board.legal_moves if board.gives_check(m))
    board.turn = current_turn
    features.append((white_checks - black_checks) / 5.0)

    # 7g. Capture moves available
    board.turn = chess.WHITE
    white_captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    board.turn = chess.BLACK
    black_captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    board.turn = current_turn
    features.append((white_captures - black_captures) / 10.0)

    # 7h. Best capture value
    def best_capture_value(board, color):
        board.turn = color
        best = 0
        for m in board.legal_moves:
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                if captured:
                    val = PIECE_VALUES[captured.piece_type]
                    attacker = board.piece_at(m.from_square)
                    if attacker:
                        # Net value = captured - attacker if we lose the piece
                        net = val  # Simplified: just the capture value
                        best = max(best, net)
        return best

    white_best_cap = best_capture_value(board, chess.WHITE) / 900.0
    black_best_cap = best_capture_value(board, chess.BLACK) / 900.0
    board.turn = current_turn
    features.append(white_best_cap - black_best_cap)

    # 8. Pawn structure (3 features)
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))

    # Doubled pawns
    white_doubled = sum(1 for f in range(8) if sum(1 for p in white_pawns if chess.square_file(p) == f) > 1)
    black_doubled = sum(1 for f in range(8) if sum(1 for p in black_pawns if chess.square_file(p) == f) > 1)
    features.append((black_doubled - white_doubled) / 4.0)

    # Isolated pawns
    def is_isolated(pawn_sq, pawns):
        pawn_file = chess.square_file(pawn_sq)
        for p in pawns:
            if p != pawn_sq:
                if abs(chess.square_file(p) - pawn_file) == 1:
                    return False
        return True

    white_isolated = sum(1 for p in white_pawns if is_isolated(p, white_pawns))
    black_isolated = sum(1 for p in black_pawns if is_isolated(p, black_pawns))
    features.append((black_isolated - white_isolated) / 4.0)

    # Passed pawns
    def is_passed(pawn_sq, color, enemy_pawns):
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)

        for ep in enemy_pawns:
            ef = chess.square_file(ep)
            er = chess.square_rank(ep)
            if abs(ef - pawn_file) <= 1:
                if color == chess.WHITE and er > pawn_rank:
                    return False
                if color == chess.BLACK and er < pawn_rank:
                    return False
        return True

    white_passed = sum(1 for p in white_pawns if is_passed(p, chess.WHITE, black_pawns))
    black_passed = sum(1 for p in black_pawns if is_passed(p, chess.BLACK, white_pawns))
    features.append((white_passed - black_passed) / 4.0)

    # 9. Game phase (1 feature)
    total_material = white_material + black_material - 40000  # Subtract king values
    game_phase = max(0, min(1, 1.0 - total_material / 6200.0))
    features.append(game_phase)

    # 10. Turn indicator (1 feature)
    features.append(1.0 if board.turn == chess.WHITE else -1.0)

    # Total: 768 + 30 = 798 features
    return torch.tensor(features, dtype=torch.float32)


# ========== MODEL ==========

class TacticalChessNet(nn.Module):
    """Neural network with emphasis on tactical pattern recognition"""

    def __init__(self, input_size=798):
        super().__init__()

        # Larger network for better tactical understanding
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(x)), 0.1))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x)), 0.1))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x)), 0.1))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x)), 0.1))
        x = F.leaky_relu(self.fc5(x), 0.1)
        x = torch.tanh(self.fc6(x))
        return x


# ========== TRAINING DATA GENERATION ==========

# Diverse opening positions including tactical themes
OPENING_POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Common openings
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # Scandi
    # Italian Game positions (tactical)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    # Fried Liver Attack position
    "r1bqkb1r/ppp2ppp/2n2n2/3Pp3/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
    # Queen's Gambit
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3",
    # Tactical middlegame positions
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
    "r2qkb1r/ppp2ppp/2n1bn2/3pp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
    # Positions with tactical opportunities
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 5",
]

# Specific tactical puzzle positions (positions where tactics exist)
TACTICAL_POSITIONS = [
    # Pin positions
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    # Fork positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5",
    # Hanging piece positions
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # King safety issues
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
]


def generate_tactical_training_data(num_positions=15000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """Generate training positions focused on tactical and positional understanding"""
    X = []
    y = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Hash": 128, "Threads": 2})

    positions_per_category = num_positions // 5

    print(f"Generating {num_positions} training positions...")
    print("Categories: Openings, Middlegames, Endgames, Tactical puzzles, Random play")

    # Category 1: Opening positions (develop good opening play)
    print("\n[1/5] Generating opening positions...")
    for _ in tqdm(range(positions_per_category)):
        opening_fen = random.choice(OPENING_POSITIONS)
        board = chess.Board(opening_fen)

        # Play 0-10 moves to get early game positions
        num_moves = random.randint(0, 10)
        for _ in range(num_moves):
            if board.is_game_over():
                break

            # 60% Stockfish moves, 40% random (to explore variety)
            if random.random() < 0.6:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.05))
                    board.push(result.move)
                except:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(random.choice(legal_moves))
            else:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))

        if not board.is_game_over():
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=18))
                score = info["score"].relative

                if score.is_mate():
                    mate_in = score.mate()
                    eval_value = np.sign(mate_in) * (10.0 - min(abs(mate_in), 10) * 0.5)
                else:
                    eval_value = np.tanh(score.score() / 300.0)

                tensor = board_to_tactical_tensor(board)
                X.append(tensor)
                y.append(eval_value)
            except Exception as e:
                continue

    # Category 2: Middlegame positions (tactical complexity)
    print("\n[2/5] Generating middlegame positions...")
    for _ in tqdm(range(positions_per_category)):
        board = chess.Board()

        # Play 10-25 moves to get middlegame
        num_moves = random.randint(10, 25)
        for _ in range(num_moves):
            if board.is_game_over():
                break

            # 70% Stockfish for quality positions
            if random.random() < 0.7:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.03))
                    board.push(result.move)
                except:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(random.choice(legal_moves))
            else:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))

        if not board.is_game_over():
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=18))
                score = info["score"].relative

                if score.is_mate():
                    mate_in = score.mate()
                    eval_value = np.sign(mate_in) * (10.0 - min(abs(mate_in), 10) * 0.5)
                else:
                    eval_value = np.tanh(score.score() / 300.0)

                tensor = board_to_tactical_tensor(board)
                X.append(tensor)
                y.append(eval_value)
            except Exception as e:
                continue

    # Category 3: Endgame positions
    print("\n[3/5] Generating endgame positions...")
    for _ in tqdm(range(positions_per_category)):
        board = chess.Board()

        # Play many moves to reach endgame
        num_moves = random.randint(30, 60)
        for _ in range(num_moves):
            if board.is_game_over():
                break

            try:
                result = engine.play(board, chess.engine.Limit(time=0.02))
                board.push(result.move)
            except:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))

        if not board.is_game_over():
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=20))
                score = info["score"].relative

                if score.is_mate():
                    mate_in = score.mate()
                    eval_value = np.sign(mate_in) * (10.0 - min(abs(mate_in), 10) * 0.5)
                else:
                    eval_value = np.tanh(score.score() / 300.0)

                tensor = board_to_tactical_tensor(board)
                X.append(tensor)
                y.append(eval_value)
            except Exception as e:
                continue

    # Category 4: Positions with clear tactical themes
    print("\n[4/5] Generating tactical puzzle positions...")
    tactical_count = 0
    attempts = 0
    max_attempts = positions_per_category * 5

    with tqdm(total=positions_per_category) as pbar:
        while tactical_count < positions_per_category and attempts < max_attempts:
            attempts += 1

            # Start from tactical positions
            if random.random() < 0.3:
                opening_fen = random.choice(TACTICAL_POSITIONS)
            else:
                opening_fen = random.choice(OPENING_POSITIONS)

            board = chess.Board(opening_fen)

            # Play some moves
            num_moves = random.randint(5, 20)
            for _ in range(num_moves):
                if board.is_game_over():
                    break

                # Mix of random and engine moves to create imbalanced positions
                if random.random() < 0.5:
                    try:
                        result = engine.play(board, chess.engine.Limit(time=0.02))
                        board.push(result.move)
                    except:
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            board.push(random.choice(legal_moves))
                else:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(random.choice(legal_moves))

            if board.is_game_over():
                continue

            try:
                # Look for positions with clear evaluation (tactics present)
                info = engine.analyse(board, chess.engine.Limit(depth=18))
                score = info["score"].relative

                if score.is_mate():
                    mate_in = score.mate()
                    eval_value = np.sign(mate_in) * (10.0 - min(abs(mate_in), 10) * 0.5)
                    tensor = board_to_tactical_tensor(board)
                    X.append(tensor)
                    y.append(eval_value)
                    tactical_count += 1
                    pbar.update(1)
                elif abs(score.score()) > 100:  # Position with clear advantage (tactics)
                    eval_value = np.tanh(score.score() / 300.0)
                    tensor = board_to_tactical_tensor(board)
                    X.append(tensor)
                    y.append(eval_value)
                    tactical_count += 1
                    pbar.update(1)
            except Exception as e:
                continue

    # Category 5: Diverse random play positions
    print("\n[5/5] Generating diverse random positions...")
    for _ in tqdm(range(positions_per_category)):
        board = chess.Board()

        num_moves = random.randint(1, 50)
        for _ in range(num_moves):
            if board.is_game_over():
                break

            # More random to get diverse positions
            if random.random() < 0.3:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.02))
                    board.push(result.move)
                except:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(random.choice(legal_moves))
            else:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))

        if not board.is_game_over():
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=16))
                score = info["score"].relative

                if score.is_mate():
                    mate_in = score.mate()
                    eval_value = np.sign(mate_in) * (10.0 - min(abs(mate_in), 10) * 0.5)
                else:
                    eval_value = np.tanh(score.score() / 300.0)

                tensor = board_to_tactical_tensor(board)
                X.append(tensor)
                y.append(eval_value)
            except Exception as e:
                continue

    engine.quit()

    print(f"\nGenerated {len(X)} training positions")
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


# ========== TRAINING ==========

def train_model(model, X, y, epochs=100, batch_size=64, lr=1e-3):
    """Train the model with validation"""

    # Split data
    n = len(X)
    indices = torch.randperm(n)
    train_size = int(0.9 * n)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # Training
        model.train()
        train_indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[train_indices]
        y_train_shuffled = y_train[train_indices]

        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "chess_model_tactical_best.pth")
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    return model


# ========== MAIN ==========

if __name__ == "__main__":
    print("=" * 70)
    print("TACTICAL CHESS AI TRAINING")
    print("=" * 70)
    print("\nThis will train a chess AI with focus on:")
    print("  - Material awareness (capturing pieces)")
    print("  - King safety (protecting the king)")
    print("  - Tactical patterns (pins, forks, hanging pieces)")
    print("  - Positional understanding (center control, piece activity)")
    print()

    # Generate training data
    print("Step 1: Generating training data with Stockfish...")
    print("(This may take 10-20 minutes depending on your computer)")
    print()

    X, y = generate_tactical_training_data(num_positions=15000)

    print(f"\nGenerated {len(X)} training positions")
    print(f"Feature vector size: {X[0].shape[0]}")
    print(f"Evaluation range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Mean evaluation: {y.mean():.3f}")

    # Create model
    print("\nStep 2: Creating tactical neural network...")
    model = TacticalChessNet(input_size=X[0].shape[0])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nStep 3: Training model...")
    print("(This may take 10-30 minutes)")
    print()

    model = train_model(model, X, y, epochs=100, batch_size=64, lr=1e-3)

    # Save final model
    torch.save(model.state_dict(), "chess_model_tactical.pth")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModels saved:")
    print("  - chess_model_tactical.pth (final model)")
    print("  - chess_model_tactical_best.pth (best validation loss)")
    print("\nTo use the new model, update backend_improved.py to load")
    print("'chess_model_tactical_best.pth' and use the new encoding function.")
    print("=" * 70)
