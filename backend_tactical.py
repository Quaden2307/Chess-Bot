"""
Tactical Chess AI Backend Server

Uses the tactically-trained neural network with enhanced evaluation features
for better understanding of:
- Material value and captures
- King safety
- Pins, forks, and hanging pieces
- Positional factors
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import chess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__, static_folder='chess-frontend/build', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

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

# Piece-square tables
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
    """Determine if position is endgame"""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minor_pieces = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                   len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
    if queens == 0:
        return True
    if queens <= 2 and minor_pieces <= 2 and rooks <= 2:
        return True
    return False


def get_piece_square_value(piece, square, endgame=False):
    """Get piece-square table value"""
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
    """Detect pinned pieces"""
    pinned_count = 0
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                pinned_count += 1
    return pinned_count


def detect_hanging_pieces(board, color):
    """Detect undefended pieces"""
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
                min_attacker_value = min(
                    PIECE_VALUES[board.piece_at(a).piece_type]
                    for a in attackers if board.piece_at(a)
                )
                if min_attacker_value < PIECE_VALUES[piece.piece_type]:
                    hanging.append((sq, piece, PIECE_VALUES[piece.piece_type] - min_attacker_value))
    return hanging


def count_attackers_near_king(board, color):
    """Count enemy pieces attacking king zone"""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    opponent = not color
    attack_count = 0
    king_zone = [king_sq] + list(board.attacks(king_sq))
    for sq in king_zone:
        attackers = board.attackers(opponent, sq)
        attack_count += len(attackers)
    return attack_count


def king_on_open_file(board, color):
    """Check if king is on open file"""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    king_file = chess.square_file(king_sq)
    own_pawns = 0
    for rank in range(8):
        sq = chess.square(king_file, rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            own_pawns += 1
    return own_pawns == 0


def detect_forks(board, color):
    """Detect knight/pawn forks"""
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


# ========== TACTICAL ENCODING (798 features) ==========

def board_to_tactical_tensor(board: chess.Board):
    """Encode board with tactical features"""
    features = []
    endgame = is_endgame(board)

    # 1. Piece positions (768 features)
    piece_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        piece_tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0
    features.extend(piece_tensor.flatten().tolist())

    # 2. Material balance
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    features.append((white_material - black_material) / 3000.0)

    # 3. Piece-square evaluation
    white_psq = sum(get_piece_square_value(p, sq, endgame) for sq, p in board.piece_map().items() if p.color == chess.WHITE)
    black_psq = sum(get_piece_square_value(p, sq, endgame) for sq, p in board.piece_map().items() if p.color == chess.BLACK)
    features.append((white_psq - black_psq) / 500.0)

    # 4. Mobility
    current_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    board.turn = current_turn
    features.append((white_mobility - black_mobility) / 30.0)

    # 5. Center control
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    extended_center = [chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.F4,
                       chess.C5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6]

    white_center = sum(1 for sq in center_squares if len(board.attackers(chess.WHITE, sq)) > len(board.attackers(chess.BLACK, sq)))
    black_center = sum(1 for sq in center_squares if len(board.attackers(chess.BLACK, sq)) > len(board.attackers(chess.WHITE, sq)))
    features.append((white_center - black_center) / 4.0)

    white_ext = sum(1 for sq in extended_center if len(board.attackers(chess.WHITE, sq)) > 0)
    black_ext = sum(1 for sq in extended_center if len(board.attackers(chess.BLACK, sq)) > 0)
    features.append((white_ext - black_ext) / 12.0)

    # 6. King safety (6 features)
    white_king_attackers = count_attackers_near_king(board, chess.WHITE)
    black_king_attackers = count_attackers_near_king(board, chess.BLACK)
    features.append((black_king_attackers - white_king_attackers) / 20.0)

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

    white_king_open = 1.0 if king_on_open_file(board, chess.WHITE) else 0.0
    black_king_open = 1.0 if king_on_open_file(board, chess.BLACK) else 0.0
    features.append(black_king_open - white_king_open)

    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # 7. Tactical features (8 features)
    white_pins = detect_pins(board, chess.WHITE)
    black_pins = detect_pins(board, chess.BLACK)
    features.append((black_pins - white_pins) / 4.0)

    white_hanging = detect_hanging_pieces(board, chess.WHITE)
    black_hanging = detect_hanging_pieces(board, chess.BLACK)
    white_hanging_value = sum(h[2] for h in white_hanging) / 1000.0
    black_hanging_value = sum(h[2] for h in black_hanging) / 1000.0
    features.append(black_hanging_value - white_hanging_value)
    features.append((len(black_hanging) - len(white_hanging)) / 4.0)

    white_forks = detect_forks(board, chess.WHITE)
    black_forks = detect_forks(board, chess.BLACK)
    features.append((white_forks - black_forks) / 3.0)

    features.append(1.0 if board.is_check() else 0.0)

    board.turn = chess.WHITE
    white_checks = sum(1 for m in board.legal_moves if board.gives_check(m))
    board.turn = chess.BLACK
    black_checks = sum(1 for m in board.legal_moves if board.gives_check(m))
    board.turn = current_turn
    features.append((white_checks - black_checks) / 5.0)

    board.turn = chess.WHITE
    white_captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    board.turn = chess.BLACK
    black_captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    board.turn = current_turn
    features.append((white_captures - black_captures) / 10.0)

    def best_capture_value(board, color):
        board.turn = color
        best = 0
        for m in board.legal_moves:
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                if captured:
                    best = max(best, PIECE_VALUES[captured.piece_type])
        return best

    white_best_cap = best_capture_value(board, chess.WHITE) / 900.0
    black_best_cap = best_capture_value(board, chess.BLACK) / 900.0
    board.turn = current_turn
    features.append(white_best_cap - black_best_cap)

    # 8. Pawn structure (3 features)
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))

    white_doubled = sum(1 for f in range(8) if sum(1 for p in white_pawns if chess.square_file(p) == f) > 1)
    black_doubled = sum(1 for f in range(8) if sum(1 for p in black_pawns if chess.square_file(p) == f) > 1)
    features.append((black_doubled - white_doubled) / 4.0)

    def is_isolated(pawn_sq, pawns):
        pawn_file = chess.square_file(pawn_sq)
        for p in pawns:
            if p != pawn_sq and abs(chess.square_file(p) - pawn_file) == 1:
                return False
        return True

    white_isolated = sum(1 for p in white_pawns if is_isolated(p, white_pawns))
    black_isolated = sum(1 for p in black_pawns if is_isolated(p, black_pawns))
    features.append((black_isolated - white_isolated) / 4.0)

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

    # 9. Game phase
    total_material = white_material + black_material - 40000
    game_phase = max(0, min(1, 1.0 - total_material / 6200.0))
    features.append(game_phase)

    # 10. Turn indicator
    features.append(1.0 if board.turn == chess.WHITE else -1.0)

    return torch.tensor(features, dtype=torch.float32)


# ========== MODEL ==========

class TacticalChessNet(nn.Module):
    def __init__(self, input_size=793):
        super().__init__()
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


# Load the trained model
model = TacticalChessNet(input_size=793)
MODEL_LOADED = False

# Try loading models in order of preference
model_files = [
    "chess_model_tactical_best.pth",
    "chess_model_tactical.pth",
    "chess_model_improved.pth",
    "chess_model_best.pth"
]

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            # Check if model architecture matches (793 features for tactical model)
            first_layer_shape = list(state_dict.values())[0].shape[1]
            if first_layer_shape == 793:
                model.load_state_dict(state_dict)
                model.eval()
                print(f"Tactical model loaded from {model_file}!")
                MODEL_LOADED = True
                break
            else:
                print(f"Skipping {model_file} - incompatible architecture (expected 793, got {first_layer_shape})")
        except Exception as e:
            print(f"Could not load {model_file}: {e}")

if not MODEL_LOADED:
    print("WARNING: No compatible tactical model found. Using random evaluation.")


# ========== MOVE SELECTION WITH TACTICAL SEARCH ==========

def minimax_eval(board, depth, alpha, beta, maximizing):
    """Simple minimax with alpha-beta for better move selection"""
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)

    if maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_eval(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_eval(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def evaluate_position(board):
    """Evaluate position using neural network"""
    if board.is_checkmate():
        return -10.0 if board.turn == chess.WHITE else 10.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    if not MODEL_LOADED:
        # Fallback to material count
        white_mat = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
        black_mat = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
        return (white_mat - black_mat) / 3000.0

    try:
        with torch.no_grad():
            x = board_to_tactical_tensor(board).unsqueeze(0)
            value = model(x).item()
        return value
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0


def get_best_moves(board, top_n=3, search_depth=2):
    """Get best moves using minimax search"""
    moves_with_scores = []
    is_white = board.turn == chess.WHITE

    for move in board.legal_moves:
        board.push(move)
        # Use minimax for better tactical awareness
        if search_depth > 1:
            score = minimax_eval(board, search_depth - 1, float('-inf'), float('inf'), not is_white)
        else:
            score = evaluate_position(board)

        # Flip score based on side to move
        if not is_white:
            score = -score

        board.pop()

        moves_with_scores.append({
            'move': move.uci(),
            'score': score,
            'san': board.san(move)
        })

    # Sort: white wants high scores, black wants low scores
    moves_with_scores.sort(key=lambda x: x['score'], reverse=is_white)
    return moves_with_scores[:top_n]


def choose_best_move(board):
    """Choose the best move"""
    best_moves = get_best_moves(board, top_n=1, search_depth=2)
    if best_moves:
        return best_moves[0]
    return None


# ========== API ENDPOINTS ==========

@app.route('/api/health', methods=['GET'])
def health():
    import sys
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'model_type': 'tactical',
        'python_version': sys.version
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        fen = data.get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        evaluation = evaluate_position(board)

        return jsonify({
            'evaluation': round(evaluation, 3),
            'fen': fen
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-move', methods=['POST'])
def best_move():
    try:
        data = request.json
        fen = data.get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({'error': 'Game is over'}), 400

        move_data = choose_best_move(board)
        if not move_data:
            return jsonify({'error': 'No legal moves available'}), 400

        return jsonify({
            'move': move_data['move'],
            'san': move_data['san'],
            'evaluation': round(move_data['score'], 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggested-moves', methods=['POST'])
def suggested_moves():
    try:
        data = request.json
        fen = data.get('fen')
        top_n = data.get('top_n', 3)
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({'error': 'Game is over'}), 400

        moves = get_best_moves(board, top_n=top_n, search_depth=2)
        return jsonify({
            'moves': moves,
            'current_evaluation': round(evaluate_position(board), 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/make-ai-move', methods=['POST'])
def make_ai_move():
    try:
        data = request.json
        fen = data.get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({
                'error': 'Game is over',
                'result': board.result()
            }), 400

        move_data = choose_best_move(board)
        if not move_data:
            # Fallback to random
            import random
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return jsonify({'error': 'No legal moves'}), 400
            move = random.choice(legal_moves)
            move_data = {'move': move.uci(), 'san': board.san(move), 'score': 0.0}

        move = chess.Move.from_uci(move_data['move'])
        board.push(move)

        return jsonify({
            'move': move_data['move'],
            'san': move_data['san'],
            'new_fen': board.fen(),
            'evaluation': round(evaluate_position(board), 3),
            'is_game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Frontend routes
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    print("Starting Tactical Chess AI Backend Server...")
    port = int(os.environ.get('PORT', 5001))
    print(f"Server running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', debug=False, port=port)
