from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import chess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add parent directory to path to import from backend_improved
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ========== ENCODING ==========

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

def board_to_enhanced_tensor(board: chess.Board):
    """Enhanced board representation with tactical features"""
    # Basic piece positions (12 channels, 8x8)
    piece_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        piece_tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0

    # Flatten piece positions
    features = piece_tensor.flatten().tolist()

    # Material count (normalized)
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = (white_material - black_material) / 10000.0
    features.append(material_advantage)

    # Mobility (number of legal moves)
    white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.push(chess.Move.null())  # Switch turn temporarily
    if board.turn == chess.BLACK:
        white_mobility = len(list(board.legal_moves))
    else:
        black_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_advantage = (white_mobility - black_mobility) / 100.0
    features.append(mobility_advantage)

    # King safety (castling rights)
    white_kingside = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    white_queenside = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    black_kingside = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    black_queenside = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    features.extend([white_kingside, white_queenside, black_kingside, black_queenside])

    # Center control (pieces in center squares)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    black_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
    center_control = (white_center - black_center) / 4.0
    features.append(center_control)

    # Check status
    in_check = 1.0 if board.is_check() else 0.0
    features.append(in_check)

    # Turn indicator
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    features.append(turn)

    # Pawn structure
    white_pawns = len([p for p in board.piece_map().values() if p.piece_type == chess.PAWN and p.color == chess.WHITE])
    black_pawns = len([p for p in board.piece_map().values() if p.piece_type == chess.PAWN and p.color == chess.BLACK])
    pawn_advantage = (white_pawns - black_pawns) / 8.0
    features.append(pawn_advantage)

    # Game phase (endgame indicator)
    total_pieces = len(board.piece_map())
    game_phase = (32 - total_pieces) / 32.0  # 0 = opening, 1 = endgame
    features.append(game_phase)

    return torch.tensor(features, dtype=torch.float32)

# ========== MODEL ==========

class ImprovedChessNet(nn.Module):
    def __init__(self, input_size=780, hidden_sizes=[512, 256, 128, 64]):
        super(ImprovedChessNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])

        self.fc5 = nn.Linear(hidden_sizes[3], 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = torch.tanh(self.fc5(x))
        return x

# Load model
input_size = 768 + 12  # 768 from piece positions + 12 features
model = ImprovedChessNet(input_size=input_size)

# Try to load the model file
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chess_model_improved.pth')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chess_model_best.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    print("Warning: No model file found. Using untrained model.")

# ========== EVALUATION FUNCTIONS ==========

def detect_mate_in_n(board, max_depth=5):
    """
    Detect if there's a forced mate within max_depth moves.
    Returns integer: positive for white advantage, negative for black advantage, or None
    """
    # Check for immediate checkmate
    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1

    # Simple mate detection: check if any move leads to checkmate
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return 1 if board.turn == chess.WHITE else -1
        board.pop()

    # For deeper search, only check 2-3 moves ahead due to performance
    if max_depth >= 2:
        for move in board.legal_moves:
            board.push(move)
            for move2 in board.legal_moves:
                board.push(move2)
                if board.is_checkmate():
                    board.pop()
                    board.pop()
                    return 2 if board.turn == chess.WHITE else -2
                board.pop()
            board.pop()

    return None

def evaluate_position(board):
    """Evaluate the current position"""
    # Check for mate first
    mate_in = detect_mate_in_n(board, max_depth=5)
    if mate_in is not None:
        # Return a large value for mate positions
        return 10.0 if mate_in > 0 else -10.0

    with torch.no_grad():
        x = board_to_enhanced_tensor(board).unsqueeze(0)  # Add batch dimension
        value = model(x).item()
    return value

def choose_best_move_improved(board, depth=2):
    """Choose the best move using minimax with alpha-beta pruning"""
    def minimax(board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return evaluate_position(board), None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return evaluate_position(board), None

        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    _, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn == chess.WHITE)
    return best_move

def get_top_moves(board, top_n=3):
    """Get top N moves with their evaluations"""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    move_scores = []
    for move in legal_moves:
        board.push(move)
        score = evaluate_position(board)
        board.pop()
        move_scores.append({
            'move': move.uci(),
            'san': board.san(move),
            'score': round(score, 3)
        })

    # Sort by score (higher is better for white)
    if board.turn == chess.WHITE:
        move_scores.sort(key=lambda x: x['score'], reverse=True)
    else:
        move_scores.sort(key=lambda x: x['score'])

    return move_scores[:top_n]

# ========== API ENDPOINTS ==========

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate a position given FEN string"""
    try:
        data = request.json
        fen = data.get('fen')

        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)

        # Check for mate first
        mate_in = detect_mate_in_n(board, max_depth=5)

        # Get normal evaluation
        evaluation = evaluate_position(board)

        response = {
            'evaluation': round(evaluation, 3),
            'fen': fen
        }

        if mate_in is not None:
            response['mate_in'] = mate_in

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/best-move', methods=['POST'])
def best_move():
    """Get the best move for a position"""
    try:
        data = request.json
        fen = data.get('fen')

        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        move = choose_best_move_improved(board)

        if move is None:
            return jsonify({'error': 'No legal moves available'}), 400

        return jsonify({
            'move': move.uci(),
            'san': board.san(move),
            'evaluation': round(evaluate_position(board), 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggested-moves', methods=['POST'])
def suggested_moves():
    """Get top N suggested moves"""
    try:
        data = request.json
        fen = data.get('fen')
        top_n = data.get('top_n', 3)

        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        moves = get_top_moves(board, top_n)
        current_eval = evaluate_position(board)

        return jsonify({
            'moves': moves,
            'current_evaluation': round(current_eval, 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/make-ai-move', methods=['POST'])
def make_ai_move():
    """Make an AI move and return the new position"""
    try:
        data = request.json
        fen = data.get('fen')

        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)
        move = choose_best_move_improved(board)

        if move is None:
            return jsonify({'error': 'No legal moves available'}), 400

        san = board.san(move)
        board.push(move)
        new_fen = board.fen()
        evaluation = evaluate_position(board)

        return jsonify({
            'move': move.uci(),
            'san': san,
            'new_fen': new_fen,
            'evaluation': round(evaluation, 3),
            'is_game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel serverless function handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()
