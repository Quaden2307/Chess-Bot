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
        black_mobility = len(list(board.legal_moves))
    else:
        white_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_diff = (white_mobility - black_mobility) / 50.0
    features.append(mobility_diff)

    # King safety - distance from center and pawn shield
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is not None:
        wk_file = chess.square_file(white_king_square)
        wk_rank = chess.square_rank(white_king_square)
        white_king_center_dist = abs(wk_file - 3.5) + abs(wk_rank - 3.5)

        # Count pawns in front of white king
        white_pawn_shield = 0
        for file_offset in [-1, 0, 1]:
            f = wk_file + file_offset
            if 0 <= f < 8:
                for rank in range(wk_rank + 1, 8):
                    sq = chess.square(f, rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        white_pawn_shield += 1
                        break
    else:
        white_king_center_dist = 0
        white_pawn_shield = 0

    if black_king_square is not None:
        bk_file = chess.square_file(black_king_square)
        bk_rank = chess.square_rank(black_king_square)
        black_king_center_dist = abs(bk_file - 3.5) + abs(bk_rank - 3.5)

        # Count pawns in front of black king
        black_pawn_shield = 0
        for file_offset in [-1, 0, 1]:
            f = bk_file + file_offset
            if 0 <= f < 8:
                for rank in range(0, bk_rank):
                    sq = chess.square(f, rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        black_pawn_shield += 1
                        break
    else:
        black_king_center_dist = 0
        black_pawn_shield = 0

    features.append((white_pawn_shield - black_pawn_shield) / 3.0)
    features.append((black_king_center_dist - white_king_center_dist) / 10.0)

    # Castling rights
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # Check status
    features.append(1.0 if board.is_check() else 0.0)

    # Turn to move
    features.append(1.0 if board.turn == chess.WHITE else -1.0)

    # Center control (count pieces attacking center squares)
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    white_center_control = 0
    black_center_control = 0
    for sq in center_squares:
        white_center_control += len(board.attackers(chess.WHITE, sq))
        black_center_control += len(board.attackers(chess.BLACK, sq))
    features.append((white_center_control - black_center_control) / 10.0)

    # Pawn structure - doubled pawns
    white_pawn_files = [chess.square_file(sq) for sq, p in board.piece_map().items()
                        if p.piece_type == chess.PAWN and p.color == chess.WHITE]
    black_pawn_files = [chess.square_file(sq) for sq, p in board.piece_map().items()
                        if p.piece_type == chess.PAWN and p.color == chess.BLACK]

    white_doubled = sum(1 for f in range(8) if white_pawn_files.count(f) > 1)
    black_doubled = sum(1 for f in range(8) if black_pawn_files.count(f) > 1)
    features.append((black_doubled - white_doubled) / 8.0)

    return torch.tensor(features, dtype=torch.float32)

# ========== MODEL ==========

class ImprovedChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 768 + 12  # 768 from piece positions + 12 features

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))

        return x

# Load the trained model
model = ImprovedChessNet()
try:
    # Try to load improved model first
    model.load_state_dict(torch.load("chess_model_improved.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Improved model loaded successfully!")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load("chess_model_best.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Best model loaded successfully!")
    except FileNotFoundError:
        print("Warning: No trained model found. Please run improved_chess_ai.py first.")

# ========== MOVE SELECTION ==========

def detect_mate_in_n(board, max_depth=5):
    """
    Detect if there's a forced mate within max_depth moves.
    Returns integer: positive for white advantage, negative for black advantage, or None
    """
    # Check for immediate checkmate
    if board.is_checkmate():
        # The side to move is checkmated, so opponent wins
        # If white is in checkmate, return negative (black wins)
        # If black is in checkmate, return positive (white wins)
        return -1 if board.turn == chess.WHITE else 1

    # Simple mate detection: check if any move leads to checkmate
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            # Found checkmate in 1
            board.pop()
            return 1 if board.turn == chess.WHITE else -1
        board.pop()

    # For deeper search, only check 2-3 moves ahead due to performance
    if max_depth >= 2:
        for move in board.legal_moves:
            board.push(move)
            # After opponent's move, can we checkmate?
            for move2 in board.legal_moves:
                board.push(move2)
                if board.is_checkmate():
                    # Found mate in 2
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

def get_best_moves(board, top_n=3):
    """Get the top N best moves with their evaluations"""
    moves_with_scores = []

    for move in board.legal_moves:
        board.push(move)
        # Negate score because we're evaluating from opponent's perspective
        score = -evaluate_position(board)
        board.pop()
        moves_with_scores.append({
            'move': move.uci(),
            'score': score,
            'san': board.san(move)
        })

    # Sort by score descending
    moves_with_scores.sort(key=lambda x: x['score'], reverse=True)
    return moves_with_scores[:top_n]

def choose_best_move(board):
    """Choose the best move with some positional understanding"""
    best_moves = get_best_moves(board, top_n=1)
    if best_moves:
        return best_moves[0]
    return None

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
    """Get top 3 suggested moves with evaluations"""
    try:
        data = request.json
        fen = data.get('fen')
        top_n = data.get('top_n', 3)

        if not fen:
            return jsonify({'error': 'FEN string required'}), 400

        board = chess.Board(fen)

        if board.is_game_over():
            return jsonify({'error': 'Game is over'}), 400

        moves = get_best_moves(board, top_n=top_n)

        return jsonify({
            'moves': moves,
            'current_evaluation': round(evaluate_position(board), 3)
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

        if board.is_game_over():
            return jsonify({
                'error': 'Game is over',
                'result': board.result()
            }), 400

        move_data = choose_best_move(board)

        if not move_data:
            return jsonify({'error': 'No legal moves available'}), 400

        # Apply the move
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
        return jsonify({'error': str(e)}), 500

# ========== FRONTEND ROUTES (for Docker deployment) ==========

@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files or frontend for any route"""
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        # For React Router - serve index.html for all non-file routes
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("Starting Improved Chess AI Backend Server...")
    port = int(os.environ.get('PORT', 5001))
    print(f"Server running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', debug=False, port=port)
