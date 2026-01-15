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

# Piece-square tables for positional evaluation
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

def board_to_advanced_tensor(board: chess.Board):
    """Advanced board encoding with tactical and positional features"""
    features = []

    # 1. Piece positions (12 channels * 64 squares = 768 features)
    piece_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        piece_tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0
    features.extend(piece_tensor.flatten().tolist())

    # 2. Material evaluation
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = (white_material - black_material) / 10000.0
    features.append(material_advantage)

    # 3. Piece-square tables evaluation (positional value)
    white_pos_value = 0
    black_pos_value = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                white_pos_value += PAWN_TABLE[square]
            else:
                black_pos_value += PAWN_TABLE[63 - square]
        elif piece.piece_type == chess.KNIGHT:
            if piece.color == chess.WHITE:
                white_pos_value += KNIGHT_TABLE[square]
            else:
                black_pos_value += KNIGHT_TABLE[63 - square]
    positional_advantage = (white_pos_value - black_pos_value) / 1000.0
    features.append(positional_advantage)

    # 4. Mobility (how many moves each side has)
    current_turn = board.turn
    white_mobility = 0
    black_mobility = 0

    if board.turn == chess.WHITE:
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
    else:
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK

    mobility_advantage = (white_mobility - black_mobility) / 50.0
    features.append(mobility_advantage)

    # 5. Center control (pieces in center squares)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    extended_center = [chess.C3, chess.D3, chess.E3, chess.F3,
                       chess.C4, chess.F4, chess.C5, chess.F5,
                       chess.C6, chess.D6, chess.E6, chess.F6]

    white_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    black_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
    center_control = (white_center - black_center) / 4.0
    features.append(center_control)

    white_ext_center = sum(1 for sq in extended_center if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    black_ext_center = sum(1 for sq in extended_center if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
    ext_center_control = (white_ext_center - black_ext_center) / 12.0
    features.append(ext_center_control)

    # 6. King safety (simplified for speed)
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    # Simple king safety: count attackers near king
    white_king_safety = 0
    black_king_safety = 0
    if white_king_sq is not None:
        white_king_safety = len(board.attackers(chess.WHITE, white_king_sq))
    if black_king_sq is not None:
        black_king_safety = len(board.attackers(chess.BLACK, black_king_sq))

    king_safety = (white_king_safety - black_king_safety) / 10.0
    features.append(king_safety)

    # 7. Castling rights
    white_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
    white_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
    black_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
    black_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)

    features.extend([
        1.0 if white_can_castle_kingside else 0.0,
        1.0 if white_can_castle_queenside else 0.0,
        1.0 if black_can_castle_kingside else 0.0,
        1.0 if black_can_castle_queenside else 0.0
    ])

    # 8. Check status
    in_check = 1.0 if board.is_check() else 0.0
    features.append(in_check)

    # 9. Attacked squares (tactical awareness)
    white_attacks = len(board.attacks(white_king_sq)) if white_king_sq else 0
    black_attacks = len(board.attacks(black_king_sq)) if black_king_sq else 0
    attack_pressure = (white_attacks - black_attacks) / 20.0
    features.append(attack_pressure)

    # 10. Pawn structure (optimized)
    white_pawn_files = []
    black_pawn_files = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                white_pawn_files.append(chess.square_file(sq))
            else:
                black_pawn_files.append(chess.square_file(sq))

    # Doubled pawns
    white_doubled = sum(1 for f in range(8) if white_pawn_files.count(f) > 1)
    black_doubled = sum(1 for f in range(8) if black_pawn_files.count(f) > 1)
    doubled_pawns = (black_doubled - white_doubled) / 8.0
    features.append(doubled_pawns)

    # 11. Game phase (endgame indicator)
    total_material = white_material + black_material
    game_phase = 1.0 - (total_material / 78000.0)  # 0 = opening, 1 = endgame
    features.append(game_phase)

    # 12. Turn indicator
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    features.append(turn)

    # Count features:
    # 768 (piece positions) + 1 (material) + 1 (positional) + 1 (mobility) + 2 (center control)
    # + 1 (king safety) + 4 (castling) + 1 (check) + 1 (attacks) + 1 (pawn structure)
    # + 1 (game phase) + 1 (turn) = 783 features
    return torch.tensor(features, dtype=torch.float32)

# ========== MODEL ==========

class AdvancedChessNet(nn.Module):
    def __init__(self, input_size=783):
        super().__init__()

        # Deeper network with residual connections
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        return x

# Load the trained model
model = AdvancedChessNet(input_size=783)
try:
    # Try to load improved model first
    model.load_state_dict(torch.load("chess_model_improved.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Advanced model loaded successfully!")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load("chess_model_best.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Best model loaded successfully!")
    except FileNotFoundError:
        print("Warning: No trained model found. Please run train_advanced_model.py first.")

# ========== MOVE SELECTION ==========

def detect_mate_in_n(board, max_depth=5):
    """
    Detect if there's a forced mate within max_depth moves.
    Returns integer: positive for white advantage, negative for black advantage, or None
    """
    try:
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
    except Exception as e:
        print(f"Error in detect_mate_in_n: {e}")
        return None

def evaluate_position(board):
    """Evaluate the current position"""
    try:
        # Check for mate first
        mate_in = detect_mate_in_n(board, max_depth=5)
        if mate_in is not None:
            # Return a large value for mate positions
            return 10.0 if mate_in > 0 else -10.0

        with torch.no_grad():
            x = board_to_advanced_tensor(board).unsqueeze(0)  # Add batch dimension
            value = model(x).item()
        return value
    except Exception as e:
        print(f"Error evaluating position: {e}")
        # Return 0 (equal position) if evaluation fails
        return 0.0

def get_best_moves(board, top_n=3):
    """Get the top N best moves with their evaluations"""
    try:
        moves_with_scores = []

        for move in board.legal_moves:
            try:
                board.push(move)
                # Negate score because we're evaluating from opponent's perspective
                score = -evaluate_position(board)
                board.pop()
                moves_with_scores.append({
                    'move': move.uci(),
                    'score': score,
                    'san': board.san(move)
                })
            except Exception as e:
                print(f"Error evaluating move {move}: {e}")
                board.pop()  # Make sure we pop even on error
                continue

        # Sort by score descending
        moves_with_scores.sort(key=lambda x: x['score'], reverse=True)
        return moves_with_scores[:top_n]
    except Exception as e:
        print(f"Error in get_best_moves: {e}")
        return []

def choose_best_move(board):
    """Choose the best move with some positional understanding"""
    try:
        best_moves = get_best_moves(board, top_n=1)
        if best_moves:
            return best_moves[0]
        return None
    except Exception as e:
        print(f"Error in choose_best_move: {e}")
        return None

# ========== API ENDPOINTS ==========

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    import sys
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'python_version': sys.version
    })

@app.route('/api/test-move', methods=['GET'])
def test_move():
    """Test endpoint to verify move generation works"""
    try:
        import random
        import traceback
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        return jsonify({
            'status': 'success',
            'move': move.uci(),
            'san': board.san(move),
            'legal_moves_count': len(legal_moves)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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

        # Try to get best move, fallback to random if fails
        try:
            move_data = choose_best_move(board)
            if not move_data:
                # Fallback to random move
                import random
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return jsonify({'error': 'No legal moves available'}), 400
                move = random.choice(legal_moves)
                move_data = {
                    'move': move.uci(),
                    'san': board.san(move),
                    'score': 0.0
                }
        except Exception as model_error:
            # If model fails, use random move
            print(f"Model error: {model_error}, using random move")
            import random
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return jsonify({'error': 'No legal moves available'}), 400
            move = random.choice(legal_moves)
            move_data = {
                'move': move.uci(),
                'san': board.san(move),
                'score': 0.0
            }

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
        print(f"Error in make_ai_move: {str(e)}")
        import traceback
        traceback.print_exc()
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
