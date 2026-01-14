from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import chess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ========== ENCODING ==========

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0
    return torch.from_numpy(tensor).flatten()

# ========== MODEL ==========

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Load the trained model
model = ChessNet()
try:
    model.load_state_dict(torch.load("chess_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")

# ========== MOVE SELECTION ==========

def evaluate_position(board):
    """Evaluate the current position"""
    with torch.no_grad():
        x = board_to_tensor(board)
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
    """Choose the best move"""
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
        evaluation = evaluate_position(board)

        return jsonify({
            'evaluation': round(evaluation, 3),
            'fen': fen
        })
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

if __name__ == '__main__':
    print("Starting Chess AI Backend Server...")
    print("Server running on http://localhost:5001")
    app.run(host='0.0.0.0', debug=True, port=5001)
