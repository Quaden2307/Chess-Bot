import torch
import time
import numpy as np
import chess
import chess.engine
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ========== ENCODING ==========

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_tensor(board: chess.Board):
    """
    Enhanced board encoding with additional features:
    - 12 channels for piece positions
    - 1 channel for castling rights
    - 1 channel for en passant
    - 1 channel for turn indicator
    """
    tensor = np.zeros((8, 8, 15), dtype=np.float32)

    # Piece positions
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0

    # Castling rights (channel 12)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 12] += 0.25
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 12] += 0.25
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 12] += 0.25
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 12] += 0.25

    # En passant (channel 13)
    if board.ep_square:
        x = chess.square_file(board.ep_square)
        y = chess.square_rank(board.ep_square)
        tensor[y, x, 13] = 1.0

    # Turn indicator (channel 14)
    if board.turn == chess.WHITE:
        tensor[:, :, 14] = 1.0

    return torch.from_numpy(tensor).flatten()

def evaluate_board_material(board: chess.Board) -> float:
    """
    Calculate material advantage with positional bonuses
    """
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }

    # Positional bonus tables
    pawn_table = [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        [0.1,  0.1,  0.2,  0.3,  0.3,  0.2,  0.1,  0.1],
        [0.05, 0.05, 0.1,  0.25, 0.25, 0.1,  0.05, 0.05],
        [0.0,  0.0,  0.0,  0.2,  0.2,  0.0,  0.0,  0.0],
        [0.05, -0.05, -0.1, 0.0,  0.0,  -0.1, -0.05, 0.05],
        [0.05, 0.1,  0.1,  -0.2, -0.2, 0.1,  0.1,  0.05],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
    ]

    knight_table = [
        [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5],
        [-0.4, -0.2,  0.0,  0.0,  0.0,  0.0, -0.2, -0.4],
        [-0.3,  0.0,  0.1,  0.15, 0.15, 0.1,  0.0, -0.3],
        [-0.3,  0.05, 0.15, 0.2,  0.2,  0.15, 0.05, -0.3],
        [-0.3,  0.0,  0.15, 0.2,  0.2,  0.15, 0.0, -0.3],
        [-0.3,  0.05, 0.1,  0.15, 0.15, 0.1,  0.05, -0.3],
        [-0.4, -0.2,  0.0,  0.05, 0.05, 0.0, -0.2, -0.4],
        [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5]
    ]

    score = 0.0

    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)

        value = piece_values[piece.piece_type]

        # Add positional bonus
        if piece.piece_type == chess.PAWN:
            pos_bonus = pawn_table[y if piece.color == chess.WHITE else 7-y][x]
            value += pos_bonus
        elif piece.piece_type == chess.KNIGHT:
            pos_bonus = knight_table[y if piece.color == chess.WHITE else 7-y][x]
            value += pos_bonus

        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

    return score / 40.0  # Normalize to roughly -1 to 1

# ========== IMPROVED MODEL ==========

class ImprovedChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Deeper network with more capacity
        self.fc1 = nn.Linear(8 * 8 * 15, 512)
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
        return torch.tanh(self.fc5(x))

# ========== DATA GENERATION ==========

def generate_data_improved(num_positions=2000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """
    Generate training data with improved diversity
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    X, y = [], []

    for _ in tqdm(range(num_positions)):
        board = chess.Board()

        # Vary the number of random moves for diverse positions
        num_moves = np.random.choice([5, 10, 20, 30, 40, 50], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])

        for _ in range(num_moves):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)

        try:
            # Use higher depth for better evaluation
            info = engine.analyse(board, chess.engine.Limit(depth=18))
            score = info["score"].white().score(mate_score=10000)

            if score is None:
                continue

            # Normalize score with improved scaling
            eval_norm = max(min(score / 1000.0, 1.0), -1.0)

            # Also add material evaluation as a feature
            material_eval = evaluate_board_material(board)

            # Blend neural and material evaluation
            combined_eval = 0.7 * eval_norm + 0.3 * material_eval

            X.append(board_to_tensor(board))
            y.append(combined_eval)
        except:
            continue

    engine.quit()
    return torch.stack(X), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ========== TRAINING ==========

def train_improved(model, X, y, epochs=50, lr=1e-3, batch_size=64):
    """
    Train with learning rate scheduling and early stopping
    """
    dataset = torch.utils.data.TensorDataset(X, y)

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "chess_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered")
                break

# ========== MOVE SELECTION WITH ALPHA-BETA PRUNING ==========

def minimax_alpha_beta(board, model, depth, alpha, beta, maximizing):
    """
    Minimax with alpha-beta pruning for better move selection
    """
    if depth == 0 or board.is_game_over():
        with torch.no_grad():
            x = board_to_tensor(board)
            value = model(x).item()
        return value, None

    legal_moves = list(board.legal_moves)
    best_move = None

    if maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax_alpha_beta(board, model, depth - 1, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax_alpha_beta(board, model, depth - 1, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_move

def choose_best_move_improved(model, board, depth=2):
    """
    Choose the best move using minimax with alpha-beta pruning
    """
    maximizing = board.turn == chess.WHITE
    value, best_move = minimax_alpha_beta(board, model, depth, -float('inf'), float('inf'), maximizing)
    return best_move, value

# ========== MAIN ==========

if __name__ == "__main__":
    train_new_model = True

    if train_new_model:
        print("Generating improved training data...")
        X, y = generate_data_improved(num_positions=500)  # Increase for better results
        print(f"Data ready: {X.shape}, {y.shape}")

        model = ImprovedChessNet()
        print("Training improved model...")
        train_improved(model, X, y, epochs=30, lr=1e-3)

        print("Model saved as 'chess_model_best.pth'")
    else:
        model = ImprovedChessNet()
        model.load_state_dict(torch.load("chess_model_best.pth"))
        model.eval()
        print("Loaded trained model.")

    # Test the model
    board = chess.Board()
    print("\n=== Testing improved model ===\n")

    for i in range(10):
        if board.is_game_over():
            break

        move, value = choose_best_move_improved(model, board, depth=2)
        board.push(move)
        print(f"Move {i+1}: {move}")
        print(f"Evaluation: {value:.3f}")
        print(board)
        print("-" * 40)

    print("Game Over! Result:", board.result())
