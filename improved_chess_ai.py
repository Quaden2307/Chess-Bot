import torch
import numpy as np
import chess
import chess.engine
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json

# ========== ENHANCED ENCODING ==========

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

    # Pawn structure - doubled pawns, isolated pawns
    white_pawn_files = [chess.square_file(sq) for sq, p in board.piece_map().items()
                        if p.piece_type == chess.PAWN and p.color == chess.WHITE]
    black_pawn_files = [chess.square_file(sq) for sq, p in board.piece_map().items()
                        if p.piece_type == chess.PAWN and p.color == chess.BLACK]

    white_doubled = sum(1 for f in range(8) if white_pawn_files.count(f) > 1)
    black_doubled = sum(1 for f in range(8) if black_pawn_files.count(f) > 1)
    features.append((black_doubled - white_doubled) / 8.0)

    return torch.tensor(features, dtype=torch.float32)

# ========== IMPROVED MODEL ==========

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

# ========== OPENING BOOK ==========

COMMON_OPENINGS = [
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    # Spanish Opening
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    # Sicilian Defense
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    # Queen's Gambit
    ["d2d4", "d7d5", "c2c4"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    # Scandinavian
    ["e2e4", "d7d5", "e4d5"],
    # London System
    ["d2d4", "g8f6", "g1f3", "d7d5", "c1f4"],
]

def generate_diverse_data(num_positions=2000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """Generate diverse training data including opening positions, middlegame, and endgame"""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    X, y = [], []

    print(f"\n{'='*60}")
    print(f"GENERATING {num_positions} TRAINING POSITIONS")
    print(f"{'='*60}")
    print("This will take approximately 1.5-2 hours...")
    print("Progress will be shown below:\n")

    for i in tqdm(range(num_positions), desc="Generating positions", unit="pos"):
        board = chess.Board()

        # 30% of positions start from common openings
        if np.random.random() < 0.3 and COMMON_OPENINGS:
            opening_idx = np.random.randint(0, len(COMMON_OPENINGS))
            opening = COMMON_OPENINGS[opening_idx]
            num_opening_moves = np.random.randint(2, min(len(opening) + 1, 6))
            for move_uci in opening[:num_opening_moves]:
                try:
                    board.push_uci(move_uci)
                except:
                    break

        # Play random moves to reach diverse positions
        # Vary depth: early game (0-15 moves), middlegame (15-40), endgame (40+)
        phase = np.random.choice(['opening', 'middlegame', 'endgame'], p=[0.2, 0.5, 0.3])
        if phase == 'opening':
            num_moves = np.random.randint(0, 15)
        elif phase == 'middlegame':
            num_moves = np.random.randint(15, 40)
        else:
            num_moves = np.random.randint(40, 80)

        for _ in range(num_moves):
            if board.is_game_over():
                break

            # Mix of random and semi-reasonable moves
            if np.random.random() < 0.7:  # 70% reasonable moves
                legal_moves = list(board.legal_moves)
                # Prefer captures and checks
                captures = [m for m in legal_moves if board.is_capture(m)]
                checks = [m for m in legal_moves if board.gives_check(m)]
                preferred = captures + checks
                if preferred:
                    move = np.random.choice(preferred)
                else:
                    move = np.random.choice(legal_moves)
            else:  # 30% random moves for variety
                move = np.random.choice(list(board.legal_moves))

            board.push(move)

        # Get Stockfish evaluation at higher depth for better training
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=18))
            score = info["score"].white().score(mate_score=10000)
            if score is None:
                continue

            # Normalize evaluation
            eval_norm = max(min(score / 1000.0, 1.0), -1.0)
            X.append(board_to_enhanced_tensor(board))
            y.append(eval_norm)
        except:
            continue

    engine.quit()
    return torch.stack(X), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ========== TRAINING ==========

def train(model, X, y, epochs=50, lr=1e-3, batch_size=64):
    """Train with learning rate scheduling and early stopping"""
    dataset = torch.utils.data.TensorDataset(X, y)

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print(f"Batch size: {batch_size}, Max epochs: {epochs}")
    print(f"Early stopping patience: 10 epochs\n")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in train_progress:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix({'batch_loss': f'{loss.item():.4f}'})

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
        scheduler.step(avg_val_loss)

        # Progress indicator
        improvement = ""
        if avg_val_loss < best_val_loss:
            improvement = " ⭐ NEW BEST!"
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "chess_model_best.pth")
            print(f"✓ Checkpoint saved: chess_model_best.pth")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train: {avg_train_loss:.6f}  "
              f"Val: {avg_val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}  "
              f"Patience: {patience_counter}/10{improvement}")

        # Early stopping
        if patience_counter >= 10:
            print(f"\n{'='*60}")
            print("EARLY STOPPING - No improvement for 10 epochs")
            print(f"{'='*60}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    if patience_counter < 10:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Final best validation loss: {best_val_loss:.6f}")

# ========== MAIN ==========

if __name__ == "__main__":
    import time
    start_time = time.time()

    print("\n" + "=" * 60)
    print("IMPROVED CHESS AI TRAINING")
    print("=" * 60)
    print("Estimated total time: 2-2.5 hours")
    print("You can monitor progress throughout the entire process")
    print("=" * 60)

    # Generate diverse training data
    print("\n[STEP 1/3] GENERATING TRAINING DATA")
    print("=" * 60)
    data_start = time.time()
    X, y = generate_diverse_data(num_positions=3000)  # Increased for better learning
    data_time = time.time() - data_start
    print(f"\n✓ Data generation complete in {data_time/60:.1f} minutes")
    print(f"  Dataset shape: {X.shape}, Labels: {y.shape}")
    print(f"  Min evaluation: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}")

    # Train model
    print("\n[STEP 2/3] TRAINING MODEL")
    print("=" * 60)
    train_start = time.time()
    model = ImprovedChessNet()
    train(model, X, y, epochs=50, lr=1e-3, batch_size=64)
    train_time = time.time() - train_start
    print(f"\n✓ Training complete in {train_time/60:.1f} minutes")

    # Save final model
    print("\n[STEP 3/3] SAVING MODEL")
    print("=" * 60)
    torch.save(model.state_dict(), "chess_model_improved.pth")
    # Also load and save in eval mode for inference
    model.load_state_dict(torch.load("chess_model_best.pth"))
    model.eval()
    torch.save(model.state_dict(), "chess_model_improved.pth")

    total_time = time.time() - start_time
    print(f"✓ Model saved as 'chess_model_improved.pth'")
    print(f"✓ Best checkpoint saved as 'chess_model_best.pth'")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Data generation: {data_time/60:.1f} min")
    print(f"  Training: {train_time/60:.1f} min")
    print(f"Training data: {len(X)} positions")
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Stop your current backend: lsof -ti:5001 | xargs kill -9")
    print("2. Start improved backend: python backend_improved.py")
    print("3. Play against the improved AI in your browser!")
    print("=" * 60 + "\n")
