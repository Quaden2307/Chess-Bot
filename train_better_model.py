import torch
import numpy as np
import chess
import chess.engine
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import sys
import importlib

# Import encoding from train_advanced_model
sys.path.insert(0, '/Users/Quaden/Documents/chessbot')
import train_advanced_model
importlib.reload(train_advanced_model)
from train_advanced_model import board_to_advanced_tensor, AdvancedChessNet

# ========== IMPROVED DATA GENERATION ==========

# Much more diverse opening library
OPENING_POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

    # King's Pawn Openings (e4)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",  # Alekhine
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # Scandinavian
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Petrov
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Three Knights

    # Queen's Pawn Openings (d4)
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",  # Queen's Pawn
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # Indian Defense
    "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2",  # Englund Gambit
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",  # Indian Game

    # English Opening
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",

    # Reti Opening
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",

    # Italian Game positions
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",

    # Spanish/Ruy Lopez
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",

    # Queen's Gambit
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",

    # French Defense
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 3",

    # Caro-Kann
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 3",

    # King's Indian Defense
    "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq e3 0 5",

    # Nimzo-Indian
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",

    # Grunfeld Defense
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4",

    # Slav Defense
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq d6 0 3",

    # Pirc Defense
    "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3",
]

def generate_better_training_data(num_positions=15000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """Generate high-quality, diverse training positions"""
    X = []
    y = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    print(f"Generating {num_positions} high-quality training positions...")
    print("Using diverse openings and balanced position generation")

    for i in tqdm(range(num_positions)):
        # Start from random opening (more diverse)
        opening_fen = random.choice(OPENING_POSITIONS)
        board = chess.Board(opening_fen)

        # Variable depth for different game phases
        # 0-15 moves: opening, 16-30: middlegame, 31+: endgame
        phase = random.choice(['opening', 'middlegame', 'endgame'])

        if phase == 'opening':
            num_moves = random.randint(0, 10)
            stockfish_depth = 18
        elif phase == 'middlegame':
            num_moves = random.randint(8, 25)
            stockfish_depth = 20
        else:  # endgame
            num_moves = random.randint(20, 40)
            stockfish_depth = 22

        # Play moves to reach target position
        for move_count in range(num_moves):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Balance between random and quality moves
            # More quality moves in opening, more random in middle/endgame
            if phase == 'opening' and random.random() < 0.4:
                # 40% best moves in opening (learn good openings)
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.02, depth=12))
                    move = result.move
                except:
                    move = random.choice(legal_moves)
            elif random.random() < 0.2:
                # 20% best moves otherwise (create realistic positions)
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.01, depth=10))
                    move = result.move
                except:
                    move = random.choice(legal_moves)
            else:
                # Random moves to create variety
                move = random.choice(legal_moves)

            board.push(move)

        # Skip if game is over
        if board.is_game_over():
            continue

        # Get Stockfish evaluation with appropriate depth
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=stockfish_depth))
            score = info["score"].relative

            if score.is_mate():
                # Convert mate scores
                mate_in = score.mate()
                if mate_in > 0:
                    eval_value = min(10.0, 5.0 + mate_in * 0.5)
                else:
                    eval_value = max(-10.0, -5.0 + mate_in * 0.5)
            else:
                # Convert centipawn to normalized value with better scaling
                cp = score.score()
                # Use tanh for smooth scaling
                eval_value = np.tanh(cp / 400.0)

            # Encode position
            tensor = board_to_advanced_tensor(board)
            X.append(tensor)
            y.append(eval_value)

        except Exception as e:
            continue

    engine.quit()

    print(f"\nSuccessfully generated {len(X)} positions")
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

# ========== IMPROVED TRAINING ==========

def train_better_model(model, X, y, epochs=80, batch_size=128, lr=0.001):
    """Train with improved hyperparameters and techniques"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE

    # Split into train/validation
    split_idx = int(0.9 * len(X))
    indices = torch.randperm(len(X))

    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_val = X[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    model.train()

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print(f"\nTraining on {len(X_train)} positions, validating on {len(X_val)} positions")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        train_indices = torch.randperm(len(X_train))
        X_shuffled = X_train[train_indices]
        y_shuffled = y_train[train_indices]

        total_loss = 0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1))

        scheduler.step()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "chess_model_best.pth")
            patience_counter = 0
            print(f"  → New best model! Val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load("chess_model_best.pth"))
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

    return model

# ========== MAIN ==========

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED CHESS AI TRAINING")
    print("Training with 15,000 diverse positions from various openings")
    print("="*70)

    # Generate data
    X, y = generate_better_training_data(num_positions=15000)
    print(f"\nGenerated {len(X)} positions")
    print(f"Evaluation range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Mean evaluation: {y.mean():.3f}")

    # Create and train model
    model = AdvancedChessNet(input_size=783)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024*1024):.1f} MB")

    model = train_better_model(model, X, y, epochs=80, batch_size=128)

    # Save final model
    torch.save(model.state_dict(), "chess_model_improved.pth")
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("Models saved:")
    print("  - chess_model_improved.pth (final model)")
    print("  - chess_model_best.pth (best validation model)")
    print("="*70)
