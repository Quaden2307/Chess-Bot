#!/usr/bin/env python3
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
from train_advanced_model import board_to_advanced_tensor, AdvancedChessNet, OPENING_POSITIONS

print("\n" + "="*70)
print("FAST CHESS AI TRAINING")
print("Training with 8,000 positions - optimized for speed and quality")
print("="*70)

def generate_fast_training_data(num_positions=8000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """Generate training data quickly with good quality"""
    X = []
    y = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    print(f"\nGenerating {num_positions} training positions...")

    for i in tqdm(range(num_positions)):
        # Start from random opening
        opening_fen = random.choice(OPENING_POSITIONS)
        board = chess.Board(opening_fen)

        # Random number of moves (0-35)
        num_moves = random.randint(0, 35)

        for _ in range(num_moves):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # 30% best moves, 70% random (creates diverse but reasonable positions)
            if random.random() < 0.3:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.01, depth=10))
                    move = result.move
                except:
                    move = random.choice(legal_moves)
            else:
                move = random.choice(legal_moves)

            board.push(move)

        # Skip if game is over
        if board.is_game_over():
            continue

        # Get Stockfish evaluation (depth 15 for speed)
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=15))
            score = info["score"].relative

            if score.is_mate():
                mate_in = score.mate()
                eval_value = 10.0 if mate_in > 0 else -10.0
            else:
                # Convert centipawn to normalized value
                eval_value = np.tanh(score.score() / 400.0)

            # Encode position
            tensor = board_to_advanced_tensor(board)
            X.append(tensor)
            y.append(eval_value)

        except Exception as e:
            continue

    engine.quit()

    print(f"Successfully generated {len(X)} positions")
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

def train_fast(model, X, y, epochs=50, batch_size=128):
    """Train efficiently"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    # Split into train/validation
    split_idx = int(0.9 * len(X))
    indices = torch.randperm(len(X))

    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_val = X[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    print(f"\nTraining on {len(X_train)} positions, validating on {len(X_val)}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}\n")

    best_val_loss = float('inf')
    patience = 12
    patience_counter = 0

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

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "chess_model_best.pth")
            patience_counter = 0
            print(f"  → Best! Val: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load("chess_model_best.pth"))
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")

    return model

# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_fast_training_data(num_positions=8000)
    print(f"\nEvaluation range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Mean evaluation: {y.mean():.3f}")

    # Create and train model
    model = AdvancedChessNet(input_size=783)
    model = train_fast(model, X, y, epochs=50, batch_size=128)

    # Save final model
    torch.save(model.state_dict(), "chess_model_improved.pth")
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("Models saved:")
    print("  - chess_model_improved.pth (final model)")
    print("  - chess_model_best.pth (best validation model)")
    print("="*70)
