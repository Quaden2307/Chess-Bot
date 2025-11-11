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

# ========== DATA GENERATION ==========

def generate_data(num_positions=1000): # -> increased num_positions for better training; may slow down execution
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    X, y = [], []

    for _ in tqdm(range(num_positions)):
        board = chess.Board()
        for _ in range(np.random.randint(0, 80)):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)

        info = engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].white().score(mate_score=1000)
        if score is None:
            continue
        
        eval_norm = max(min(score / 1000.0, 1.0), -1.0)
        X.append(board_to_tensor(board))
        y.append(eval_norm)

    engine.quit()
    return torch.stack(X), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ========== TRAINING ==========

def train(model, X, y, epochs=20, lr=1e-3, batch_size=32): # Train model with higher epochs for better performance (may slow down execution)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(loader):.6f}")

# ========== MOVE SELECTION ==========

def choose_best_move(model, board):
    best_move = None
    best_value = -float('inf')

    for move in board.legal_moves:
        board.push(move)
        x = board_to_tensor(board)
        value = model(x).item()
        board.pop()
        if value > best_value:
            best_value = value
            best_move = move

    return best_move, best_value

# ========== MAIN ==========

if __name__ == "__main__":
    # Train (or skip if already trained)
    train_new_model = True  # change to False if already have chess_model.pth

    if train_new_model:
        print("Generating training data...")
        X, y = generate_data(num_positions=200)  # smaller number for faster run
        print(f"Data ready: {X.shape}, {y.shape}")

        model = ChessNet()
        print("Training model...")
        train(model, X, y, epochs=5, lr=1e-3)

        torch.save(model.state_dict(), "chess_model.pth")
        print("Model saved as 'chess_model.pth'")

    # Skip Training Process
    else:
        model = ChessNet()
        model.load_state_dict(torch.load("chess_model.pth"))
        model.eval()
        print("Loaded trained model.")

    # Step 2: Let it play a self-game
    board = chess.Board()
    print("\n=== Starting self-play game ===\n")

    while not board.is_game_over():
        move, value = choose_best_move(model, board)
        board.push(move)
        print(board)
        time.sleep(2)  # pause for readability --> Must implement board visualization on gui
        print("Predicted eval:", round(value, 3))
        print("-" * 40)

    print("Game Over! Result:", board.result())


#Key Notes:

#Current model almost always ends in self-playing eval of 0.900 < eval < 1 
#Always advantages for white
#Aiming to implement openings for more variance in advantage
#More training data and higher epochs also cause games to be much longer