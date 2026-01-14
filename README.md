# Chess AI with React Frontend

🎮 **[Live Demo on Render.com](https://your-app-name.onrender.com)** (Deploy link will be added after deployment)

A modern chess application featuring a React TypeScript frontend with a PyTorch-powered AI opponent. The application includes live position evaluation, move suggestions with visual arrows, complete chess rules implementation, and a beautiful, responsive UI.

## Features

### Frontend Features
- **Modern React UI**: Built with TypeScript and React
- **Interactive Chessboard**: Drag-and-drop piece movement
- **Live Evaluation Bar**: Visual representation of position evaluation
- **Move Suggestions**: Top 3 suggested moves with arrows on the board
- **Move History**: Complete game notation with move numbers
- **Game Controls**: Reset, undo, and mode selection
- **Multiple Game Modes**:
  - Player vs AI
  - Player vs Player
- **Complete Chess Rules**:
  - Castling (kingside and queenside)
  - En passant
  - Pawn promotion
  - Check detection
  - Checkmate detection
  - Stalemate detection
  - Draw conditions

### Backend Features
- **PyTorch Neural Network**: Deep learning model for position evaluation
- **RESTful API**: Flask backend with CORS support
- **Multiple Endpoints**:
  - Position evaluation
  - Best move calculation
  - Move suggestions
  - AI move execution
- **Improved AI Model**: Enhanced with:
  - Deeper neural network
  - Positional evaluation
  - Material balance
  - Minimax with alpha-beta pruning

## Project Structure

```
chessbot/
├── chess-frontend/          # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx         # Main application component
│   │   └── App.css         # Styling
│   └── package.json
├── backend.py              # Flask API server
├── improved_ai.py          # Enhanced AI training script
├── ai.py                   # Original AI implementation
├── gui.py                  # Original Pygame GUI
├── game.py                 # Game logic utilities
├── chess_model.pth         # Trained model weights
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+
- pip
- Stockfish (for training new models)

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Train a new model:
```bash
python improved_ai.py
```

This will generate a better model with improved evaluation. You can adjust:
- `num_positions`: Number of training positions (default 500)
- `epochs`: Training epochs (default 30)
- `depth`: Stockfish analysis depth (default 18)

3. Start the Flask backend:
```bash
python backend.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd chess-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will open automatically at `http://localhost:3000`

## Usage

### Playing Against the AI

1. Make sure both backend and frontend are running
2. Open `http://localhost:3000` in your browser
3. Select "Player vs AI" mode
4. Choose your color (White or Black)
5. Make moves by dragging and dropping pieces
6. The AI will automatically respond after your move

### Features Guide

#### Position Evaluation Bar
- Shows the current position evaluation
- Green: White is winning
- Red: Black is winning
- Yellow: Equal position
- Updates in real-time after each move

#### Move Suggestions
- Top 3 best moves are displayed with:
  - Move notation (e.g., "Nf3")
  - Evaluation score
  - Visual arrows on the board
- Gold border: Best move
- Silver border: Second best
- Bronze border: Third best

#### Game Controls
- **New Game**: Reset the board to starting position
- **Undo Move**: Take back the last move
- **Mode**: Switch between Player vs AI and Player vs Player
- **Play as**: Choose your color (in AI mode)

#### Move History
- Shows all moves in standard algebraic notation
- Numbered by move pairs (1. e4 e5, 2. Nf3...)
- Scrollable list for long games

## API Endpoints

### POST `/api/evaluate`
Evaluate a chess position.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response:**
```json
{
  "evaluation": 0.023,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

### POST `/api/best-move`
Get the best move for a position.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response:**
```json
{
  "move": "e2e4",
  "san": "e4",
  "evaluation": 0.156
}
```

### POST `/api/suggested-moves`
Get top N suggested moves.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "top_n": 3
}
```

**Response:**
```json
{
  "moves": [
    {"move": "e2e4", "san": "e4", "score": 0.156},
    {"move": "d2d4", "san": "d4", "score": 0.142},
    {"move": "g1f3", "san": "Nf3", "score": 0.128}
  ],
  "current_evaluation": 0.023
}
```

### POST `/api/make-ai-move`
Make an AI move and get the new position.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
}
```

**Response:**
```json
{
  "move": "e7e5",
  "san": "e5",
  "new_fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
  "evaluation": -0.012,
  "is_game_over": false,
  "result": null
}
```

## Improving the AI

### Training a Better Model

The `improved_ai.py` script includes several enhancements:

1. **Enhanced Board Encoding**:
   - 12 channels for pieces
   - Castling rights encoding
   - En passant encoding
   - Turn indicator

2. **Deeper Neural Network**:
   - 5 layers vs 3 layers
   - Batch normalization
   - Dropout for regularization

3. **Better Training Data**:
   - More diverse positions
   - Higher Stockfish depth (18 vs 15)
   - Combined material + engine evaluation

4. **Improved Move Selection**:
   - Minimax with alpha-beta pruning
   - Configurable search depth

### Training Configuration

Edit `improved_ai.py` to customize:

```python
# More positions = better model (slower training)
X, y = generate_data_improved(num_positions=2000)

# More epochs = better convergence (slower training)
train_improved(model, X, y, epochs=50, lr=1e-3)

# Deeper search = stronger play (slower moves)
move, value = choose_best_move_improved(model, board, depth=3)
```

### Using the Improved Model

1. Train the improved model:
```bash
python improved_ai.py
```

2. Update `backend.py` to use the improved model:
```python
# Replace ChessNet with ImprovedChessNet
from improved_ai import ImprovedChessNet, board_to_tensor

model = ImprovedChessNet()
model.load_state_dict(torch.load("chess_model_best.pth"))
```

## Technologies Used

### Frontend
- React 18
- TypeScript
- chess.js (Chess logic)
- react-chessboard (Board UI)
- axios (HTTP client)

### Backend
- Flask (Web framework)
- Flask-CORS (Cross-origin requests)
- PyTorch (Neural network)
- python-chess (Chess engine)
- NumPy (Numerical computing)

## Performance Optimization

### Frontend
- React hooks for efficient re-renders
- Memoized callbacks to prevent unnecessary API calls
- Optimistic UI updates

### Backend
- Model evaluation caching
- Efficient tensor operations
- Batch processing support

### AI
- Alpha-beta pruning reduces search space
- Move ordering for better pruning
- Iterative deepening (can be added)

## Future Enhancements

- [ ] Opening book integration
- [ ] Endgame tablebase support
- [ ] Adjustable AI difficulty levels
- [ ] Game analysis and review mode
- [ ] Save/load games (PGN format)
- [ ] Multiplayer with WebSockets
- [ ] Mobile responsive improvements
- [ ] Theme customization
- [ ] Sound effects
- [ ] Clock/timer for timed games

## Troubleshooting

### Backend won't start
- Ensure Python dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is available
- Verify model file exists: `chess_model.pth`

### Frontend can't connect to backend
- Ensure backend is running on port 5000
- Check browser console for CORS errors
- Verify API_URL in App.tsx matches backend address

### AI moves are slow
- Reduce search depth in `choose_best_move`
- Use a smaller model
- Consider caching positions

### Model evaluation seems random
- Train with more positions
- Increase training epochs
- Use higher Stockfish depth for training data

## License

MIT License - See LICENSE file for details

## Credits

- Chess logic: chess.js and python-chess
- Board UI: react-chessboard
- AI training: Stockfish engine
- Neural network: PyTorch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
