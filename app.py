# app.py
import streamlit as st
import chess
import chess.svg
import random
import numpy as np
from io import BytesIO
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set page configuration
st.set_page_config(
    page_title="Chess AI Game",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the neural network architecture for the chess AI
class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Function to convert board to feature vector
def board_to_feature(board):
    # Create a simple feature representation of the board
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6}
    
    feature = np.zeros(64, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            feature[i] = pieces.get(piece.symbol(), 0)
    
    # Add additional features (simplified)
    features_extended = np.zeros(768, dtype=np.float32)
    for i in range(64):
        features_extended[i*12 + int(feature[i]) + 6] = 1
    
    return features_extended

# Function to evaluate a move
def evaluate_move(model, board, move):
    # Make the move on a copy of the board
    test_board = board.copy()
    test_board.push(move)
    
    # Convert to feature vector
    features = board_to_feature(test_board)
    
    # Predict the value
    with torch.no_grad():
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        evaluation = model(features_tensor).item()
    
    return evaluation

# Function to get AI move
def get_ai_move(model, board, depth=2):
    best_move = None
    best_value = float('-inf') if board.turn else float('inf')
    
    # Simple minimax search (simplified for example)
    for move in board.legal_moves:
        # Evaluate the move
        move_value = evaluate_move(model, board, move)
        
        # If it's AI's turn (black), maximize the value
        if board.turn == chess.BLACK:
            if move_value > best_value:
                best_value = move_value
                best_move = move
        # If it's player's turn (white), minimize the value
        else:
            if move_value < best_value:
                best_value = move_value
                best_move = move
    
    # Fallback to random move if no best move found
    if best_move is None and board.legal_moves:
        best_move = random.choice(list(board.legal_moves))
    
    return best_move

# Function to render chess board
def render_board(board, flipped=False):
    svg = chess.svg.board(board=board, flipped=flipped, size=400)
    svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{svg_base64}"/>'

# Function to generate PGN-like move history
def generate_move_history(board):
    move_history = []
    temp_board = chess.Board()
    
    for move in board.move_stack:
        move_history.append(temp_board.san(move))
        temp_board.push(move)
    
    # Format moves in PGN style
    pgn_moves = []
    for i in range(0, len(move_history), 2):
        move_pair = []
        if i < len(move_history):
            move_pair.append(f"{i//2 + 1}. {move_history[i]}")
        if i + 1 < len(move_history):
            move_pair.append(move_history[i + 1])
        pgn_moves.append(" ".join(move_pair))
    
    return " ".join(pgn_moves)

# Initialize or load the model
@st.cache_resource
def load_model():
    model = ChessAI()
    # In a real application, you would load pre-trained weights here
    # model.load_state_dict(torch.load('chess_ai_model.pth'))
    return model

def main():
    # Load the model
    model = load_model()
    
    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'player_color' not in st.session_state:
        st.session_state.player_color = chess.WHITE
    if 'flip_board' not in st.session_state:
        st.session_state.flip_board = False
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    
    # Title and description
    st.title("♟️ Chess AI Game")
    st.markdown("""
    Play chess against an AI opponent. Select your color and make your move!
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Game Controls")
        
        # Player color selection
        player_color = st.radio("Play as:", ["White", "Black"])
        st.session_state.player_color = chess.WHITE if player_color == "White" else chess.BLACK
        st.session_state.flip_board = (player_color == "Black")
        
        # Difficulty level
        difficulty = st.slider("AI Difficulty", 1, 10, 5)
        
        # New game button
        if st.button("New Game"):
            st.session_state.board = chess.Board()
            st.session_state.game_over = False
            st.session_state.move_history = []
            st.rerun()
        
        # Display game information
        st.divider()
        st.subheader("Game Info")
        st.write(f"Current turn: {'White' if st.session_state.board.turn else 'Black'}")
        
        # Display move history
        if st.session_state.move_history:
            st.write("Move history:")
            for i, move in enumerate(st.session_state.move_history):
                st.write(f"{i+1}. {move}")
        
        # Display game status
        if st.session_state.board.is_check():
            st.warning("Check!")
        
        if st.session_state.game_over:
            if st.session_state.board.is_checkmate():
                winner = "Black" if st.session_state.board.turn else "White"
                st.error(f"Checkmate! {winner} wins!")
            elif st.session_state.board.is_stalemate():
                st.info("Stalemate! It's a draw!")
            elif st.session_state.board.is_insufficient_material():
                st.info("Draw by insufficient material!")
            elif st.session_state.board.is_seventyfive_moves():
                st.info("Draw by 75-move rule!")
            elif st.session_state.board.is_fivefold_repetition():
                st.info("Draw by fivefold repetition!")
    
    # Main game area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the chess board
        st.markdown(render_board(st.session_state.board, st.session_state.flip_board), unsafe_allow_html=True)
    
    with col2:
        # Move input
        st.subheader("Make a Move")
        
        if not st.session_state.game_over:
            # Display legal moves for player
            legal_moves = [move.uci() for move in st.session_state.board.legal_moves]
            
            # Input for move
            move_input = st.text_input("Enter your move (e.g., e2e4):", key="move_input")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Submit Move") and move_input:
                    if move_input in legal_moves:
                        move = chess.Move.from_uci(move_input)
                        st.session_state.board.push(move)
                        st.session_state.move_history.append(move_input)
                        
                        # Check if game is over after player's move
                        if st.session_state.board.is_game_over():
                            st.session_state.game_over = True
                            st.rerun()
                        
                        # AI's turn
                        with st.spinner("AI is thinking..."):
                            ai_move = get_ai_move(model, st.session_state.board)
                            if ai_move:
                                st.session_state.board.push(ai_move)
                                st.session_state.move_history.append(ai_move.uci())
                                
                                # Check if game is over after AI's move
                                if st.session_state.board.is_game_over():
                                    st.session_state.game_over = True
                        
                        st.rerun()
                    else:
                        st.error("Invalid move! Please enter a legal move.")
            
            with col_b:
                if st.button("Suggest Move"):
                    if st.session_state.board.legal_moves:
                        suggested_move = random.choice(list(st.session_state.board.legal_moves))
                        st.info(f"Suggested move: {suggested_move.uci()}")
            
            # Show legal moves
            st.write("Legal moves:")
            st.write(", ".join(legal_moves[:10]))
            if len(legal_moves) > 10:
                st.write(f"... and {len(legal_moves) - 10} more")
        else:
            st.info("Game over! Start a new game to play again.")
            
            if st.button("Analyze Game"):
                st.subheader("Game Analysis")
                st.write("Move history:", " → ".join(st.session_state.move_history))
    
    # Display FEN and PGN for advanced users
    with st.expander("Advanced Game Data"):
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("FEN (Forsyth–Edwards Notation):")
            st.code(st.session_state.board.fen())
        
        with col4:
            st.write("Move History (PGN-like format):")
            pgn_like = generate_move_history(st.session_state.board)
            st.text_area("PGN", pgn_like, height=100)

if __name__ == "__main__":
    main()