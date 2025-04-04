import os
import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Embedding, LayerNormalization, MultiHeadAttention
)
import random

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
CNN_MODEL_PATH = "/docker_files/connect4_cnn_2channel.h5"
TRANSFORMER_MODEL_PATH = "/docker_files/transformer_model_try2.h5"

# ✅ Define Learnable Positional Encoding Layer (🔥 FIXED)
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_length, embed_dim, **kwargs):
        super(LearnablePositionalEncoding, self).__init__(**kwargs)
        self.pos_embedding = Embedding(input_dim=seq_length, output_dim=embed_dim)

    def call(self, inputs):
        batch_size, seq_length, embed_dim = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        position_indices = tf.range(seq_length)
        position_embeddings = self.pos_embedding(position_indices)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return inputs + position_embeddings

# ✅ Define Transformer Block (No changes needed)
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)
    
custom_objects = {
    'LearnablePositionalEncoding': LearnablePositionalEncoding,
    'TransformerBlock': TransformerBlock
}

# ✅ Declare `models` globally
models = {"cnn": None, "transformer": None}
selected_model = "cnn"  # Default to CNN

# ✅ Load AI Models at Startup
if os.path.exists(CNN_MODEL_PATH):
    print(f"✅ CNN Model found at {CNN_MODEL_PATH}. Loading...")
    try:
        models["cnn"] = load_model(CNN_MODEL_PATH, custom_objects=custom_objects)
        print("✅ CNN Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")

if os.path.exists(TRANSFORMER_MODEL_PATH):
    print(f"✅ Transformer Model found at {TRANSFORMER_MODEL_PATH}. Loading...")
    try:
        models["transformer"] = load_model(TRANSFORMER_MODEL_PATH, custom_objects=custom_objects)
        print("✅ Transformer Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Transformer model: {e}")

# Connect to Anvil
ANVIL_UPLINK_KEY = "server_TRNS3LCHBYXJUDLSTS5JCIEY-A5EXJJ6GAJICDJBC"
anvil.server.connect(ANVIL_UPLINK_KEY)

# ── Helper Functions for Improved AI Logic (operating on a 6×7 integer board) ──

def update_board_int(board, piece, col):
    """
    Simulates dropping a piece (1 for human, 2 for AI) into the board.
    Returns a copy of the new board and the row index where the piece lands.
    """
    new_board = board.copy()
    for row in range(ROW_COUNT - 1, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = piece
            return new_board, row
    return None, None

def legal_moves_int(board):
    """Returns a list of column indices that are not full."""
    return [col for col in range(COLUMN_COUNT) if board[0, col] == 0]

def winning_moves_int(board, piece):
    """
    For the given piece, returns a list of moves (columns) that would result
    in an immediate win. If none, returns -1.
    """
    moves = []
    for col in legal_moves_int(board):
        sim_board, row = update_board_int(board, piece, col)
        if sim_board is not None and check_winner(sim_board, piece):
            moves.append(col)
    return moves if moves else -1

def nonlosing_moves_int(board, piece):
    """
    Returns a list of moves that do not allow the opponent to win immediately.
    """
    moves = []
    opponent = 1 if piece == 2 else 2
    for col in legal_moves_int(board):
        sim_board, row = update_board_int(board, piece, col)
        if sim_board is not None:
            opponent_can_win = False
            for opp_col in legal_moves_int(sim_board):
                sim_board2, row2 = update_board_int(sim_board, opponent, opp_col)
                if sim_board2 is not None and check_winner(sim_board2, opponent):
                    opponent_can_win = True
                    break
            if not opponent_can_win:
                moves.append(col)
    return moves

def improved_ai_move(board):
    """
    Improved AI move selection following the test_2.py logic:
      1. Check for immediate winning moves.
      2. Identify safe (nonlosing) moves.
      3. Use the model’s prediction (with heavy penalties for illegal/unsafe moves)
         to choose the best move.
    """
    piece = 2  # AI is represented by 2
    # Step 1: Check for an immediate win.
    win_moves = winning_moves_int(board, piece)
    if win_moves != -1:
        chosen_move = random.choice(win_moves)
        print(f"AI plays immediate winning move in column {chosen_move}")
        return chosen_move
    
    # Step 2: Identify safe moves.
    safe_moves = nonlosing_moves_int(board, piece)
    print(f"Safe moves for AI: {safe_moves}")
    
    # Step 3: Use the model’s prediction.
    # Convert the int board to a 6×7×2 representation.
    board_6x7x2 = convert_to_6x7x2(board)
    # Swap channels so that the AI’s pieces (2) are always in channel 0.
    board_swapped = np.flip(board_6x7x2, axis=-1)
    board_tensor = tf.convert_to_tensor(board_swapped[np.newaxis, ...], dtype=tf.float32)
    logits = models[selected_model].predict(board_tensor)[0]  # shape: (7,)
    
    # Penalize illegal moves and heavily penalize moves not in the safe set.
    legal = legal_moves_int(board)
    for col in range(COLUMN_COUNT):
        if col not in legal:
            logits[col] = -1e9
        elif col not in safe_moves:
            logits[col] -= 1e6
    
    best_move = int(np.argmax(logits))
    print(f"Model logits after penalization: {logits}, selected column {best_move}")
    return best_move

# ── Anvil Server Callables ──

@anvil.server.callable
def set_model(model_name):
    """Switches the AI model between CNN and Transformer."""
    global selected_model, models

    print(f"🔄 Attempting to switch to {model_name.upper()} model...")
    if model_name not in models or models[model_name] is None:
        print(f"❌ Model '{model_name}' is not available.")
        return {"error": f"Model '{model_name}' not available."}
    selected_model = model_name
    print(f"✅ Successfully switched to {model_name.upper()} model.")
    return {"success": f"Using {model_name.upper()} model for AI moves."}

@anvil.server.callable
def initialize_board():
    """Creates an empty Connect 4 board on the server."""
    return [[0] * 7 for _ in range(6)]  # 6×7 board filled with zeros

def check_winner(board, piece):
    """Checks if the given piece has a winning 4-in-a-row on the board."""
    board = np.array(board)
    # Horizontal check
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if np.all(board[r, c:c+4] == piece):
                return True
    # Vertical check
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if np.all(board[r:r+4, c] == piece):
                return True
    # Diagonal check (bottom-left to top-right)
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i, c+i] == piece for i in range(4)):
                return True
    # Diagonal check (top-left to bottom-right)
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r-i, c+i] == piece for i in range(4)):
                return True
    return False

@anvil.server.callable
def handle_move(board, col, turn, game_mode):
    """
    Handles a human player's move:
      - Drops the player's piece into the chosen column.
      - Checks for a win or a draw.
      - Returns the updated board and game status.
    """
    board = np.array(board, dtype=int) if isinstance(board, list) else board
    print(f"🟢 [handle_move] Received board type: {type(board)}")
    
    # Drop the piece into the chosen column.
    last_row = None
    for row in range(ROW_COUNT - 1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = turn
            last_row = row
            break
    else:
        return {"error": "Column is full! Choose another column."}
    
    if check_winner(board, turn):
        print(f"🏆 [handle_move] Player {turn} wins!")
        return {"board": board.tolist(), "game_over": True, "message": f"🎉 Player {turn} wins!", "last_row": last_row}
    
    if np.all(board != 0):
        print("🔄 [handle_move] Game is a draw.")
        return {"board": board.tolist(), "game_over": True, "message": "It's a draw!", "last_row": last_row}
    
    print("🔄 [handle_move] Returning board after player move.")
    return {"board": board.tolist(), "game_over": False, "last_row": last_row}

@anvil.server.callable
def ai_move(board):
    """
    Uses the selected AI model with improved heuristics to select the best move.
    Returns the chosen column.
    """
    global selected_model, models  
    print(f"🔍 [ai_move] Received board type: {type(board)}")
    board = np.array(board, dtype=int) if isinstance(board, list) else board
    chosen_move = improved_ai_move(board)
    print(f"✅ [ai_move] AI selected column: {chosen_move}")
    return chosen_move

@anvil.server.callable
def handle_ai_move(board):
    """
    AI selects a move and returns the column and row where its piece will land,
    along with the updated board and game status.
    """
    global selected_model
    board = np.array(board, dtype=int) if isinstance(board, list) else board
    print(f"🔍 [handle_ai_move] Received board type: {type(board)}")
    
    ai_col = ai_move(board)
    
    final_row = None
    for row in range(ROW_COUNT - 1, -1, -1):
        if board[row, ai_col] == 0:
            final_row = row
            break
    if final_row is None:
        print("⚠️ [handle_ai_move] AI tried to move in a full column!")
        return {"error": "AI tried to move in a full column!"}
    
    print(f"✅ [handle_ai_move] AI chose column {ai_col}, row {final_row}")
    board[final_row][ai_col] = 2  # AI is represented by 2
    
    if check_winner(board, 2):
        print("🏆 [handle_ai_move] AI wins!")
        return {"board": board.tolist(), "game_over": True, "message": "🎉 AI (O) wins!", "col": ai_col, "row": final_row}
    
    print("🔄 [handle_ai_move] Returning board after AI move.")
    return {"board": board.tolist(), "game_over": False, "col": ai_col, "row": final_row}

@anvil.server.callable
def force_ui_update():
    """Forces a UI update in Anvil by making a dummy server call."""
    return "UI updated"

@anvil.server.callable
def convert_to_6x7x2(board):
    """
    Converts a 6×7 board (with values 0, 1, or 2) into a 6×7×2 array
    where one channel represents player 1 and the other player 2.
    """
    print(f"🔍 [convert_to_6x7x2] Received board type: {type(board)}")
    board = np.array(board, dtype=int) if isinstance(board, list) else board
    board_6x7x2 = np.zeros((6, 7, 2), dtype=np.float32)
    board_6x7x2[:, :, 0] = (board == 1).astype(np.float32)  # Player 1
    board_6x7x2[:, :, 1] = (board == 2).astype(np.float32)  # Player 2
    return board_6x7x2

# Keep the connection alive
anvil.server.wait_forever()
