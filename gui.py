import os
import tkinter as tk
import numpy as np
import pygame
from tensorflow.keras.models import load_model

my_model = True
if my_model:
    model = load_model("./models/my_model_sigmoid_binary_5000_up30_2moves.h5")
else:
    model = load_model("./models/20201213_202430.h5")
sound_dir = os.path.join(os.path.dirname(__file__), "sound")


def announce(message):
    file_path = os.path.join(sound_dir, message + ".mp3")
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
def on_click(event):
    # Do nothing if the game is over
    if game_over:
        return

    col = round((event.x - board_left) / cell_size)
    row = round((event.y - board_top) / cell_size)

    # Do nothing if piece is already placed
    if board[row][col] != 0:
        return

    # Draw the piece
    if 0 <= col < board_size and 0 <= row < board_size:
        canvas.create_oval(
            col * cell_size + board_left - oval_offset,
            row * cell_size + board_top - oval_offset,
            (col + 1) * cell_size + board_left - oval_offset,
            (row + 1) * cell_size + board_top - oval_offset,
            fill="black",
        )

    # Update the board state
    board[row][col] = 1

    # Check if the game is over
    check_game_over(player=1)
    
    # Check if board is full
    if check_board_full():
        return

    # Make a prediction with the trained model
    if my_model:
        input = -board
        output = model.predict(np.array([input]))
        while True:
            predicted_index = np.argmax(output)
            row = predicted_index // 20
            col = predicted_index % 20
            if board[row][col] == 0:
                break
            output[0][predicted_index] = 0
    else:
        input = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
        output = model.predict(input).squeeze()
        output = output.reshape((20, 20))
        row, col = np.unravel_index(np.argmax(output), output.shape)

    announce(chr(ord('a') + row))
    announce(str(col + 1))

    # Draw the predicted piece
    canvas.create_oval(
        col * cell_size + board_left - oval_offset,
        row * cell_size + board_top - oval_offset,
        (col + 1) * cell_size + board_left - oval_offset,
        (row + 1) * cell_size + board_top - oval_offset,
        fill="white",
    )

    # Update the board state
    board[row][col] = -1

    # Check if the game is over
    check_game_over(player=-1)
    
    # Check if board is full
    if check_board_full():
        return

def check_board_full():
    global game_over
    if np.count_nonzero(board) == board_size * board_size:
        game_over = True
        draw_message = "Draw!"
        canvas.create_text(
            canvas_width / 2,
            board_bottom + y_offset / 2,
            text=draw_message,
            font=("Helvetica", cell_size),
            fill="black",
        )
        return True
    return False


def check_game_over(player):
    global game_over
    won_player = 0

    # Check if the player won
    for row in range(board_size):
        for col in range(board_size):
            try:
                if (
                    board[row][col] == player
                    and board[row + 1][col] == player
                    and board[row + 2][col] == player
                    and board[row + 3][col] == player
                    and board[row + 4][col] == player
                ):
                    won_player = player
                    break
            except:
                pass
            try:
                if (
                    board[row][col] == player
                    and board[row][col + 1] == player
                    and board[row][col + 2] == player
                    and board[row][col + 3] == player
                    and board[row][col + 4] == player
                ):
                    won_player = player
                    break
            except:
                pass
            try:
                if (
                    board[row][col] == player
                    and board[row + 1][col + 1] == player
                    and board[row + 2][col + 2] == player
                    and board[row + 3][col + 3] == player
                    and board[row + 4][col + 4] == player
                ):
                    won_player = player
                    break
            except:
                pass
            try:
                if (
                    col >= 4
                    and board[row][col] == player
                    and board[row + 1][col - 1] == player
                    and board[row + 2][col - 2] == player
                    and board[row + 3][col - 3] == player
                    and board[row + 4][col - 4] == player
                ):
                    won_player = player
                    break
            except:
                pass
        if won_player != 0:
            break
    if won_player != 0:
        winner_message = "Black won!" if won_player == 1 else "White won!"
        if won_player == 1:
            announce("blackWin")
        else:
            announce("whiteWin")
        canvas.create_text(
            canvas_width / 2,
            board_bottom + y_offset / 2,
            text=winner_message,
            font=("Helvetica", cell_size),
            fill="black",
        )
        game_over = True


# Set up the main window
root = tk.Tk()
root.title("Gomoku")

# Set the cell size and board size
cell_size = 30
board_size = 20

# Oval offset
oval_offset = cell_size / 2

# Set the size of the canvas
canvas_width = board_size * cell_size * 1.5
canvas_height = board_size * cell_size * 1.5

# Create a canvas
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="grey")
canvas.pack()

# Calculate the offset to center the grid
x_offset = (canvas_width - board_size * cell_size) / 2
y_offset = (canvas_height - board_size * cell_size) / 2

# Calculate board boundaries
board_left = x_offset
board_right = (board_size - 1) * cell_size + x_offset
board_top = y_offset
board_bottom = (board_size - 1) * cell_size + y_offset

# Setup initial board state
board = np.zeros((board_size, board_size))
game_over = False

# Draw the grid lines
for i in range(board_size):
    # Vertical lines with alphabet index
    canvas.create_text(
        board_left - 20,
        board_top + i * cell_size,
        text=chr(65 + i), 
         # Adding 1 to make it 1-based index
    )
    canvas.create_text(
        board_left + i * cell_size - 10,
        board_top - 10,
        
         text=str(i + 1),  # Using ASCII values to get alphabet index
    )

    canvas.create_line(
        board_left + i * cell_size, board_top, board_left + i * cell_size, board_bottom
    )  # vertical
    canvas.create_line(
        board_left, board_top + i * cell_size, board_right, board_top + i * cell_size
    )  # horizontal

# Bind the click event
canvas.bind("<Button-1>", on_click)

# Run the Tkinter event loop
root.mainloop()
