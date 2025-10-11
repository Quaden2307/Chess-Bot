import chess
import pygame
from gui import Piece 
from gui import white_pieces, black_pieces, draw_pieces, chessscreen, board_size, beige, brown


chessboard = chess.Board()
x_index = None
y_index = None

def get_mouse_position():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_x = mouse_x - 40  # Adjust for the board surface offset
    mouse_y = mouse_y - 20  # Adjust for the board surface offset
    if mouse_x < 0 or mouse_y < 0 or mouse_x >= 400 or mouse_y >= 400:
        return None, None  # Mouse is outside the board area
    else:
        return mouse_x, mouse_y

def get_starting_piece_position(mouse_x, mouse_y, x_index, y_index, chessboard):
    x_index = mouse_x // 50
    y_index = 7 - (mouse_y // 50)  # Invert y-axis for chess board coordinates
    start_square = chess.square(x_index, y_index)
    selected_piece = chessboard.piece_at(chess.square(x_index, y_index))
    return x_index, y_index, start_square, selected_piece


def move_piece(start_square):
    end_x, end_y = get_mouse_position()
    
    if end_x is None or end_y is None:
        return False
    end_x = end_x // 50
    end_y = 7 - (end_y //50)
    end_square = chess.square(end_x, end_y)
    move = chess.Move(start_square, end_square)

    if move in chessboard.legal_moves:
        chessboard.push(move)
        return True
    return False



while not chessboard.is_game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if x_index is None and y_index is None:
            
                mouse_x, mouse_y = get_mouse_position()
                x_index, y_index, start_square, selected_piece = get_starting_piece_position(mouse_x, mouse_y, x_index, y_index, chessboard)

            elif x_index is not None and y_index is not None:
                if move_piece(start_square):
                    x_index = None
                    y_index = None

        

            
        
        
                
                




