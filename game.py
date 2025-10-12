import chess
import pygame

pygame.init()

chessboard = chess.Board()


def get_mouse_position():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_x = mouse_x - 40  # Adjust for the board surface offset
    mouse_y = mouse_y - 20  # Adjust for the board surface offset
    if mouse_x < 0 or mouse_y < 0 or mouse_x >= 400 or mouse_y >= 400:
        return None, None  # Mouse is outside the board area
    else:
        return mouse_x, mouse_y

def get_starting_piece_position(mouse_x, mouse_y, x_index, y_index, chessboard):
    if mouse_x is None or mouse_y is None:
        return None, None, None, None
    x_index = mouse_x // 50
    y_index = 7 - (mouse_y // 50)  # Invert y-axis for chess board coordinates
    start_square = chess.square(x_index, y_index)
    selected_piece = chessboard.piece_at(chess.square(x_index, y_index))
    return x_index, y_index, start_square, selected_piece
  


def move_piece(start_square, end_x, end_y):
    
    if end_x is None or end_y is None:
        return False
    end_x = end_x // 50
    end_y = 7 - (end_y //50)
    end_square = chess.square(end_x, end_y)
    move = chess.Move(start_square, end_square)

    back_end_piece = chessboard.piece_at(start_square)
    if back_end_piece is None:
        return False
    
    if back_end_piece.piece_type == chess.PAWN:
        if (back_end_piece.color == chess.WHITE and chess.square_rank(end_square) == 7) or \
            (back_end_piece.color == chess.BLACK and chess.square_rank(end_square) == 0):
            print("Pawn Promotion")
            move = chess.Move(start_square, end_square, promotion = chess.QUEEN)


    if move in chessboard.legal_moves:
        chessboard.push(move)
        

        return end_square
    return False



        

            
        
        
                
                




