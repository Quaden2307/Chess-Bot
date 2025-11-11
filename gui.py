import pygame
import os
from game import chessboard, get_mouse_position, get_starting_piece_position, move_piece
import chess

os.chdir(os.path.dirname(__file__))

pygame.init()

# Main Screen
gamescreen = pygame.display.set_mode((800, 600))

# Chess Board Screen
chessscreen = pygame.Surface((400, 400))
pygame.display.set_caption('Chess Bot')
font = pygame.font.Font(None, 36)

# Colours
white = (255, 255, 255)
black = (0, 0, 0)
brown = (139, 69, 19)
beige = (245, 222, 179)
light_green = (144, 238, 144)
dark_green = (0, 100, 0)

# Classes
class Piece:
    def __init__(self, name, color, x_position_index, y_position_index, id):
        self.name = name
        self.color = color
        self.x_position_index = x_position_index
        self.y_position_index = y_position_index
        self.id = id
        self.image = None
        self.id = id
        self.has_moved = False
    
    def draw(self, surface):
        current_color = self.color
        if self.color.collidepoint(pygame.mouse.get_pos()):
            current_color = self.hover_color
        pygame.draw.rect(surface, current_color, self.rect)
        text_surface = self.font.render(self.text, True, black)
        text_rectangle = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rectangle)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()

class Button:
    def __init__(self, x, y, width, height, color, hover_color, text):
        self.rect = pygame.Rect(x, y, width, height) 
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.font = pygame.font.Font(None, 36)

    def draw(self, surface):
            mous_pos = pygame.mouse.get_pos()
            current_color = self.color if self.rect.collidepoint(mous_pos) else self.hover_color
            pygame.draw.rect(surface, current_color, self.rect)
            text_surface = self.font.render(self.text, True, black)
            text_rectangle = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rectangle)
    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False



#Buttons
reset_board = Button(500, 100, 100, 50, brown, beige, "Reset")


# Pieces
def load_piece_images():
    for piece in white_pieces:
        if piece.name == "Pawn":
            piece.image = pygame.image.load('assets/wP.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)
        if piece.name == "Rook":
            piece.image = pygame.image.load('assets/wR.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)
        if piece.name == "Knight":
            piece.image = pygame.image.load('assets/wN.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)
        if piece.name == "Bishop":
            piece.image = pygame.image.load('assets/wB.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)
        if piece.name == "Queen":
            piece.image = pygame.image.load('assets/wQ.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)
        if piece.name == "King":
            piece.image = pygame.image.load('assets/wK.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            white_pieces_images.append(piece.image)

    for piece in black_pieces:
        if piece.name == "Pawn":
            piece.image = pygame.image.load('assets/bP.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)
        if piece.name == "Rook":
            piece.image = pygame.image.load('assets/bR.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)
        if piece.name == "Knight":
            piece.image = pygame.image.load('assets/bN.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)
        if piece.name == "Bishop":
            piece.image = pygame.image.load('assets/bB.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)
        if piece.name == "Queen":
            piece.image = pygame.image.load('assets/bQ.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)
        if piece.name == "King":
            piece.image = pygame.image.load('assets/bK.png')
            piece.image = pygame.transform.scale(piece.image, (50, 50))
            black_pieces_images.append(piece.image)

def original_state(): 
    global white_pieces, black_pieces, board, white_pieces_images, black_pieces_images
    white_pieces = [
        Piece("Rook", "White", 0, 0, "a1"),
        Piece("Knight", "White", 1, 0, "b1"),
        Piece("Bishop", "White", 2, 0, "c1"),
        Piece("Queen", "White", 3, 0, "d1"),
        Piece("King", "White", 4, 0, "e1"),
        Piece("Bishop", "White", 5, 0, "f1"),
        Piece("Knight", "White", 6, 0, "g1"),
        Piece("Rook", "White", 7, 0, "h1"),
        Piece("Pawn", "White", 0, 1, "a2"),
        Piece("Pawn", "White", 1, 1, "b2"),
        Piece("Pawn", "White", 2, 1, "c2"),
        Piece("Pawn", "White", 3, 1, "d2"),
        Piece("Pawn", "White", 4, 1, "e2"),
        Piece("Pawn", "White", 5, 1, "f2"),
        Piece("Pawn", "White", 6, 1, "g2"),
        Piece("Pawn", "White", 7, 1, "h2")

        
    ]
    white_pieces_images = []
    black_pieces = [
        Piece("Rook", "Black", 0, 7, "a8"),
        Piece("Knight", "Black", 1, 7, "b8"),
        Piece("Bishop", "Black", 2, 7, "c8"),
        Piece("Queen", "Black", 3, 7, "d8"),
        Piece("King", "Black", 4, 7, "e8"),
        Piece("Bishop", "Black", 5, 7, "f8"),
        Piece("Knight", "Black", 6, 7, "g8"),
        Piece("Rook", "Black", 7, 7, "h8"),
        Piece("Pawn", "Black", 0, 6, "a7"),
        Piece("Pawn", "Black", 1, 6, "b7"),
        Piece("Pawn", "Black", 2, 6, "c7"),
        Piece("Pawn", "Black", 3, 6, "d7"),
        Piece("Pawn", "Black", 4, 6, "e7"),
        Piece("Pawn", "Black", 5, 6, "f7"),
        Piece("Pawn", "Black", 6, 6, "g7"),
        Piece("Pawn", "Black", 7, 6, "h7")
    ]
    black_pieces_images = []
    
    # Board Matrix
    board = [[None for _ in range(8)] for _ in range(8)]
    for piece in white_pieces + black_pieces:
        board[piece.y_position_index][piece.x_position_index] = piece


    load_piece_images()
    
def initial_draw_pieces():
    for piece in white_pieces + black_pieces:
        if piece.image:
            x = piece.x_position_index * 50
            y = (7 - piece.y_position_index) * 50
            chessscreen.blit(piece.image, (x, y))

def clear_square(x, y):
    square_size = 50
    color = beige if (x + y) % 2 == 0 else brown
    rect = pygame.Rect(x * square_size, (7 - y) * square_size, square_size, square_size)
    pygame.draw.rect(chessscreen, color, rect)

def update_drawing_piece(start_square, end_square, chessboard):
    start_sqaure_x = chess.square_file(start_square)
    start_square_y = chess.square_rank(start_square)
    end_square_x = chess.square_file(end_square)
    end_square_y = chess.square_rank(end_square)

    piece = board[start_square_y][start_sqaure_x]

    if piece is not None:

        clear_square(start_sqaure_x, start_square_y)
        clear_square(end_square_x, end_square_y)

        if board[end_square_y][end_square_x] is not None:
            captured_piece = board[end_square_y][end_square_x]
            print(f"Captured {captured_piece.color}{captured_piece.name}")

            if captured_piece.color == "White":
                white_pieces.remove(captured_piece)
            else:
                black_pieces.remove(captured_piece)
            
        if piece.name == "Pawn" and (end_square_y == 0 or end_square_y == 7):
            piece.name = "Queen"
            if piece.color == "White":
                piece.image = pygame.image.load('assets/wQ.png')
            elif piece.color == "Black":
                piece.image = pygame.image.load('assets/bQ.png')

            piece.image = pygame.transform.scale(piece.image, (50, 50))
        
        #White Castling
        if (
            piece.name == "King"
            and not piece.has_moved
            and piece.color == "White"
            and end_square_y == 0
        ):
            # Kingside castling
            if end_square_x == 6 and not board[0][7].has_moved:
                rook_piece = board[0][7]
                if rook_piece:
                    board[0][5] = rook_piece
                    board[0][7] = None
                    rook_piece.x_position_index = 5
                    rook_piece.y_position_index = 0
                    rook_piece.has_moved = True
                    clear_square(7, 0)
                    x_pix = 5 * 50
                    y_pix = (7 - 0) * 50
                    chessscreen.blit(rook_piece.image, (x_pix, y_pix))
            # Queenside castling
            elif end_square_x == 2 and not board[0][0].has_moved:
                rook_piece = board[0][0]
                if rook_piece:
                    board[0][3] = rook_piece
                    board[0][0] = None
                    rook_piece.x_position_index = 3
                    rook_piece.y_position_index = 0
                    rook_piece.has_moved = True
                    clear_square(0, 0)
                    x_pix = 3 * 50
                    y_pix = (7 - 0) * 50
                    chessscreen.blit(rook_piece.image, (x_pix, y_pix))

        #Black Castling
        if (
            piece.name == "King"
            and not piece.has_moved
            and piece.color == "Black"
            and end_square_y == 7
        ):
            # Kingside (black)
            if end_square_x == 6 and not board[7][7].has_moved:
                rook_piece = board[7][7]
                if rook_piece:
                    board[7][5] = rook_piece
                    board[7][7] = None
                    clear_square(7, 7)

                    rook_piece.x_position_index = 5
                    rook_piece.y_position_index = 7
                    rook_piece.has_moved = True

                    x_pix = 5 * 50
                    y_pix = (7 - 7) * 50
                    chessscreen.blit(rook_piece.image, (x_pix, y_pix))

            # Queenside (black)
            elif end_square_x == 2 and not board[7][0].has_moved:
                rook_piece = board[7][0]
                if rook_piece:
                    board[7][3] = rook_piece
                    board[7][0] = None
                    clear_square(0, 7)

                    rook_piece.x_position_index = 3
                    rook_piece.y_position_index = 7
                    rook_piece.has_moved = True

                    x_pix = 3 * 50
                    y_pix = (7 - 7) * 50
                    chessscreen.blit(rook_piece.image, (x_pix, y_pix))

        
        
                
        board[end_square_y][end_square_x] = piece
        board[start_square_y][start_sqaure_x] = None

        piece.x_position_index = end_square_x
        piece.y_position_index = end_square_y

        piece.has_moved = True

        x_pix = piece.x_position_index * 50
        y_pix = (7 - piece.y_position_index) * 50
        chessscreen.blit(piece.image, (x_pix, y_pix))


# Board Setup
board_size = 8

# Axis Labels
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
letter_surface = pygame.Surface((400, 30))
letter_surface.fill(white)

numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
number_surface = pygame.Surface((20, 400))
number_surface.fill(white)

for number in numbers:
    index = numbers.index(number)
    number_text = font.render(number, True, black)
    number_surface.blit(number_text, (0, 400 - (index * 50) - 25))

for letter in letters:
    index = letters.index(letter)
    letter_text = font.render(letter, True, black)
    letter_surface.blit(letter_text, (index * 50 + 25, 0))


def display_board():
    square_size = 400 // board_size
    for row in range(board_size):
        for col in range(board_size):

            letter_axis = col*square_size
            number_axis = (7-row)*square_size
            
            color = beige if (row + col) % 2 == 0 else brown
            pygame.draw.rect(chessscreen, color, (letter_axis, number_axis, square_size, square_size))

def mouse_hover_square():
    global selected_square
    mouse_x, mouse_y = get_mouse_position()


    if mouse_x is None or mouse_y is None:
        return
  
    for row in range(8):
        for col in range(8):
            if x_index is None and y_index is None:
                if piece := board[row][col] is None:
                    continue
                else:
                    rect_x = col * 50
                    rect_y = (7 - row) * 50
                    rectangle = pygame.Rect(rect_x, rect_y, 50, 50)

                    if rectangle.collidepoint(mouse_x, mouse_y):
                        pygame.draw.rect(chessscreen, light_green, rectangle, 3)
                        return
                    
            elif x_index is not None and y_index is not None:
                if piece := board[row][col] is None:
                    continue      
                else:
                    rect_x = col * 50
                    rect_y = (7 - row) * 50
                    rectangle = pygame.Rect(rect_x, rect_y, 50, 50)
                    if rectangle.collidepoint(mouse_x, mouse_y):
                        pygame.draw.rect(chessscreen, dark_green, rectangle, 3)
                        return
    if selected_square is not None:
        col, row = selected_square
        rect_x = col * 50
        rect_y = (7 - row) * 50
        rectangle = pygame.Rect(rect_x, rect_y, 50, 50)
        pygame.draw.rect(chessscreen, dark_green, rectangle, 3)
                        

running = True
x_index = None
y_index = None
selected_square = None

original_state()

while running and not chessboard.is_game_over():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif reset_board.is_clicked(event):
                chessboard.reset()
                original_state()
                print("Board Reset")
                print(board)
                x_index = None
                y_index = None
                selected_square = None
    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if x_index is None and y_index is None:
                
                    mouse_x, mouse_y = get_mouse_position()
                    x_index, y_index, start_square, selected_piece = get_starting_piece_position(mouse_x, mouse_y, x_index, y_index, chessboard)

                    selected_square = (x_index, y_index)

                    print(f"Selected Piece: {selected_piece} at {start_square}")
                elif x_index is not None and y_index is not None:
                    
                    end_x, end_y = get_mouse_position()
                    if move_piece(start_square, end_x, end_y):
                        last_move = chessboard.peek()
                        
                        update_drawing_piece(last_move.from_square, last_move.to_square, chessboard)
            
                        print(board)
                    x_index = None
                    y_index = None
                    selected_square = None
        


        gamescreen.fill(white)
        gamescreen.blit(chessscreen, (40, 20))
        text = font.render('Chess Bot GUI', True, (0, 0, 0))

        gamescreen.blit(letter_surface, (35, 420))
        gamescreen.blit(number_surface, (20, 10))
        reset_board.draw(gamescreen)
        chessscreen.fill(white)
        display_board()
        mouse_hover_square()    
        initial_draw_pieces()
        
        pygame.display.flip()

pygame.quit()
