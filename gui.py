import streamlit as st
import pygame
import os
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



class Piece:
    def __init__(self, name, color, x_position_index, y_position_index, id):
        self.name = name
        self.color = color
        self.x_position_index = x_position_index
        self.y_position_index = y_position_index
        self.id = id
        self.image = None
        self.id = id

# Pieces
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

print(board)

def draw_pieces():
    for piece in white_pieces + black_pieces:
        if piece.image:
            x = piece.x_position_index * 50
            y = (7 - piece.y_position_index) * 50
            chessscreen.blit(piece.image, (x, y))


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
    

def display_board():
    square_size = 400 // board_size
    for row in range(board_size):
        for col in range(board_size):

            letter_axis = col*square_size
            number_axis = (7-row)*square_size

            color = beige if (row + col) % 2 == 0 else brown
            pygame.draw.rect(chessscreen, color, (letter_axis, number_axis, square_size, square_size))


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False



    gamescreen.fill(white)
    gamescreen.blit(chessscreen, (40, 20))
    text = font.render('Chess Bot GUI', True, (0, 0, 0))

    gamescreen.blit(letter_surface, (35, 420))
    gamescreen.blit(number_surface, (20, 10))

    display_board()
    draw_pieces()

    pygame.display.flip()

pygame.quit()
