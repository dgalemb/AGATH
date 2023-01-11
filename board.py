import numpy as np
import random
import sys
import time


SIZE = 8

BLACK = 1
WHITE = -1
EMPTY = 0

PROFONDEUR = 42

pass_move = SIZE * SIZE

# directions, dans le sens direct, en commençant par la droite
directions = (1, 1 - SIZE, -SIZE, -1 - SIZE, -1, -1 + SIZE, SIZE, 1 + SIZE)

# renvoie le signe de x
def sign(x):
    return int(x > 0) - int(x < 0)

# limites d'itération en fonction de la direction, dans le sens direct, en commençant par la droite
limits = [[(SIZE - 1 - j, min(i, SIZE - 1 - j), i, min(i, j), j, min(SIZE - 1 - i, j), SIZE - 1 - i, min(SIZE - 1 - i, SIZE - 1 - j))
            for j in range(SIZE)] for i in range(SIZE)]

# indique s'il y a des pions à retourner dans une direction donnée
# player est la couleur de celui qui joue
# limit est la limite à ne pas dépasser afin de ne pas sortir du plateau
def check1D(board, position, player, direction, limit):
    k = 1
    position += direction
    while k <= limit and board[position] == -player:
        position += direction
        k = k + 1

    # Il faut qu'il y ait eu au moins un pion adverse
    return (k > 1 and k <= limit and board[position] == player)

# Calcule le suivant dans une direction donnée 
# player est la couleur de celui qui joue
# limit est la limite à ne pas dépasser afin de ne pas sortir du plateau
# On suppose qu'on doit effectivement retourner des pions dans cette direction
# i. e. check1D a été appelé avant
def play1D(board, position, player, direction, limit):
    k = 1
    position += direction
    while k <= limit and board[position] == -player:
        board[position] = player
        position += direction
        k = k + 1

# Calcule le successeur en place, obtenu en ajoutant un pion de player
# sur la position donnée
def play_(board, position, player):
    if position != pass_move:
        # La position en 2D
        i = position // SIZE   
        j = position % SIZE

        # La case jouée
        board[position] = player

        # Retourne les pions dans toutes les directions
        for direction, limit in zip(directions, limits[i][j]):
            if check1D(board, position, player, direction, limit):
                play1D(board, position, player, direction, limit) 

# Successeur avec copie
def play(board, position, player):
    r = np.copy(board)
    play_(r, position, player)

    return r

# Intialise le plateau
def init_board():
    # Crée et initialise tout à vide
    b = np.array([EMPTY for k in range(SIZE*SIZE)])

    # Place les quatre premiers pions
    b[3 * SIZE + 3] = WHITE
    b[4 * SIZE + 4] = WHITE
    b[3 * SIZE + 4] = BLACK
    b[4 * SIZE + 3] = BLACK

    return b

# Affiche le plateau
def print_board(board):
    # Numéros de colonne
    print("  ", end='')
    for i in range(SIZE):
        print(f'{i + 1}', end=' ')
    print()

    for i in range(SIZE):
        # Affiche le numéro de la ligne
        print(f'{SIZE - i}', end=' ')
        for j in range(SIZE):
            # Contenu de la case
            p = board[i * SIZE + j]

            if p == EMPTY:
                print(".", end = ' ')
            elif p == BLACK:
                print("O", end = ' ')
            elif p == WHITE:
                print("X", end = ' ')
            else:
                print("?", end = ' ')

        # Réaffiche le numéro de la ligne
        print(f'{SIZE - i}')

    # Numéros de colonne à nouveau
    print("  ", end='')
    for i in range(SIZE):
        print(f'{i + 1}', end=' ')
    print()
    print()

# Trouve les coups légaux pour le joueur player
def legal_moves(board, player):
    L = []
    for p in range(SIZE * SIZE):
        # Il faut au moins que la case soit vide
        if board[p] == EMPTY:
            # On cherche au moins une direction dans laquelle c'est valide
            i = p // SIZE
            j = p % SIZE
            
            lims = limits[i][j]

            valid = False
            n = 0
            while n < 8 and not valid:
                valid = check1D(board, p, player, directions[n], lims[n]) 
                n = n + 1

            # Valide, on enregistre
            if valid:
                L.append(p)

    # Pas de coup à jouer: il faut passer
    if not L:
        L.append(pass_move)

    return L

# Vérifie si la partie est finie
# Similaire à legal_moves mais on teste les deux joueurs
# pour chaque case
def terminal(board):
    r = True
    p = 0
    while p < SIZE * SIZE and r:
        # Il faut au moins que la case soit vide
        if board[p] == EMPTY:
            # On cherche au moins une direction dans laquelle c'est valide
            i = p // SIZE
            j = p % SIZE

            lims = limits[i][j]

            n = 0
            while n < 8 and r:
                # check1D nous dit si on peut jouer dans cette direction
                # si on peut alors la position n'est pas terminale
                r = not (check1D(board, p, BLACK, directions[n], lims[n])
                    or check1D(board, p, WHITE, directions[n], lims[n])) 
                n = n + 1

        p = p +1

    return r

# Trouve le joueur qui a le plus de pions
def winner(board):
    score = 0
    for i in range(SIZE * SIZE):
        score += board[i]

    return sign(score)

def human(board, player):

    L = legal_moves(board, player)
    L2 = list(map(lambda x: (SIZE - x//SIZE, x%SIZE + 1), L))
    flag = True

    while(flag):

        desired_move = tuple(int(n) for n in input('Please type  your desired move (the coordinates of the end case), as X Y (i.e. 6, 7 === 6 7)').split(" "))

        if desired_move in L2:    
            flag = False

        else:
            print("Desired move not possible. Retry.")

    return L[L2.index(desired_move)]

def alea(board, player):

    L = legal_moves(board, player)
    aleatory_move = random.choice(L)

    return aleatory_move

# Player vs Player or Random vs Player
def main1():

    b = init_board()
    rand = int(input("Type 1 for a match between humans, 2 for a match between two random players"))

    if rand == 1:
        print("Player BLACK begins:")
        print_board(b)

        while(not terminal(b)):
            playy = human(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            playy = human(b, WHITE)
            b = play(b, playy, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

    else:
        print_board(b)

        while(not terminal(b)):
            playy = alea(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            playy = alea(b, WHITE)
            b = play(b, playy, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')


#########

def evaluate(board):

    BLACKs = 0
    WHITEs = 0

    line1 = [10, -1, 5, 5, 5, 5, -1, 10]
    line2 = [-1, -3, 1, 1, 1, 1, -3, -1]
    line3 = [5, -3, 1, 1, 1, 1, -3, 5]

    weights = line1 + line2 + line3 + line3 + line3 + line3 + line2 + line1

    for i in range(SIZE * SIZE):

        if board[i] == BLACK:
            BLACKs += weights[i]

        if board[i] == WHITE:
            WHITEs += weights[i]

    if (BLACKs + WHITEs) == 0: positionement = 0
    else:
        positionement = 100 * (BLACKs - WHITEs) / (BLACKs + WHITEs)


    BLACK_legal_moves = len(legal_moves(board, BLACK))
    WHITE_legal_moves = len(legal_moves(board, WHITE))

    moves = 100 * (BLACK_legal_moves - WHITE_legal_moves) / (BLACK_legal_moves + WHITE_legal_moves)


    return 0.5*positionement + 0.5*moves

def minimax(board, player, depth):

    if depth == 0:

        return evaluate(board)

    if terminal(board):
    
        if winner(board) == BLACK:
            return 100

        else:
            return -100
        
    if player == BLACK:

        max_score = -sys.maxsize
        L = legal_moves(board, BLACK)

        if len(L) == 0:

            max_score = minimax(board, WHITE, depth)

        else:

            for move in L:
                board1 = play(board, move, BLACK)

                score = minimax(board1, WHITE, depth - 1)
                max_score = max(max_score, score)

        return max_score

    else:

        min_score = sys.maxsize
        L = legal_moves(board, WHITE)

        if len(L) == 0:

            min_score = minimax(board, BLACK, depth)

        else:

            for move in L:
                board1 = play(board, move, WHITE)

                score = minimax(board1, BLACK, depth - 1)
                min_score = min(min_score, score)

        return min_score

# Player vs Minimax or Random vs Minimax
def main2(PROFONDEUR):

    b = init_board()
    rand = int(input("Type 1 for a match between human and minimax, 2 for a match between random player and minimax"))

    if rand == 1:
        print("Player BLACK (human) begins:")
        print_board(b)

        while(not terminal(b)):
            playy = human(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = minimax(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the minimax move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

    else:
        print_board(b)

        while(not terminal(b)):
            playy = alea(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = minimax(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the minimax move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

def minimax_tr(board, player, depth, table = {}):

    if depth == 0:

        return evaluate(board)

    if terminal(board):
    
        if winner(board) == BLACK:
            return 100

        else:
            return -100
        
    if player == BLACK:

        max_score = -sys.maxsize
        L = legal_moves(board, BLACK)

        if len(L) == 0:

            max_score = minimax_tr(board, WHITE, depth)

        else:

            for move in L:
                board1 = play(board, move, BLACK)
                if (tuple(board1), WHITE) in table:
                    if table[(tuple(board1), WHITE)][0] >= depth:
                        score = table[(tuple(board1), WHITE)][1]

                    else:
                        score = minimax_tr(board1, WHITE, depth - 1)

                else:
                        score = minimax_tr(board1, WHITE, depth - 1)
                        table[(tuple(board1), WHITE)] = (depth, max_score, move)


                max_score = max(max_score, score)

        return max_score

    else:

        min_score = sys.maxsize
        L = legal_moves(board, WHITE)

        if len(L) == 0:

            min_score = minimax_tr(board, BLACK, depth)

        else:

            for move in L:
                board1 = play(board, move, WHITE)
                if (tuple(board1), BLACK) in table:
                    if table[(tuple(board1), BLACK)][0] >= depth:
                        score = table[(tuple(board1), BLACK)][1]

                    else:
                        score = minimax_tr(board1, BLACK, depth - 1)

                else:
                        score = minimax_tr(board1, BLACK, depth - 1)
                        table[(tuple(board1), BLACK)] = (depth, min_score, move)


                min_score = min(min_score, score)

        return min_score

# Player vs Minimax_tr or Random vs Minimax_tr
def main3(PROFONDEUR):

    b = init_board()
    rand = int(input("Type 1 for a match between human and minimax, 2 for a match between random player and minimax"))

    if rand == 1:
        print("Player BLACK (human) begins:")
        print_board(b)

        while(not terminal(b)):
            playy = human(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = minimax_tr(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the minimax move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

    else:
        print_board(b)

        while(not terminal(b)):
            playy = alea(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = minimax_tr(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the minimax move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

def alphabeta(board, player, depth, alpha = -sys.maxsize, beta = sys.maxsize):

    if depth == 0:

        return evaluate(board)

    if terminal(board):
    
        if winner(board) == BLACK:
            return 100

        else:
            return -100
        
    if player == BLACK:

        max_score = -sys.maxsize
        L = legal_moves(board, BLACK)

        if len(L) == 0:

            max_score = alphabeta(board, WHITE, depth, alpha, beta)

        else:

            for move in L:
                board1 = play(board, move, BLACK)

                score = alphabeta(board1, WHITE, depth, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)

                if beta <= alpha:
                    break

        return max_score

    else:

        min_score = sys.maxsize
        L = legal_moves(board, WHITE)

        if len(L) == 0:

            min_score = alphabeta(board, BLACK, depth, alpha, beta)

        else:

            for move in L:
                board1 = play(board, move, WHITE)

                score = alphabeta(board1, BLACK, depth - 1, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)

                if beta <= alpha:
                    break

        return min_score

def main4(PROFONDEUR):

    b = init_board()
    rand = int(input("Type 1 for a match between human and alphabeta, 2 for a match between random player and alphabeta"))

    if rand == 1:
        print("Player BLACK (human) begins:")
        print_board(b)

        while(not terminal(b)):
            playy = human(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = alphabeta(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the alphabeta move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')

    else:
        print_board(b)

        while(not terminal(b)):
            playy = alea(b, BLACK)
            b = play(b, playy, BLACK)
            print_board(b)

            if terminal(b):
                break

            t = time.process_time()

            plays = legal_moves(b, WHITE)
            best_val = float("inf")
            
            for playy in plays:
                b1 = play(b, playy, WHITE)
                move_val = alphabeta(b1, WHITE, PROFONDEUR)

                if move_val < best_val:
                    best_move = playy
                    best_val = move_val

            elapsed_time = time.process_time() - t
            print(f'It took {elapsed_time} to choose the alphabeta move')

            b = play(b, best_move, WHITE)
            print_board(b)

        won = winner(b)
        if won == -1:
            print('WHITE wins!')
        else:
            print('BLACK wins!')


main4(PROFONDEUR)

