import os
import sys
import random
import math
import threading
import time
import copy
import pygame

# --- Constants ---
BOARD_WIDTH, BOARD_HEIGHT = 800, 800
SIDEBAR_WIDTH = 250
WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
HEIGHT = BOARD_HEIGHT
SQ_SIZE = BOARD_WIDTH // 8

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)

# Theme Palette: (Light, Dark, Name)
BOARD_THEMES = [
    ((240, 217, 181), (181, 136, 99), "Original"),
    ((255, 255, 255), (135, 206, 250), "Light Blue"),
    ((235, 236, 208), (119, 149, 86), "Classic"),
    ((255, 255, 255), (40, 40, 40), "Contrast"),
    ((222, 184, 135), (139, 69, 19), "Wood"),
    ((245, 222, 179), (101, 67, 33), "Real Wood"),
    ((238, 238, 210), (118, 150, 86), "Tournament"),
    ((173, 189, 143), (111, 143, 114), "Emerald"),
    ((157, 172, 255), (111, 115, 210), "Marine"),
    ((204, 183, 174), (112, 102, 119), "Dusk"),
    ((234, 240, 206), (187, 190, 100), "Wheat")
]
current_theme_index = 0

# Piece-Square Tables (PST)
PAWN_PST = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
]

KNIGHT_PST = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]

BISHOP_PST = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]

ROOK_PST = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
]

QUEEN_PST = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0,  0],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
]

KING_PST = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
]
HIGHLIGHT = (186, 202, 68)
SELECTED = (214, 214, 114)
CHECKMATE_COLOR = (255, 50, 50) # Red for King
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
SIDEBAR_BG = (50, 50, 50)
BUTTON_COLOR = (119, 149, 86)
BUTTON_HOVER_COLOR = (149, 179, 116)
BUTTON_SELECTED_COLOR = (200, 200, 50) # Yellowish for selected difficulty
PROMOTION_BG_COLOR = (200, 200, 200)
ARROW_COLOR = (0, 200, 0, 180) 

# Font Sizes
FONT_SIZE = 70
TIMER_SECONDS = 600 # 10 minutes

IMAGES = {}
SOUNDS = {}
SOUND_ENABLED = True

# Menu Layout
BTN_WIDTH, BTN_HEIGHT = 400, 70
CENTER_X = WIDTH // 2 - BTN_WIDTH // 2
CENTER_Y = HEIGHT // 2
# Define buttons relative to window center
BTN_START_RECT = pygame.Rect(CENTER_X, CENTER_Y - 250, BTN_WIDTH, BTN_HEIGHT)
BTN_WHITE_RECT = pygame.Rect(CENTER_X, CENTER_Y - 160, BTN_WIDTH, BTN_HEIGHT)
BTN_BLACK_RECT = pygame.Rect(CENTER_X, CENTER_Y - 70, BTN_WIDTH, BTN_HEIGHT)

# Difficulty Buttons (Row) - Now 4 buttons
DIFF_Y = CENTER_Y + 40
DIFF_WIDTH = 85
DIFF_HEIGHT = 50
BTN_EASY_RECT = pygame.Rect(CENTER_X - 10, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_MED_RECT = pygame.Rect(CENTER_X + 85, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_HARD_RECT = pygame.Rect(CENTER_X + 180, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_EXPERT_RECT = pygame.Rect(CENTER_X + 275, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_CLOCK_RECT = pygame.Rect(CENTER_X - 10, DIFF_Y + 65, 120, DIFF_HEIGHT)
BTN_THEME_RECT = pygame.Rect(CENTER_X + 120, DIFF_Y + 65, BTN_WIDTH - 120, 60)

# Sidebar Buttons
BTN_PAUSE_RECT = pygame.Rect(BOARD_WIDTH + 35, 460, 180, 45)
BTN_RESTART_RECT = pygame.Rect(BOARD_WIDTH + 35, 520, 180, 45)
BTN_SOUND_RECT = pygame.Rect(BOARD_WIDTH + 35, 580, 180, 45)
BTN_MENU_RECT = pygame.Rect(BOARD_WIDTH + 35, HEIGHT - 80, 180, 50)
BTN_BOTH_RECT = pygame.Rect(BOARD_WIDTH + 35, 400, 180, 45)

# --- Classes ---

class SearchTimeout(Exception):
    """Exception raised when the AI search exceeds its time limit."""
    pass

class Piece:
    def __init__(self, color, name, value):
        self.color = color
        self.name = name
        self.value = value
        self.has_moved = False

    def get_valid_moves(self, pos, board, game=None):
        raise NotImplementedError

class Pawn(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 10)

    def get_valid_moves(self, pos, board, game=None):
        moves = []
        r, c = pos
        direction = -1 if self.color == 'white' else 1
        
        # Forward
        if 0 <= r + direction < 8:
            if board[r + direction][c] is None:
                moves.append((r + direction, c))
                # Double Forward
                if (self.color == 'white' and r == 6) or (self.color == 'black' and r == 1):
                    if board[r + 2 * direction][c] is None:
                        moves.append((r + 2 * direction, c))
        
        # Captures
        for dc in [-1, 1]:
            if 0 <= r + direction < 8 and 0 <= c + dc < 8:
                target = board[r + direction][c + dc]
                if target and target.color != self.color:
                    moves.append((r + direction, c + dc))
                
                # En Passant
                if game and game.en_passant_target:
                    ep_r, ep_c = game.en_passant_target
                    if ep_r == r + direction and ep_c == c + dc:
                        moves.append((ep_r, ep_c))

        return moves

class Rook(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 50)

    def get_valid_moves(self, pos, board, game=None):
        return get_linear_moves(pos, board, self.color, [(1,0),(-1,0),(0,1),(0,-1)])

class Knight(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 30)

    def get_valid_moves(self, pos, board, game=None):
        moves = []
        r, c = pos
        offsets = [(1,2), (1,-2), (-1,2), (-1,-2), (2,1), (2,-1), (-2,1), (-2,-1)]
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = board[nr][nc]
                if not target or target.color != self.color:
                    moves.append((nr, nc))
        return moves

class Bishop(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 30)

    def get_valid_moves(self, pos, board, game=None):
        return get_linear_moves(pos, board, self.color, [(1,1), (1,-1), (-1,1), (-1,-1)])

class Queen(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 90)

    def get_valid_moves(self, pos, board, game=None):
        return get_linear_moves(pos, board, self.color, 
                                [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)])

class King(Piece):
    def __init__(self, color, name):
        super().__init__(color, name, 900)

    def get_valid_moves(self, pos, board, game=None):
        moves = []
        r, c = pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target = board[nr][nc]
                    if not target or target.color != self.color:
                        moves.append((nr, nc))
        return moves

def get_linear_moves(pos, board, color, directions):
    moves = []
    r, c = pos
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]
            if target is None:
                moves.append((nr, nc))
            elif target.color != color:
                moves.append((nr, nc))
                break
            else:
                break
            nr += dr
            nc += dc
    return moves

class AI:
    def __init__(self, depth=2):
        self.depth = depth
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(10)]  # Track killer moves per depth
        # Balanced move limits for speed and strength
        if self.depth >= 7:
            self.move_limit = 15  # Expert: Top 15 (improved from 8)
        elif self.depth == 6:
            self.move_limit = 18
        elif self.depth == 5:
            self.move_limit = 20  # Hard: Top 20 (improved from 12)
        elif self.depth == 4:
            self.move_limit = 22
        elif self.depth == 3:
            self.move_limit = 25
        elif self.depth == 2:
            self.move_limit = 30
        else:
            self.move_limit = None
        
        # Time-based search control
        self.search_start_time = None
        self.time_limit = None

    def evaluate_board(self, board, game_ref=None):
        score = 0
        
        # material weights
        # Mobility and King Safety skipped for brevity in search to keep it FAST
        # but let's add basic mobility if game_ref is provided
        
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece:
                    # Material Value
                    val = piece.value
                    
                    # Position Value (PST)
                    pst_val = 0
                    if isinstance(piece, Pawn): pst_val = PAWN_PST[r][c]
                    elif isinstance(piece, Knight): pst_val = KNIGHT_PST[r][c]
                    elif isinstance(piece, Bishop): pst_val = BISHOP_PST[r][c]
                    elif isinstance(piece, Rook): pst_val = ROOK_PST[r][c]
                    elif isinstance(piece, Queen): pst_val = QUEEN_PST[r][c]
                    elif isinstance(piece, King): pst_val = KING_PST[r][c]
                    
                    # Mirror for Black
                    if piece.color == 'black':
                        pst_name = piece.__class__.__name__.upper() + '_PST'
                        pst_table = globals().get(pst_name)
                        pst_val = pst_table[7-r][c] if pst_table else 0
                        score -= (val + pst_val)
                    else:
                        score += (val + pst_val)
        
        return score

    def get_board_hash(self, board, turn):
        # Faster tuple-based hashing for transposition table
        board_tuple = []
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p:
                    board_tuple.append((r, c, p.name, p.color, p.has_moved))
        return (tuple(board_tuple), turn)

    def get_best_move(self, board, game_ref, ai_color):
        maximize = (ai_color == 'white')
        best_move = None
        
        # Isolation: create a copy of the board for the search thread
        search_board = self.fast_copy_board(board)
        
        # Use Iterative Deepening for better time management (especially for Hard/Expert)
        target_depth = self.depth
        depth_reached = 0
        
        try:
            # We always start with a shallow depth to have a baseline move
            for d in range(1, target_depth + 1):
                # Clear transposition table for each depth if needed, 
                # but often keeping it helps (just check depth)
                # self.transposition_table = {} 
                
                current_best_score = -float('inf') if maximize else float('inf')
                current_best_move = None
                
                all_moves = self.get_all_legal_moves_via_game_logic(search_board, ai_color, game_ref)
                all_moves.sort(key=lambda mv: self.move_priority(search_board, mv, d), reverse=True)
                
                if self.move_limit and len(all_moves) > self.move_limit:
                    all_moves = all_moves[:self.move_limit]
                
                if not all_moves:
                    break

                for start, end in all_moves:
                    # Check for timeout inside the move loop
                    if self.time_limit and self.search_start_time:
                        if time.time() - self.search_start_time > self.time_limit:
                            raise SearchTimeout()

                    move_info = self.make_move(search_board, start, end)
                    score = self.minimax(search_board, d - 1, -float('inf'), float('inf'), not maximize, game_ref)
                    self.undo_move(search_board, start, end, move_info)
                    
                    if maximize:
                        if score > current_best_score:
                            current_best_score = score
                            current_best_move = (start, end)
                    else:
                        if score < current_best_score:
                            current_best_score = score
                            current_best_move = (start, end)
                
                # If we completed this depth fully, save the results
                best_move = current_best_move
                depth_reached = d
                # print(f"DEBUG: Completed depth {d}")

        except SearchTimeout:
            print(f"DEBUG: AI Search timeout reached. Returning best move from depth {depth_reached}")
        
        return best_move

    def fast_copy_board(self, board):
        new_board = [[None for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p:
                    new_p = p.__class__(p.color, p.name)
                    new_p.has_moved = p.has_moved
                    new_board[r][c] = new_p
        return new_board

    def minimax(self, board, depth, alpha, beta, maximizing_player, game_ref):
        # Check time limit periodically
        if self.time_limit is not None and self.search_start_time is not None:
            if time.time() - self.search_start_time > self.time_limit:
                raise SearchTimeout()
        
        state_hash = self.get_board_hash(board, 'white' if maximizing_player else 'black')
        if state_hash in self.transposition_table:
            cached_depth, cached_eval = self.transposition_table[state_hash]
            if cached_depth >= depth:
                return cached_eval

        if depth == 0:
            eval = self.evaluate_board(board)
            self.transposition_table[state_hash] = (depth, eval)
            return eval

        if maximizing_player: 
            max_eval = -float('inf')
            moves = self.get_all_legal_moves_via_game_logic(board, 'white', game_ref)
            moves.sort(key=lambda mv: self.move_priority(board, mv, depth), reverse=True)
            
            if self.move_limit and len(moves) > self.move_limit: 
                moves = moves[:self.move_limit]
            
            if not moves:
                if game_ref.is_check('white', board): return -20000 - depth
                return 0
            
            for i, (start, end) in enumerate(moves):
                move_info = self.make_move(board, start, end)
                eval = self.minimax(board, depth - 1, alpha, beta, False, game_ref)
                self.undo_move(board, start, end, move_info)
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Store killer move
                    if depth < 10 and board[end[0]][end[1]] is None:
                        if self.killer_moves[depth][0] != (start, end):
                            self.killer_moves[depth][1] = self.killer_moves[depth][0]
                            self.killer_moves[depth][0] = (start, end)
                    break
            
            self.transposition_table[state_hash] = (depth, max_eval)
            return max_eval
        
        else: 
            min_eval = float('inf')
            moves = self.get_all_legal_moves_via_game_logic(board, 'black', game_ref)
            moves.sort(key=lambda mv: self.move_priority(board, mv, depth), reverse=True)
            
            if self.move_limit and len(moves) > self.move_limit:
                moves = moves[:self.move_limit]
                
            if not moves:
                if game_ref.is_check('black', board): return 20000 + depth
                return 0
            
            for i, (start, end) in enumerate(moves):
                move_info = self.make_move(board, start, end)
                eval = self.minimax(board, depth - 1, alpha, beta, True, game_ref)
                self.undo_move(board, start, end, move_info)
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    # Store killer move
                    if depth < 10 and board[end[0]][end[1]] is None:
                        if self.killer_moves[depth][0] != (start, end):
                            self.killer_moves[depth][1] = self.killer_moves[depth][0]
                            self.killer_moves[depth][0] = (start, end)
                    break
            
            self.transposition_table[state_hash] = (depth, min_eval)
            return min_eval

    def get_all_legal_moves_via_game_logic(self, board, color, game_ref):
        moves = []
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and piece.color == color:
                    valid_moves = game_ref.get_legal_moves(piece, (r, c), board)
                    for move in valid_moves:
                        moves.append(((r, c), move))
        return moves

    def move_priority(self, board, move, depth=0):
        # Enhanced move ordering: Killer moves + MVV-LVA
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]
        
        if target:
            # Capture: Highest priority (MVV-LVA)
            return 10000 + 10 * target.value - piece.value
        
        # Killer move heuristic: moves that caused cutoffs at this depth
        if depth < 10 and move in self.killer_moves[depth]:
            return 5000
        
        # Non-captures: Low priority
        return 0

    def make_move(self, board, start, end):
        r1, c1 = start
        r2, c2 = end
        piece = board[r1][c1]
        target = board[r2][c2]
        
        prev_has_moved = piece.has_moved
        captured_rook_info = None
        
        # Castling
        if isinstance(piece, King) and abs(c2 - c1) == 2:
            if c2 == 6: # Kingside
                rook = board[r1][7]
                board[r1][5] = rook
                board[r1][7] = None
                if rook:
                    captured_rook_info = (rook, (r1, 7), rook.has_moved)
                    rook.has_moved = True
            elif c2 == 2: # Queenside
                rook = board[r1][0]
                board[r1][3] = rook
                board[r1][0] = None
                if rook:
                    captured_rook_info = (rook, (r1, 0), rook.has_moved)
                    rook.has_moved = True

        # AI Queen Promotion (simplified for search)
        promoted_from = None
        if isinstance(piece, Pawn):
             if (piece.color == 'white' and r2 == 0) or (piece.color == 'black' and r2 == 7):
                 promoted_from = piece
                 piece = Queen(piece.color, 'Queen')

        board[r2][c2] = piece
        board[r1][c1] = None
        piece.has_moved = True
        
        return (target, prev_has_moved, captured_rook_info, promoted_from)

    def undo_move(self, board, start, end, move_info):
        r1, c1 = start
        r2, c2 = end
        target, prev_has_moved, captured_rook_info, promoted_from = move_info
        
        piece = board[r2][c2]
        
        # If it was a promotion, revert the piece type
        if promoted_from:
            piece = promoted_from
            
        board[r1][c1] = piece
        board[r2][c2] = target
        piece.has_moved = prev_has_moved
        
        # Undo Castling
        if captured_rook_info:
            rook, prev_rook_pos, rook_has_moved = captured_rook_info
            # The rook was moved to r1,5 or r1,3
            if end[1] == 6: # Kingside
                board[r1][5] = None
            else: # Queenside
                board[r1][3] = None
            board[prev_rook_pos[0]][prev_rook_pos[1]] = rook
            rook.has_moved = rook_has_moved

    def move_heuristic(self, board, move):
        return self.move_priority(board, move)

class Game:
    def __init__(self, player_color='white', ai_depth=2):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.turn = 'white'
        self.player_color = player_color
        self.selected_piece = None
        self.selected_pos = None
        self.valid_moves = []
        self.ai = AI(depth=ai_depth)
        self.paused = True 
        self.start_game()
        
    def start_game(self):
        self.setup_board()
        self.game_over = False
        self.winner = None
        self.promotion_pending = False
        self.promotion_pos = None 
        self.last_move = None 
        self.paused = True 
        self.en_passant_target = None
        self.in_check = False
        self.full_move_number = 0
        self.game_over_time = 0
        
        # Performance Feedback
        self.performance_stars = 0
        self.performance_msg = ""
        
        # Draw Rules
        self.board_history = {} 
        self.half_move_clock = 0
        
        # AI Threading & State
        self.ai_thinking = False
        self.ai_move_result = None

        # Chess Clock
        self.white_time = TIMER_SECONDS
        self.black_time = TIMER_SECONDS
        self.last_update_time = pygame.time.get_ticks()

    def setup_board(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        for c in range(8):
            self.board[6][c] = Pawn('white', 'Pawn')
            self.board[1][c] = Pawn('black', 'Pawn')
        layout = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for c, cls in enumerate(layout):
            self.board[7][c] = cls('white', cls.__name__)
            self.board[0][c] = cls('black', cls.__name__)

    def update_timers(self):
        current_time = pygame.time.get_ticks()
        
        if self.game_over or self.promotion_pending or self.paused or not globals().get('CLOCK_ENABLED', True): 
            self.last_update_time = current_time 
            return

        dt = (current_time - self.last_update_time) / 1000.0
        self.last_update_time = current_time

        if self.turn == 'white':
            self.white_time -= dt
            if self.white_time <= 0:
                self.white_time = 0
                self.game_over = True
                self.winner = 'black'
        else:
            self.black_time -= dt
            if self.black_time <= 0:
                self.black_time = 0
                self.game_over = True
                self.winner = 'white'
        
        if self.game_over:
            if not self.game_over_time:
                self.game_over_time = pygame.time.get_ticks()
            self.calculate_performance()

    def get_piece(self, r, c):
        return self.board[r][c]

    def select(self, r, c):
        if self.game_over or self.promotion_pending or self.paused: return
        if self.turn != self.player_color and globals().get('AI_ENABLED', True): return

        if self.selected_piece is None:
            piece = self.get_piece(r, c)
            if piece and piece.color == self.turn:
                self.selected_piece = piece
                self.selected_pos = (r, c)
                self.valid_moves = self.get_legal_moves(piece, (r, c), self.board)
                return True
        else:
            if (r, c) in self.valid_moves:
                self.move(self.selected_pos, (r, c))
                self.selected_piece = None
                self.selected_pos = None
                self.valid_moves = []
                
                if not self.promotion_pending:
                    self.switch_turn()
                return True
            else:
                piece = self.get_piece(r, c)
                if piece and piece.color == self.turn:
                    self.selected_piece = piece
                    self.selected_pos = (r, c)
                    self.valid_moves = self.get_legal_moves(piece, (r, c), self.board)
                    return True
                else:
                    self.selected_piece = None
                    self.selected_pos = None
                    self.valid_moves = []
        return False

    def move(self, start, end):
        r1, c1 = start
        r2, c2 = end
        piece = self.board[r1][c1]
        # Remember if this was a pawn (to update half-move clock correctly)
        was_pawn = isinstance(piece, Pawn)

        # Track captured piece (including en-passant) for half-move logic
        captured_piece = self.board[r2][c2]

        # Castling
        if isinstance(piece, King) and abs(c2 - c1) == 2:
            if c2 == 6: 
                rook = self.board[r1][7]
                self.board[r1][5] = rook
                self.board[r1][7] = None
                if rook: rook.has_moved = True
            elif c2 == 2:
                rook = self.board[r1][0]
                self.board[r1][3] = rook
                self.board[r1][0] = None
                if rook: rook.has_moved = True

                # En Passant Capture
                if isinstance(piece, Pawn) and (r2, c2) == self.en_passant_target:
                        # Capture the pawn behind
                        capture_r = r1
                        capture_c = c2
                        # record captured pawn for half-move clock
                        captured_piece = self.board[capture_r][capture_c]
                        self.board[capture_r][capture_c] = None

        # Set En Passant Target for NEXT turn
        if isinstance(piece, Pawn) and abs(r2 - r1) == 2:
             self.en_passant_target = ((r1 + r2) // 2, c1)
        else:
             self.en_passant_target = None
        
        # Promotion Check
        if isinstance(piece, Pawn):
            if (piece.color == 'white' and r2 == 0) or (piece.color == 'black' and r2 == 7):
                if self.turn == self.player_color:
                    self.promotion_pending = True
                    self.promotion_pos = (r2, c2)
                else:
                    piece = Queen(piece.color, 'Queen')

        self.board[r2][c2] = piece
        self.board[r1][c1] = None
        piece.has_moved = True

        # Play Move/Capture Sound
        if SOUND_ENABLED:
            if captured_piece:
                if 'capture' in SOUNDS: 
                    print("DEBUG: Playing capture sound")
                    SOUNDS['capture'].play()
            else:
                if 'move' in SOUNDS: 
                    print("DEBUG: Playing move sound")
                    SOUNDS['move'].play()
        
        # Track Last Move
        self.last_move = (start, end)
        
        # Record history for repetition
        state_key = self.get_board_state_key()
        self.board_history[state_key] = self.board_history.get(state_key, 0) + 1

        # Update half-move clock: reset on pawn move or capture, otherwise increment
        if was_pawn or (captured_piece is not None):
            self.half_move_clock = 0
        else:
            self.half_move_clock = getattr(self, 'half_move_clock', 0) + 1

    def promote_pawn(self, piece_class):
        if not self.promotion_pending: return
        
        r, c = self.promotion_pos
        self.board[r][c] = piece_class(self.turn, piece_class.__name__)
        self.promotion_pending = False
        self.promotion_pos = None
        self.switch_turn()

    def switch_turn(self):
        # Update Move Counter when Black finishes turn (moving to White)
        if self.turn == 'black':
            self.full_move_number += 1

        self.turn = 'black' if self.turn == 'white' else 'white'
        self.in_check = self.is_check(self.turn, self.board) # Update check status
        
        # Check Game Over Conditions
        if self.is_checkmate(self.turn, self.board):
            self.game_over = True
            self.winner = 'black' if self.turn == 'white' else 'white'
        elif self.is_stalemate(self.turn, self.board):
            self.game_over = True
            self.winner = 'Draw (Stalemate)'
        elif self.is_insufficient_material():
            self.game_over = True
            self.winner = 'Draw (Insufficient Material)'
        elif self.check_threefold_repetition():
            self.game_over = True
            self.winner = 'Draw (3-fold Repetition)'

        # Sound for Check
        if not self.game_over and self.in_check and SOUND_ENABLED:
            if 'check' in SOUNDS: 
                print("DEBUG: Playing check sound")
                SOUNDS['check'].play()

        # 50-move rule: if 100 half-moves (50 full moves) have occurred without a pawn move or capture
        elif getattr(self, 'half_move_clock', 0) >= 100:
            self.game_over = True
            self.winner = 'Draw (50-move rule)'

        if self.game_over:
            if not self.game_over_time:
                self.game_over_time = pygame.time.get_ticks()
            self.calculate_performance()

    def calculate_performance(self):
        # Calculate material balance
        white_val = 0
        black_val = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p:
                    if p.color == 'white': white_val += p.value
                    else: black_val += p.value
        
        diff = white_val - black_val if self.player_color == 'white' else black_val - white_val
        
        if self.winner == self.player_color:
            if diff > 50:
                self.performance_stars = 5
                self.performance_msg = "EXCELLENT! YOU PLAYED GOOD"
            else:
                self.performance_stars = 4
                self.performance_msg = "YOU PLAYED GOOD"
        elif "Draw" in str(self.winner):
            self.performance_stars = 3
            self.performance_msg = "NOT BAD, KEEP PRACTICING"
        else: # Player Lost
            if diff > -30:
                self.performance_stars = 2
                self.performance_msg = "YOU WERE CLOSE! TRY AGAIN"
            else:
                self.performance_stars = 1
                self.performance_msg = "YOU HAVE A LOT MORE TO LEARN"

    def get_legal_moves(self, piece, pos, board):
        pseudo_moves = piece.get_valid_moves(pos, board, self)
        legal_moves = []
        
        for move in pseudo_moves:
            if not self.simulate_move(pos, move, piece.color, board):
                legal_moves.append(move)
        
        # Castling Logic
        if isinstance(piece, King) and not piece.has_moved and not self.is_check(piece.color, board):
            r, c = pos
            if piece.color == 'white' and r == 7:
                if board[7][7] and isinstance(board[7][7], Rook) and not board[7][7].has_moved:
                    if board[7][5] is None and board[7][6] is None:
                        if not self.is_square_attacked((7, 5), 'white', board) and \
                           not self.is_square_attacked((7, 6), 'white', board):
                            legal_moves.append((7, 6))
                if board[7][0] and isinstance(board[7][0], Rook) and not board[7][0].has_moved:
                    if board[7][1] is None and board[7][2] is None and board[7][3] is None:
                        if not self.is_square_attacked((7, 3), 'white', board) and \
                           not self.is_square_attacked((7, 2), 'white', board):
                            legal_moves.append((7, 2))
            
            elif piece.color == 'black' and r == 0:
                if board[0][7] and isinstance(board[0][7], Rook) and not board[0][7].has_moved:
                    if board[0][5] is None and board[0][6] is None:
                        if not self.is_square_attacked((0, 5), 'black', board) and \
                           not self.is_square_attacked((0, 6), 'black', board):
                            legal_moves.append((0, 6))
                if board[0][0] and isinstance(board[0][0], Rook) and not board[0][0].has_moved:
                    if board[0][1] is None and board[0][2] is None and board[0][3] is None:
                        if not self.is_square_attacked((0, 3), 'black', board) and \
                           not self.is_square_attacked((0, 2), 'black', board):
                            legal_moves.append((0, 2))

        return legal_moves

    def is_square_attacked(self, pos, color, board):
        # Optimized attack detection: check rays and knight offsets instead of scanning all pieces
        enemy_color = 'black' if color == 'white' else 'white'
        r, c = pos

        # Knight offset checks
        knight_offsets = [(1,2), (1,-2), (-1,2), (-1,-2), (2,1), (2,-1), (-2,1), (-2,-1)]
        for dr, dc in knight_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                p = board[nr][nc]
                if p and p.color == enemy_color and isinstance(p, Knight):
                    return True

        # Ray checks (Rook, Bishop, Queen, King)
        directions = [
            (1,0), (-1,0), (0,1), (0,-1), # Straights
            (1,1), (1,-1), (-1,1), (-1,-1) # Diagonals
        ]
        for i, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            dist = 1
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = board[nr][nc]
                if p:
                    if p.color == enemy_color:
                        if i < 4: # Straights
                            if isinstance(p, (Rook, Queen)): return True
                        else: # Diagonals
                            if isinstance(p, (Bishop, Queen)): return True
                        # King can only attack at distance 1
                        if dist == 1 and isinstance(p, King): return True
                        # Pawn special case
                        if dist == 1 and isinstance(p, Pawn):
                            pawn_dir = -1 if enemy_color == 'white' else 1 # White moves up (-r), Black moves down (+r)
                            if r == nr + pawn_dir and (c == nc + 1 or c == nc - 1):
                                return True
                    break # Blocked by any piece
                nr += dr
                nc += dc
                dist += 1
        return False

    def simulate_move(self, start, end, color, board):
        r1, c1 = start
        r2, c2 = end
        target = board[r2][c2]
        piece = board[r1][c1]
        
        board[r2][c2] = piece
        board[r1][c1] = None
        
        in_check = self.is_check(color, board)
        
        board[r1][c1] = piece
        board[r2][c2] = target
        
        return in_check

    def is_check(self, color, board):
        king_pos = self.find_king(board, color)
        if not king_pos: return False 
        return self.is_square_attacked(king_pos, color, board)

    def find_king(self, board, color):
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p and p.color == color and isinstance(p, King):
                    return (r, c)
        return None

    def is_checkmate(self, color, board):
        if not self.is_check(color, board):
            return False
        return self.has_no_legal_moves(color, board)

    def is_stalemate(self, color, board):
        if self.is_check(color, board):
            return False
        return self.has_no_legal_moves(color, board)

    def has_no_legal_moves(self, color, board):
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p and p.color == color:
                    moves = self.get_legal_moves(p, (r, c), board)
                    if moves: 
                        return False
        return True

    def is_insufficient_material(self):
        pieces = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c]:
                    pieces.append(self.board[r][c])
        
        if len(pieces) == 2: return True # K vs K
        
        if len(pieces) == 3:
            for p in pieces:
                if isinstance(p, (Knight, Bishop)): return True # K+N vs K or K+B vs K
        
        return False

    def get_board_state_key(self):
        # Create a simple string representation
        # Includes turn, En Passant target, Castle rights (simplified via has_moved)
        # and board layout
        state = f"{self.turn}|{self.en_passant_target}|"
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p:
                    state += f"{p.name}{p.color}{r}{c}"
        return state
    
    def check_threefold_repetition(self):
        state_key = self.get_board_state_key()
        return self.board_history.get(state_key, 0) >= 3
    
    def ai_move(self):
        if self.game_over or self.paused or self.ai_thinking: return
        
        self.ai_thinking = True
        self.ai_move_result = None
        
        def run_ai_search():
            try:
                print(f"AI ({self.turn}) is thinking...")
                start_time = time.time()
                
                # For Hard/Expert levels, set a 2-second time limit
                if self.ai.depth >= 5:
                    self.ai.search_start_time = start_time
                    self.ai.time_limit = 2.0
                else:
                    self.ai.search_start_time = None
                    self.ai.time_limit = None
                
                move = self.ai.get_best_move(self.board, self, self.turn)
                
                elapsed = time.time() - start_time
                if self.ai.depth >= 5:
                    level_name = "Expert" if self.ai.depth >= 7 else "Hard"
                    print(f"{level_name} AI found move in {elapsed:.1f} seconds")
                else:
                    # For Easy/Medium levels, ensure minimum 1 second response time
                    if elapsed < 1.0:
                        time.sleep(1.0 - elapsed)
                        print(f"AI took {1.0:.1f} seconds (minimum delay)")
                    else:
                        print(f"AI found move in {elapsed:.1f} seconds")
                
                self.ai_move_result = move
                if move:
                    print(f"AI found move: {move}")
                else:
                    print("AI found no legal moves.")
            except Exception as e:
                print(f"CRITICAL AI ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.ai_thinking = False
        
        thread = threading.Thread(target=run_ai_search)
        thread.daemon = True
        thread.start()

    def apply_ai_move(self):
        if not self.ai_thinking and self.ai_move_result:
            start, end = self.ai_move_result
            self.ai_move_result = None
            self.move(start, end)
            self.switch_turn()
            pygame.display.set_caption("Python Chess (vs AI)")
        elif not self.ai_thinking and self.ai_move_result is None:
             # AI gave up or error, but switch_turn usually handles stalemate/checkmate
             pass


# --- GUI ---

def load_images():
    pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    for p in pieces:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "images", p + ".png")
        try:
            image = pygame.image.load(path)
            IMAGES[p] = pygame.transform.scale(image, (SQ_SIZE, SQ_SIZE))
        except pygame.error as e:
            print(f"Error loading image {path}: {e}")
            sys.exit()
    
    # Load realistic photographic wood textures
    for wood in ['light_wood', 'dark_wood']:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "images", wood + ".png")
        if os.path.exists(path):
            try:
                img = pygame.image.load(path)
                IMAGES[wood] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
                print(f"Successfully loaded {wood} texture.")
            except:
                pass

def load_sounds():
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"MIXER ERROR: {e}")
        return

    sounds = ['move', 'capture', 'check', 'game_over']
    for s in sounds:
        # Support .wav, .ogg, and .mp3 (prefer .ogg/.wav)
        found = False
        for ext in ['.ogg', '.wav', '.mp3']:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sounds", s + ext)
            if os.path.exists(path):
                try:
                    target_sound = pygame.mixer.Sound(path)
                    target_sound.set_volume(1.0)
                    SOUNDS[s] = target_sound
                    found = True
                    print(f"Successfully loaded: {s}{ext}")
                    break
                except Exception as e:
                    print(f"Error loading sound {s}{ext}: {e}")
        
        if not found:
            print(f"Sound file not found for: {s}")

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

def draw_arrow(surface, start_pos, end_pos, color):
    arrow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    
    start = pygame.Vector2(start_pos)
    end = pygame.Vector2(end_pos)
    
    arrow_vec = end - start
    if arrow_vec.length() == 0: return 
    
    direction = arrow_vec.normalize()
    pygame.draw.line(arrow_surf, color, start, end, 14)
    
    # Triangle head
    head_size = 32
    rotation = math.atan2(end.y - start.y, end.x - start.x)
    arrow_angle = math.pi / 6 
    
    p1 = end
    p2 = end - pygame.Vector2(math.cos(rotation + arrow_angle) * head_size, math.sin(rotation + arrow_angle) * head_size)
    p3 = end - pygame.Vector2(math.cos(rotation - arrow_angle) * head_size, math.sin(rotation - arrow_angle) * head_size)
    
    pygame.draw.polygon(arrow_surf, color, [p1, p2, p3])
    
    surface.blit(arrow_surf, (0, 0))

def draw_star(surface, center, size, color):
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        radius = size if i % 2 == 0 else size // 2
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    pygame.draw.polygon(surface, color, points)

def draw_sidebar(win, game, font):
    mouse_pos = pygame.mouse.get_pos()
    # Sidebar Background
    sidebar_rect = pygame.Rect(BOARD_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
    pygame.draw.rect(win, SIDEBAR_BG, sidebar_rect)
    
    # Title
    title_rect = pygame.Rect(BOARD_WIDTH, 20, SIDEBAR_WIDTH, 60)
    title_surf = font.render("CHESS", True, WHITE_COLOR)
    title_text_rect = title_surf.get_rect(center=title_rect.center)
    win.blit(title_surf, title_text_rect)
    
    # Turn Indicator
    turn_text = "White's Turn" if game.turn == 'white' else "Black's Turn"
    color = WHITE_COLOR if game.turn == 'white' else (150, 150, 150)
    ind_font = pygame.font.SysFont("arial", 36)
    ind_surf = ind_font.render(turn_text, True, color)
    ind_rect = ind_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 110))
    win.blit(ind_surf, ind_rect)

    # Move Counter
    move_font = pygame.font.SysFont("arial", 26)
    move_text = f"Move: {game.full_move_number}"
    move_surf = move_font.render(move_text, True, (200, 200, 200))
    move_rect = move_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 160))
    win.blit(move_surf, move_rect)

    # Half-move clock display (for 50-move draw rule)
    half_text = f"Half-Moves: {getattr(game, 'half_move_clock', 0)}"
    half_surf = move_font.render(half_text, True, (200, 200, 200))
    half_rect = half_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 200))
    win.blit(half_surf, half_rect)

    # AI Toggle Button (Both/AI)
    both_color = BUTTON_COLOR if not BTN_BOTH_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, both_color, BTN_BOTH_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_BOTH_RECT, 3, border_radius=10)
    ai_state = globals().get('AI_ENABLED', True)
    both_text = "AI: On" if ai_state else "AI: Off"
    both_surf = font.render(both_text, True, WHITE_COLOR)
    both_rect = both_surf.get_rect(center=BTN_BOTH_RECT.center)
    win.blit(both_surf, both_rect)

    

    # Timers (Only if Clock is Enabled)
    if globals().get('CLOCK_ENABLED', True):
        # Black Time (Top)
        b_time_str = format_time(game.black_time)
        b_surf = font.render(b_time_str, True, WHITE_COLOR)
        b_rect = b_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 260))
        pygame.draw.rect(win, BLACK_COLOR, b_rect.inflate(20, 10), border_radius=5)
        win.blit(b_surf, b_rect)
        
        label_font = pygame.font.SysFont("arial", 24)
        b_label = label_font.render("Black", True, (200, 200, 200))
        win.blit(b_label, (BOARD_WIDTH + 20, 230))

        # White Time (Below Black)
        w_time_str = format_time(game.white_time)
        w_surf = font.render(w_time_str, True, WHITE_COLOR)
        w_rect = w_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 350))
        pygame.draw.rect(win, BLACK_COLOR, w_rect.inflate(20, 10), border_radius=5)
        win.blit(w_surf, w_rect)

        w_label = label_font.render("White", True, (200, 200, 200))
        win.blit(w_label, (BOARD_WIDTH + 20, 320))

    # Controls (Pause / Restart) - Shifted Down
    
    # Pause Button
    BTN_PAUSE_RECT.y = 440
    p_color = BUTTON_COLOR if not BTN_PAUSE_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, p_color, BTN_PAUSE_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_PAUSE_RECT, 3, border_radius=10)
    
    if game.paused:
        p_text = "Start" if game.last_move is None else "Resume"
    else:
        p_text = "Pause"
        
    p_surf = font.render(p_text, True, WHITE_COLOR)
    p_rect = p_surf.get_rect(center=BTN_PAUSE_RECT.center)
    win.blit(p_surf, p_rect)
    
    # Restart Button
    BTN_RESTART_RECT.y = 505
    r_color = BUTTON_COLOR if not BTN_RESTART_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, r_color, BTN_RESTART_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_RESTART_RECT, 3, border_radius=10)
    r_surf = font.render("Restart", True, WHITE_COLOR)
    r_rect = r_surf.get_rect(center=BTN_RESTART_RECT.center)
    win.blit(r_surf, r_rect)

    # Sound Toggle Button
    BTN_SOUND_RECT.y = 570
    s_color = BUTTON_COLOR if not BTN_SOUND_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, s_color, BTN_SOUND_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_SOUND_RECT, 3, border_radius=10)
    s_text = "Sound: On" if SOUND_ENABLED else "Sound: Off"
    s_surf = font.render(s_text, True, WHITE_COLOR)
    s_rect = s_surf.get_rect(center=BTN_SOUND_RECT.center)
    win.blit(s_surf, s_rect)

    # Sidebar AI/Both Toggle positioning
    BTN_BOTH_RECT.y = 375
    
    # Draw Menu Button in Sidebar
    BTN_MENU_RECT.y = 720
    color = BUTTON_COLOR if not BTN_MENU_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, color, BTN_MENU_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_MENU_RECT, 3, border_radius=10)
    m_surf = font.render("Menu", True, WHITE_COLOR)
    m_rect = m_surf.get_rect(center=BTN_MENU_RECT.center)
    win.blit(m_surf, m_rect)
    
    # Pause Overlay text if paused
    if game.paused:
        overlay = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        win.blit(overlay, (0, 0))
        
        pause_font = pygame.font.SysFont("arial", 60, bold=True)
        pause_surf = pause_font.render("PAUSED", True, WHITE_COLOR)
        pause_rect = pause_surf.get_rect(center=(BOARD_WIDTH // 2, BOARD_HEIGHT // 2))
        win.blit(pause_surf, pause_rect)

def draw_menu(win, font, current_difficulty, dropdown_open=False):
    win.fill(LIGHT_SQ) 
    
    title_font = pygame.font.SysFont("arial", 80, bold=True)
    title_surf = title_font.render("CHESS", True, BLACK_COLOR)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 10))
    win.blit(title_surf, title_rect)
    
    mouse_pos = pygame.mouse.get_pos()
    
    def draw_btn(rect, text, selected=False):
        if selected:
            color = BUTTON_SELECTED_COLOR
        else:
            color = BUTTON_COLOR if not rect.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
            
        pygame.draw.rect(win, color, rect, border_radius=10)
        pygame.draw.rect(win, BLACK_COLOR, rect, 3, border_radius=10)
        t = font.render(text, True, WHITE_COLOR)
        r = t.get_rect(center=rect.center)
        win.blit(t, r)

    draw_btn(BTN_START_RECT, "Start Game (Random)")
    draw_btn(BTN_WHITE_RECT, "Play as White")
    draw_btn(BTN_BLACK_RECT, "Play as Black")
    
    # Draw Difficulty Buttons
    diff_font = pygame.font.SysFont("arial", 24)
    
    # Label
    l_surf = diff_font.render("Difficulty:", True, BLACK_COLOR)
    win.blit(l_surf, (BTN_EASY_RECT.x, BTN_EASY_RECT.y - 35))
    
    def draw_diff_btn(rect, text, level):
        selected = (current_difficulty == level)
        if selected:
             color = BUTTON_SELECTED_COLOR
        else:
             color = BUTTON_COLOR if not rect.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
        
        pygame.draw.rect(win, color, rect, border_radius=10)
        pygame.draw.rect(win, BLACK_COLOR, rect, 3, border_radius=10)
        t = diff_font.render(text, True, WHITE_COLOR)
        r = t.get_rect(center=rect.center)
        win.blit(t, r)

    draw_diff_btn(BTN_EASY_RECT, "Easy", 1)
    draw_diff_btn(BTN_MED_RECT, "Med", 2)
    draw_diff_btn(BTN_HARD_RECT, "Hard", 5)
    draw_diff_btn(BTN_EXPERT_RECT, "Expert", 7)
    
    # Clock Toggle Button
    clock_enabled = globals().get('CLOCK_ENABLED', True)
    c_color = BUTTON_SELECTED_COLOR if clock_enabled else BUTTON_COLOR
    if not clock_enabled and BTN_CLOCK_RECT.collidepoint(mouse_pos):
        c_color = BUTTON_HOVER_COLOR
        
    pygame.draw.rect(win, c_color, BTN_CLOCK_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_CLOCK_RECT, 3, border_radius=10)
    c_text = "Clock: On" if clock_enabled else "No Clock"
    c_surf = diff_font.render(c_text, True, WHITE_COLOR)
    c_rect = c_surf.get_rect(center=BTN_CLOCK_RECT.center)
    win.blit(c_surf, c_rect)

    # Change Board Theme Button
    t_color = BUTTON_COLOR if not BTN_THEME_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, t_color, BTN_THEME_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_THEME_RECT, 3, border_radius=10)
    
    theme_name = BOARD_THEMES[globals().get('current_theme_index', 0)][2]
    t_surf = diff_font.render(f"Board: {theme_name}", True, WHITE_COLOR)
    t_rect = t_surf.get_rect(center=BTN_THEME_RECT.center)
    win.blit(t_surf, t_rect)
    
    # Arrow Indicator for Dropdown
    arrow_char = "▲" if dropdown_open else "▼"
    a_surf = diff_font.render(arrow_char, True, WHITE_COLOR)
    a_rect = a_surf.get_rect(midright=(BTN_THEME_RECT.right - 15, BTN_THEME_RECT.centery))
    win.blit(a_surf, a_rect)

    dropdown_rects = []
    if dropdown_open:
        option_height = 40
        for i, theme in enumerate(BOARD_THEMES):
            option_rect = pygame.Rect(BTN_THEME_RECT.x, BTN_THEME_RECT.bottom + i * option_height, BTN_THEME_RECT.width, option_height)
            o_color = (80, 80, 80) if not option_rect.collidepoint(mouse_pos) else (100, 100, 100)
            if globals().get('current_theme_index', 0) == i:
                o_color = BUTTON_SELECTED_COLOR
            
            pygame.draw.rect(win, o_color, option_rect)
            pygame.draw.rect(win, BLACK_COLOR, option_rect, 1)
            
            o_text = diff_font.render(theme[2], True, WHITE_COLOR)
            o_rect = o_text.get_rect(center=option_rect.center)
            win.blit(o_text, o_rect)
            dropdown_rects.append((option_rect, i))
            
    return dropdown_rects
    

def draw_promotion_menu(win, game):
    overlay = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    win.blit(overlay, (0, 0))
    
    menu_width, menu_height = 400, 200
    menu_rect = pygame.Rect((BOARD_WIDTH - menu_width) // 2, (HEIGHT - menu_height) // 2, menu_width, menu_height)
    pygame.draw.rect(win, PROMOTION_BG_COLOR, menu_rect, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, menu_rect, 3, border_radius=10)
    
    options = [Queen, Rook, Bishop, Knight]
    width_per_option = menu_width // 4
    
    rects = []
    for i, cls in enumerate(options):
        color_prefix = 'w' if game.turn == 'white' else 'b'
        type_map = {Rook:'r', Knight:'n', Bishop:'b', Queen:'q'}
        key = color_prefix + type_map[cls]
        
        x = menu_rect.x + (i * width_per_option) + (width_per_option - SQ_SIZE) // 2
        y = menu_rect.y + (menu_height - SQ_SIZE) // 2
        
        if key in IMAGES:
            win.blit(IMAGES[key], (x, y))
            
        rect = pygame.Rect(menu_rect.x + (i * width_per_option), menu_rect.y, width_per_option, menu_height)
        rects.append((rect, cls))
        
    return rects

def draw(win, game, font):
    flip = (game.player_color == 'black')

    # Draw Board
    theme = BOARD_THEMES[globals().get('current_theme_index', 0)]
    l_color, d_color = theme[0], theme[1]

    for r in range(8):
        for c in range(8):
            draw_r = 7 - r if flip else r
            draw_c = 7 - c if flip else c
            
            color = l_color if (r + c) % 2 == 0 else d_color
            # Draw base square color
            pygame.draw.rect(win, color, (draw_c * SQ_SIZE, draw_r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

            # Add Wood Texture (Photographic or Procedural)
            if "Wood" in theme[2]:
                is_light = ((r + c) % 2 == 0)
                texture_key = 'light_wood' if is_light else 'dark_wood'
                if texture_key in IMAGES:
                    # Direct blit of the high-contrast wood texture
                    win.blit(IMAGES[texture_key], (draw_c * SQ_SIZE, draw_r * SQ_SIZE))
                
                # Add a warm wood tint to light squares if they appear too white
                if is_light:
                    tint = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    tint.fill((100, 60, 20, 25)) # Subtle warm orange/brown tint
                    win.blit(tint, (draw_c * SQ_SIZE, draw_r * SQ_SIZE))

                # Overlay wavy organic grain lines with HIGH contrast
                grain_surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                state = random.getstate()
                random.seed(r * 8 + c + 150) # Changed seed offset
                
                # High density and dark grain for light squares
                line_count = 22 if is_light else 12 
                alpha_min = 60 if is_light else 20 # Significantly higher alpha
                alpha_max = 90 if is_light else 45
                
                for _ in range(line_count):
                    points = []
                    start_x = random.randint(0, SQ_SIZE)
                    wobble = random.randint(8, 20)
                    
                    for i in range(6):
                        py = (i / 5) * SQ_SIZE
                        px = start_x + random.randint(-wobble, wobble)
                        points.append((px, py))
                    
                    alpha = random.randint(alpha_min, alpha_max)
                    if len(points) > 1:
                        # Richer darker fiber for light squares
                        g_color = (40, 20, 5, alpha) if is_light else (35, 15, 0, alpha)
                        pygame.draw.lines(grain_surf, g_color, False, points, random.randint(2, 3))
                
                # Stronger knots for visibility
                for _ in range(random.randint(0, 1)):
                    kx, ky = random.randint(10, SQ_SIZE-30), random.randint(10, SQ_SIZE-30)
                    kw, kh = random.randint(20, 40), random.randint(10, 20)
                    pygame.draw.ellipse(grain_surf, (45, 25, 10, 60 if is_light else 40), (kx, ky, kw, kh))
                    pygame.draw.ellipse(grain_surf, (30, 15, 5, 45 if is_light else 30), (kx+5, ky+2, kw-10, kh-4))
                
                win.blit(grain_surf, (draw_c * SQ_SIZE, draw_r * SQ_SIZE))
                random.setstate(state) 

            # Draw transparent overlay if selected or highlight
            if game.selected_pos == (r, c):
                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                overlay.fill((*SELECTED, 160)) # Increased from 128 for darker look
                win.blit(overlay, (draw_c * SQ_SIZE, draw_r * SQ_SIZE))
            elif (r, c) in game.valid_moves:
                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                overlay.fill((*HIGHLIGHT, 160))
                win.blit(overlay, (draw_c * SQ_SIZE, draw_r * SQ_SIZE))

            # Checkmate Highlight
            if game.game_over and game.winner and "Draw" not in game.winner:
                piece = game.board[r][c]
                loser = 'black' if game.winner == 'white' else 'white'
                if piece and isinstance(piece, King) and piece.color == loser:
                    overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    overlay.fill((*CHECKMATE_COLOR, 180)) # Increased from 150
                    win.blit(overlay, (draw_c * SQ_SIZE, draw_r * SQ_SIZE))

    # Draw Last Move Arrow
    if game.last_move:
        start_pos, end_pos = game.last_move
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        draw_r1 = 7 - r1 if flip else r1
        draw_c1 = 7 - c1 if flip else c1
        draw_r2 = 7 - r2 if flip else r2
        draw_c2 = 7 - c2 if flip else c2
        
        start_px = (draw_c1 * SQ_SIZE) + SQ_SIZE // 2
        start_py = (draw_r1 * SQ_SIZE) + SQ_SIZE // 2
        end_px = (draw_c2 * SQ_SIZE) + SQ_SIZE // 2
        end_py = (draw_r2 * SQ_SIZE) + SQ_SIZE // 2
        
        draw_arrow(win, (start_px, start_py), (end_px, end_py), ARROW_COLOR)

    # Draw Pieces
    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece:
                draw_r = 7 - r if flip else r
                draw_c = 7 - c if flip else c
                
                color_prefix = 'w' if piece.color == 'white' else 'b'
                type_map = {Pawn:'p', Rook:'r', Knight:'n', Bishop:'b', Queen:'q', King:'k'}
                piece_type = type_map.get(piece.__class__, 'p')
                key = color_prefix + piece_type
                if key in IMAGES:
                    win.blit(IMAGES[key], (draw_c * SQ_SIZE, draw_r * SQ_SIZE))
            
    draw_sidebar(win, game, font)

    # Check Indicator (if in check and not game over)
    if game.in_check and not game.game_over:
        check_font = pygame.font.SysFont("arial", 110, bold=True) # Increased from 60
        check_surf = check_font.render("CHECK!", True, CHECKMATE_COLOR)
        # Position in center of board
        check_rect = check_surf.get_rect(center=(BOARD_WIDTH // 2, BOARD_HEIGHT // 2))
        
        # Add a shadow/outline for better visibility
        shadow_surf = check_font.render("CHECK!", True, BLACK_COLOR)
        shadow_rect = shadow_surf.get_rect(center=(BOARD_WIDTH // 2 + 4, BOARD_HEIGHT // 2 + 4)) # Increased shadow offset
        
        win.blit(shadow_surf, shadow_rect)
        win.blit(check_surf, check_rect)

    # Game Over Message (Now Drawn Last on Top of Board)
    if game.game_over:
        res_font = pygame.font.SysFont("arial", 36, bold=True)
        winner_name = game.winner
        # Clean up name if it's "white" or "black"
        if winner_name == 'white' or winner_name == 'black':
             res_text = f"{winner_name.title()} Wins!"
        else:
             res_text = winner_name # It's a Draw message
        
        # Create text surface with background for visibility
        res_surf = res_font.render(res_text, True, WHITE_COLOR)
        
        # Background box
        padding = 20
        bg_rect = pygame.Rect(0, 0, res_surf.get_width() + padding * 2, res_surf.get_height() + padding * 2)
        bg_rect.center = (BOARD_WIDTH // 2, 50) # Top of Board
        
        pygame.draw.rect(win, (50, 50, 50), bg_rect, border_radius=10)
        pygame.draw.rect(win, WHITE_COLOR, bg_rect, 2, border_radius=10)
        
        res_rect = res_surf.get_rect(center=bg_rect.center)
        win.blit(res_surf, res_rect)
        
        # Draw Rating Stars and Feedback Message (Delayed by 4 seconds)
        current_time = pygame.time.get_ticks()
        if game.game_over_time and current_time - game.game_over_time >= 4000:
            feedback_font = pygame.font.SysFont("arial", 28, bold=True)
            msg_surf = feedback_font.render(game.performance_msg, True, (255, 255, 0))
            msg_rect = msg_surf.get_rect(center=(BOARD_WIDTH // 2, 120))
            
            # Draw background for feedback
            msg_bg = msg_rect.inflate(20, 10)
            pygame.draw.rect(win, (0, 0, 0, 150), msg_bg, border_radius=5)
            win.blit(msg_surf, msg_rect)
            
            # Star rating position
            star_y = 160
            start_x = (BOARD_WIDTH - (5 * 40)) // 2 + 20
            for i in range(5):
                color = (255, 215, 0) if i < game.performance_stars else (50, 50, 50)
                draw_star(win, (start_x + i * 40, star_y), 18, color)
                # Outline for stars
                outline_points = []
                for j in range(10):
                    angle = math.radians(j * 36 - 90)
                    radius = 18 if j % 2 == 0 else 9
                    outline_points.append((start_x + i * 40 + radius * math.cos(angle), star_y + radius * math.sin(angle)))
                pygame.draw.polygon(win, WHITE_COLOR, outline_points, 1)

def main():
    print("Initializing Pygame...")
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Python Chess (vs AI)')
    
    print("Loading images...")
    load_images()
    print("Images loaded successfully.")

    print("Loading sounds...")
    load_sounds()

    font = pygame.font.SysFont("arial", 40)

    game = None
    clock = pygame.time.Clock()
    
    MENU = "MENU"
    PLAYING = "PLAYING"
    state = MENU
    
    selected_difficulty = 2 # Medium by default
    dropdown_open = False
    theme_option_rects = []
    
    print("Starting game loop...")
    try:
        while True:
            if state == PLAYING and game:
                game.update_timers()
                
                if not game.game_over and not game.promotion_pending:
                     if game.turn != game.player_color and globals().get('AI_ENABLED', True):
                         if game.ai_move_result:
                             game.apply_ai_move()
                         elif not game.ai_thinking:
                             game.ai_move()
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if state == MENU:
                        # Handle Dropdown Clicks first to intercept
                        if dropdown_open:
                            clicked_on_option = False
                            for rect, idx in theme_option_rects:
                                if rect.collidepoint(event.pos):
                                    globals()['current_theme_index'] = idx
                                    print(f"Theme changed to: {BOARD_THEMES[idx][2]}")
                                    dropdown_open = False
                                    clicked_on_option = True
                                    break
                            if clicked_on_option: continue

                        # Difficulty Selection
                        if BTN_EASY_RECT.collidepoint(event.pos):
                            selected_difficulty = 1
                            print("Difficulty set to Easy")
                        elif BTN_MED_RECT.collidepoint(event.pos):
                            selected_difficulty = 2
                            print("Difficulty set to Medium")
                        elif BTN_HARD_RECT.collidepoint(event.pos):
                            selected_difficulty = 5
                            print("Difficulty set to Hard (Depth 5)")
                        elif BTN_EXPERT_RECT.collidepoint(event.pos):
                            selected_difficulty = 7
                            print("Difficulty set to Expert (Depth 7 - Ultra Strength!)")
                        
                        elif BTN_CLOCK_RECT.collidepoint(event.pos):
                            globals()['CLOCK_ENABLED'] = not globals().get('CLOCK_ENABLED', True)
                            print(f"Clock Enabled: {globals()['CLOCK_ENABLED']}")
                        
                        elif BTN_THEME_RECT.collidepoint(event.pos):
                            dropdown_open = not dropdown_open
                            print(f"Dropdown Open: {dropdown_open}")
                        
                        # Game Mode Selection
                        elif BTN_START_RECT.collidepoint(event.pos):
                            print("Starting Random Game...")
                            color = random.choice(['white', 'black'])
                            game = Game(player_color=color, ai_depth=selected_difficulty)
                            state = PLAYING
                            dropdown_open = False
                        elif BTN_WHITE_RECT.collidepoint(event.pos):
                            print("Starting Game as White...")
                            game = Game(player_color='white', ai_depth=selected_difficulty)
                            state = PLAYING
                            dropdown_open = False
                        elif BTN_BLACK_RECT.collidepoint(event.pos):
                            print("Starting Game as Black...")
                            game = Game(player_color='black', ai_depth=selected_difficulty)
                            state = PLAYING
                            dropdown_open = False
                        
                        # Close dropdown if clicked elsewhere
                        elif dropdown_open:
                            dropdown_open = False
                    
                    elif state == PLAYING:
                        # Handle Sidebar Buttons
                        if BTN_BOTH_RECT.collidepoint(event.pos):
                            globals()['AI_ENABLED'] = not globals().get('AI_ENABLED', True)
                            print(f"AI Enabled: {globals()['AI_ENABLED']}")
                            continue

                        if BTN_MENU_RECT.collidepoint(event.pos):
                            print("Returning to Menu...")
                            state = MENU
                            game = None
                            continue
                        
                        if BTN_RESTART_RECT.collidepoint(event.pos):
                            print("Restarting Game...")
                            color = game.player_color
                            game = Game(player_color=color, ai_depth=selected_difficulty)
                            continue
                            
                        if BTN_PAUSE_RECT.collidepoint(event.pos):
                            game.paused = not game.paused
                            print(f"Game Paused: {game.paused}")
                            # Reset last update time on resume to avoid time jumping
                            if not game.paused:
                                game.last_update_time = pygame.time.get_ticks()
                            continue

                        if BTN_SOUND_RECT.collidepoint(event.pos):
                            globals()['SOUND_ENABLED'] = not globals().get('SOUND_ENABLED', True)
                            print(f"Sound Enabled: {globals()['SOUND_ENABLED']}")
                            continue

                        if game.promotion_pending:
                             promo_options = draw_promotion_menu(win, game)
                             for rect, cls in promo_options:
                                 if rect.collidepoint(event.pos):
                                     game.promote_pawn(cls)
                        elif not game.game_over and (game.turn == game.player_color or not globals().get('AI_ENABLED', True)):
                            pos = pygame.mouse.get_pos()
                            
                            # Only handle clicks inside board area
                            if pos[0] < BOARD_WIDTH:
                                if game.player_color == 'black':
                                    c = 7 - (pos[0] // SQ_SIZE)
                                    r = 7 - (pos[1] // SQ_SIZE)
                                else:
                                    c = pos[0] // SQ_SIZE
                                    r = pos[1] // SQ_SIZE
                                    
                                game.select(r, c)
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: 
                        state = MENU
                        game = None
                    elif event.key == pygame.K_ESCAPE: 
                         state = MENU
                         game = None
                    elif event.key == pygame.K_p:
                         if state == PLAYING and game:
                             game.paused = not game.paused
                             if not game.paused:
                                 game.last_update_time = pygame.time.get_ticks()

            if state == MENU:
                theme_option_rects = draw_menu(win, font, selected_difficulty, dropdown_open)
            elif state == PLAYING and game:
                draw(win, game, font)
                if game.promotion_pending:
                    draw_promotion_menu(win, game)
                
            pygame.display.update()
            clock.tick(60)
            
    except KeyboardInterrupt:
        print("\nGame Terminated by User.")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
