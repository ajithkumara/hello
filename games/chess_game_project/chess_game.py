import pygame
import sys
import copy
import os
import random
import math
import socket
import threading
import queue
import argparse
import time
import json
import webbrowser
import urllib.parse
import smtplib
from email.message import EmailMessage

# --- Constants ---
BOARD_WIDTH, BOARD_HEIGHT = 640, 640
SIDEBAR_WIDTH = 250
WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
HEIGHT = BOARD_HEIGHT
SQ_SIZE = BOARD_WIDTH // 8

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
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
FONT_SIZE = 60
TIMER_SECONDS = 600 # 10 minutes

IMAGES = {}

# Menu Layout
BTN_WIDTH, BTN_HEIGHT = 420, 80
CENTER_X = WIDTH // 2 - BTN_WIDTH // 2
CENTER_Y = HEIGHT // 2
# Define buttons relative to window center
BTN_START_RECT = pygame.Rect(CENTER_X, CENTER_Y - 140, BTN_WIDTH, BTN_HEIGHT)
BTN_WHITE_RECT = pygame.Rect(CENTER_X, CENTER_Y - 40, BTN_WIDTH, BTN_HEIGHT)
BTN_BLACK_RECT = pygame.Rect(CENTER_X, CENTER_Y + 60, BTN_WIDTH, BTN_HEIGHT)
BTN_INVITE_RECT = pygame.Rect(CENTER_X, CENTER_Y + 140, 200, 50)

# Difficulty Buttons (Row)
DIFF_Y = CENTER_Y + 200
DIFF_WIDTH = 90
DIFF_HEIGHT = 50
BTN_EASY_RECT = pygame.Rect(CENTER_X, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_MED_RECT = pygame.Rect(CENTER_X + 105, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)
BTN_HARD_RECT = pygame.Rect(CENTER_X + 210, DIFF_Y, DIFF_WIDTH, DIFF_HEIGHT)

# Sidebar Buttons
BTN_PAUSE_RECT = pygame.Rect(BOARD_WIDTH + 25, 380, 200, 50)
BTN_RESTART_RECT = pygame.Rect(BOARD_WIDTH + 25, 450, 200, 50)
BTN_MENU_RECT = pygame.Rect(BOARD_WIDTH + 25, HEIGHT - 100, 200, 60)
BTN_BOTH_RECT = pygame.Rect(BOARD_WIDTH + 25, 320, 200, 50)
# Menu Create Game Button
BTN_CREATE_RECT = pygame.Rect(CENTER_X, DIFF_Y + 80, 200, 50)

# --- Classes ---

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
        # limit branching factor for deeper searches to speed up thinking
        # smaller number -> faster but weaker
        if self.depth >= 3:
            self.move_limit = 16
        elif self.depth == 2:
            self.move_limit = 28
        else:
            self.move_limit = None

    def evaluate_board(self, board):
        score = 0
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece:
                    val = piece.value
                    if piece.color == 'black':
                        val = -val
                    score += val
        return score

    def get_best_move(self, board, game_ref, ai_color):
        maximize = (ai_color == 'white')
        best_score = -float('inf') if maximize else float('inf')
        best_move = None
        
        all_moves = self.get_all_legal_moves_via_game_logic(board, ai_color, game_ref)
        # order candidate moves by simple capture heuristic (captures first)
        all_moves.sort(key=lambda mv: self.move_heuristic(board, mv), reverse=maximize)
        if self.move_limit:
            all_moves = all_moves[:self.move_limit]
        
        if not all_moves:
             return None 

        for start, end in all_moves:
            temp_board = self.deep_copy_board(board)
            self.simulate_apply_move(temp_board, start, end)
            
            score = self.minimax(temp_board, self.depth - 1, -float('inf'), float('inf'), not maximize, game_ref)
            
            if maximize:
                if score > best_score:
                    best_score = score
                    best_move = (start, end)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (start, end)
                
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player, game_ref):
        if depth == 0:
            return self.evaluate_board(board)

        if maximizing_player: 
            max_eval = -float('inf')
            moves = self.get_all_legal_moves_via_game_logic(board, 'white', game_ref)
            # move ordering: prefer captures and high-value captures
            moves.sort(key=lambda mv: self.move_heuristic(board, mv), reverse=True)
            if self.move_limit:
                moves = moves[:self.move_limit]
            if not moves:
                return self.evaluate_board(board)
            
            for start, end in moves:
                temp_board = self.deep_copy_board(board)
                self.simulate_apply_move(temp_board, start, end)
                eval = self.minimax(temp_board, depth - 1, alpha, beta, False, game_ref)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        
        else: 
            min_eval = float('inf')
            moves = self.get_all_legal_moves_via_game_logic(board, 'black', game_ref)
            moves.sort(key=lambda mv: self.move_heuristic(board, mv))
            if self.move_limit:
                moves = moves[:self.move_limit]
            if not moves:
                return self.evaluate_board(board)
            
            for start, end in moves:
                temp_board = self.deep_copy_board(board)
                self.simulate_apply_move(temp_board, start, end)
                eval = self.minimax(temp_board, depth - 1, alpha, beta, True, game_ref)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_all_legal_moves_via_game_logic(self, board, color, game_ref):
        moves = []
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and piece.color == color:
                    # NOTE: We can't use game_ref directly here because game_ref might rely on internal state (en_passant)
                    # that is not in the 'board' passed to it. 
                    # However, for simplicity in this implementation, we will trust game_ref's get_legal_moves
                    # but we must be careful. 
                    # Ideally, AI simulation should carry full game state clones.
                    # For now, we only deep copy the BOARD, not the whole GAME state.
                    # This means AI might not "see" en passant in future depths properly, 
                    # but it will see it for the immediate move if game_ref has it set.
                    valid_moves = game_ref.get_legal_moves(piece, (r, c), board)
                    for move in valid_moves:
                        moves.append(((r, c), move))
        return moves

    def move_heuristic(self, board, move):
        # Simple heuristic: prefer captures, higher-value captures first
        start, end = move
        r2, c2 = end
        target = board[r2][c2]
        if target:
            return target.value
        return 0

    def deep_copy_board(self, board):
        return copy.deepcopy(board)

    def simulate_apply_move(self, board, start, end):
        r1, c1 = start
        r2, c2 = end
        piece = board[r1][c1]
        
        if not piece: return # Should not happen

        # Castling Update
        if isinstance(piece, King) and abs(c2 - c1) == 2:
            if c2 == 6: 
                rook = board[r1][7]
                board[r1][5] = rook
                board[r1][7] = None
                if rook: rook.has_moved = True
            elif c2 == 2:
                rook = board[r1][0]
                board[r1][3] = rook
                board[r1][0] = None
                if rook: rook.has_moved = True

        # AI Queen Promotion
        if isinstance(piece, Pawn):
             if (piece.color == 'white' and r2 == 0) or (piece.color == 'black' and r2 == 7):
                 piece = Queen(piece.color, 'Queen')

        board[r2][c2] = piece
        board[r1][c1] = None
        piece.has_moved = True


class NetworkHandler:
    """Simple TCP network handler for sending/receiving moves.
    Usage: host starts a server; client connects. Messages are newline-terminated.
    """
    def __init__(self, is_host=False, host='', port=5000):
        self.is_host = is_host
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.connected = False
        self.recv_queue = queue.Queue()
        self._stop_event = threading.Event()

    def start(self):
        if self.is_host:
            threading.Thread(target=self._start_server, daemon=True).start()
        else:
            threading.Thread(target=self._start_client, daemon=True).start()

    def _start_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host if self.host else '', self.port))
            self.sock.listen(1)
            print(f"Hosting on {socket.gethostbyname(socket.gethostname())}:{self.port} (waiting for connection)")
            self.conn, addr = self.sock.accept()
            print(f"Client connected: {addr}")
            self.connected = True
            self._recv_loop(self.conn)
        except Exception as e:
            print("Network server error:", e)
            self.close()

    def _start_client(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.conn = self.sock
            self.connected = True
            print(f"Connected to host {self.host}:{self.port}")
            self._recv_loop(self.conn)
        except Exception as e:
            print("Network client error:", e)
            self.close()

    def _recv_loop(self, conn):
        buf = b''
        try:
            while not self._stop_event.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    try:
                        text = line.decode('utf-8')
                        self.recv_queue.put(text)
                    except Exception:
                        pass
        except Exception as e:
            print('Network receive error:', e)
        finally:
            self.close()

    def send(self, text):
        try:
            if self.connected and self.conn:
                self.conn.sendall((text + "\n").encode('utf-8'))
        except Exception as e:
            print('Network send error:', e)

    def send_move(self, r1, c1, r2, c2):
        self.send(f"MOVE {r1},{c1},{r2},{c2}")

    def close(self):
        self._stop_event.set()
        self.connected = False
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass


class ServerClient:
    """Client for the authoritative lobby/matchmaking server (JSON line protocol)."""
    def __init__(self, host='localhost', port=5000, name='Player'):
        self.host = host
        self.port = port
        self.name = name
        self.sock = None
        self.recv_thread = None
        self.recv_queue = queue.Queue()
        self.connected = False
        self._stop = threading.Event()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.connected = True
            self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self.recv_thread.start()
            # send auth
            self.send({'type': 'auth', 'name': self.name})
            return True
        except Exception as e:
            print('Server connect error:', e)
            self.close()
            return False

    def _recv_loop(self):
        f = self.sock.makefile('rb')
        try:
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode('utf-8').strip())
                    self.recv_queue.put(msg)
                except Exception:
                    pass
        except Exception as e:
            print('Server recv error', e)
        finally:
            self.close()

    def send(self, obj):
        try:
            if self.connected and self.sock:
                data = (json.dumps(obj) + '\n').encode('utf-8')
                self.sock.sendall(data)
        except Exception as e:
            print('Server send error', e)

    def close(self):
        self._stop.set()
        self.connected = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

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
        self.in_check = False # Track check status
        self.full_move_number = 0 # Start at move 0
        
        # Draw Rules
        self.board_history = {} # Threefold repetition

        # Half-move clock (number of half-moves since last pawn move or capture)
        # Used for 50-move draw rule (50 full moves = 100 half-moves)
        self.half_move_clock = 0

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
        
        if self.game_over or self.promotion_pending or self.paused: 
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
                    # If a network connection exists, send the move to opponent
                    net = globals().get('NETWORK')
                    if net and getattr(net, 'connected', False):
                        try:
                            if self.last_move:
                                (sr, sc), (er, ec) = self.last_move
                                net.send_move(sr, sc, er, ec)
                        except Exception:
                            pass
                    # If playing via central server, send the move there as well
                    try:
                        srv_gid = globals().get('current_server_game')
                        server = globals().get('SERVER')
                        if srv_gid and server and getattr(server, 'connected', False) and self.last_move:
                            (sr, sc), (er, ec) = self.last_move
                            mvtxt = f"{sr},{sc},{er},{ec}"
                            server.send({'type': 'move', 'game_id': srv_gid, 'move': mvtxt})
                            cms = globals().get('chat_messages', [])
                            cms.append(f"[move] {mvtxt}")
                            globals()['chat_messages'] = cms[-200:]
                    except Exception:
                        pass
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

        # 50-move rule: if 100 half-moves (50 full moves) have occurred without a pawn move or capture
        elif getattr(self, 'half_move_clock', 0) >= 100:
            self.game_over = True
            self.winner = 'Draw (50-move rule)'

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
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p and p.color == enemy_color:
                    # Note: We pass None for game here to avoid infinite recursion or complex dependency,
                    # and typical attacks don't depend on en passant for check detection usually, but strictly speaking they could.
                    # For basic check logic, pseudo moves are fine.
                    if pos in p.get_valid_moves((r, c), board, None): 
                        return True
        return False

    def simulate_move(self, start, end, color, board):
        r1, c1 = start
        r2, c2 = end
        target = board[r2][c2]
        piece = board[r1][c1]
        
        # Store en passant state if needed to restore (simpler simulation just checks king safety)
        # We don't fully simulate detailed en passant capture removal here for simple check validation
        # because the 'target' square is empty in EP. 
        # But if we want 100% accuracy, we should remove the pawn.
        # For this simple engine, we'll assume displacing the piece is enough to check for check.
        
        board[r2][c2] = piece
        board[r1][c1] = None
        
        in_check = self.is_check(color, board)
        
        board[r1][c1] = piece
        board[r2][c2] = target
        
        return in_check

    def is_check(self, color, board):
        king_pos = None
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p and p.color == color and isinstance(p, King):
                    king_pos = (r, c)
                    break
            if king_pos: break
            
        if not king_pos: return False 
        return self.is_square_attacked(king_pos, color, board)

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
        if self.game_over or self.paused: return
        print("AI Thinking...")
        pygame.display.set_caption("Python Chess (AI Thinking...)")
        pygame.event.pump() 
        
        move = self.ai.get_best_move(self.board, self, self.turn)
        if move:
            start, end = move
            self.move(start, end)
            self.switch_turn()
        else:
            # If AI has no moves, check logic handles Game Over
            self.switch_turn() # Will trigger game over check in switch_turn
            if not self.game_over: # Should be game over if no moves
                 print("AI Resigns (Error?)")
                 self.game_over = True
                 self.winner = 'white' if self.turn == 'black' else 'black'
        
        pygame.display.set_caption("Python Chess (vs AI)")


# --- GUI ---

def load_images():
    pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    for p in pieces:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "images", p + ".png")
        try:
            image = pygame.image.load(path)
            IMAGES[p] = pygame.transform.scale(image, (SQ_SIZE, SQ_SIZE))
        except Exception as e:
            # If image missing or cannot be loaded, create a simple placeholder surface
            print(f"Warning: couldn't load image {path}: {e}")
            surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            surf.fill((120, 120, 120))
            # draw a simple letter to indicate piece type
            try:
                font = pygame.font.SysFont('arial', SQ_SIZE // 2, bold=True)
                label = p[1].upper()
                text_surf = font.render(label, True, (255, 255, 255))
                tr = text_surf.get_rect(center=(SQ_SIZE // 2, SQ_SIZE // 2))
                surf.blit(text_surf, tr)
            except Exception:
                pass
            IMAGES[p] = surf

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
    ind_font = pygame.font.SysFont("arial", 30)
    ind_surf = ind_font.render(turn_text, True, color)
    ind_rect = ind_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 100))
    win.blit(ind_surf, ind_rect)

    # Move Counter
    move_font = pygame.font.SysFont("arial", 22)
    move_text = f"Move: {game.full_move_number}"
    move_surf = move_font.render(move_text, True, (200, 200, 200))
    move_rect = move_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 140))
    win.blit(move_surf, move_rect)

    # Half-move clock display (for 50-move draw rule)
    half_text = f"Half-Moves: {getattr(game, 'half_move_clock', 0)}"
    half_surf = move_font.render(half_text, True, (200, 200, 200))
    half_rect = half_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 170))
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

    

    # Timers
    # Black Time (Top)
    b_time_str = format_time(game.black_time)
    b_surf = font.render(b_time_str, True, WHITE_COLOR)
    b_rect = b_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 210))
    pygame.draw.rect(win, BLACK_COLOR, b_rect.inflate(20, 10), border_radius=5)
    win.blit(b_surf, b_rect)
    
    label_font = pygame.font.SysFont("arial", 20)
    b_label = label_font.render("Black", True, (200, 200, 200))
    win.blit(b_label, (BOARD_WIDTH + 20, 190))

    # White Time (Below Black)
    w_time_str = format_time(game.white_time)
    w_surf = font.render(w_time_str, True, WHITE_COLOR)
    w_rect = w_surf.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 290))
    pygame.draw.rect(win, BLACK_COLOR, w_rect.inflate(20, 10), border_radius=5)
    win.blit(w_surf, w_rect)

    w_label = label_font.render("White", True, (200, 200, 200))
    win.blit(w_label, (BOARD_WIDTH + 20, 270))

    # Controls (Pause / Restart) - Shifted Down
    
    # Pause Button
    BTN_PAUSE_RECT.y = 380
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
    BTN_RESTART_RECT.y = 450
    r_color = BUTTON_COLOR if not BTN_RESTART_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, r_color, BTN_RESTART_RECT, border_radius=10)
    pygame.draw.rect(win, BLACK_COLOR, BTN_RESTART_RECT, 3, border_radius=10)
    r_surf = font.render("Restart", True, WHITE_COLOR)
    r_rect = r_surf.get_rect(center=BTN_RESTART_RECT.center)
    win.blit(r_surf, r_rect)
    
    # Draw Menu Button in Sidebar
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

    # Move List (from server or local)
    moves_rect = pygame.Rect(BOARD_WIDTH + 10, 330, SIDEBAR_WIDTH - 20, 120)
    pygame.draw.rect(win, (40, 40, 40), moves_rect, border_radius=6)
    pygame.draw.rect(win, (80, 80, 80), moves_rect, 2, border_radius=6)
    mlist_font = pygame.font.SysFont('arial', 16)
    # Gather moves: prefer server-provided chat_messages entries starting with [move]
    chat_msgs = globals().get('chat_messages', [])
    move_lines = [line for line in chat_msgs if line.startswith('[move]')]
    if not move_lines and game.last_move:
        sm, em = game.last_move
        move_lines = [f"[move] {sm[0]},{sm[1]},{em[0]},{em[1]}"]
    for i, line in enumerate(move_lines[-6:]):
        txt = line.replace('[move] ', '')
        win.blit(mlist_font.render(txt, True, WHITE_COLOR), (moves_rect.x + 6, moves_rect.y + 6 + i * 18))

    # Chat area
    chat_rect = pygame.Rect(BOARD_WIDTH + 10, 460, SIDEBAR_WIDTH - 20, 150)
    pygame.draw.rect(win, (30, 30, 30), chat_rect, border_radius=6)
    pygame.draw.rect(win, (80, 80, 80), chat_rect, 2, border_radius=6)
    c_font = pygame.font.SysFont('arial', 14)
    msgs = globals().get('chat_messages', [])
    for i, line in enumerate(msgs[-6:]):
        win.blit(c_font.render(line, True, WHITE_COLOR), (chat_rect.x + 6, chat_rect.y + 6 + i * 20))

    # Input box
    input_rect = pygame.Rect(chat_rect.x + 6, chat_rect.y + chat_rect.h - 32, chat_rect.w - 12, 26)
    pygame.draw.rect(win, (255, 255, 255), input_rect, 2, border_radius=4)
    # expose rects for event handling
    globals()['CHAT_INPUT_RECT'] = input_rect
    globals()['CHAT_AREA_RECT'] = chat_rect
    # render current input
    cur_input = globals().get('chat_input', '')
    win.blit(c_font.render(cur_input[-30:], True, WHITE_COLOR), (input_rect.x + 6, input_rect.y + 3))

def draw_menu(win, font, current_difficulty):
    win.fill(LIGHT_SQ) 
    
    title_font = pygame.font.SysFont("arial", 80, bold=True)
    title_surf = title_font.render("CHESS", True, BLACK_COLOR)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 5))
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
    # Invite by email (opens default mail client)
    inv_color = BUTTON_COLOR if not BTN_INVITE_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
    pygame.draw.rect(win, inv_color, BTN_INVITE_RECT, border_radius=8)
    pygame.draw.rect(win, BLACK_COLOR, BTN_INVITE_RECT, 2, border_radius=8)
    inv_font = pygame.font.SysFont('arial', 20)
    win.blit(inv_font.render('Invite by Email', True, WHITE_COLOR), (BTN_INVITE_RECT.x + 16, BTN_INVITE_RECT.y + 12))
    # Invite modal
    if globals().get('invite_mode'):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,180))
        win.blit(overlay, (0,0))
        modal_w, modal_h = 520, 220
        mx = (WIDTH - modal_w) // 2
        my = (HEIGHT - modal_h) // 2
        modal = pygame.Rect(mx, my, modal_w, modal_h)
        pygame.draw.rect(win, (240,240,240), modal, border_radius=8)
        pygame.draw.rect(win, BLACK_COLOR, modal, 2, border_radius=8)
        mh_font = pygame.font.SysFont('arial', 26, bold=True)
        win.blit(mh_font.render('Invite by Email', True, BLACK_COLOR), (mx + 16, my + 12))
        info_font = pygame.font.SysFont('arial', 18)
        win.blit(info_font.render('Recipient email:', True, BLACK_COLOR), (mx + 16, my + 56))
        # input box
        in_rect = pygame.Rect(mx + 16, my + 86, modal_w - 32, 36)
        pygame.draw.rect(win, WHITE_COLOR, in_rect)
        pygame.draw.rect(win, BLACK_COLOR, in_rect, 2)
        email_text = globals().get('invite_input', '')
        win.blit(info_font.render(email_text[-48:], True, BLACK_COLOR), (in_rect.x + 8, in_rect.y + 6))
        # Buttons
        send_rect = pygame.Rect(mx + 80, my + modal_h - 56, 120, 36)
        cancel_rect = pygame.Rect(mx + modal_w - 200, my + modal_h - 56, 120, 36)
        mpos = pygame.mouse.get_pos()
        s_color = BUTTON_HOVER_COLOR if send_rect.collidepoint(mpos) else BUTTON_COLOR
        c_color = BUTTON_HOVER_COLOR if cancel_rect.collidepoint(mpos) else BUTTON_COLOR
        pygame.draw.rect(win, s_color, send_rect, border_radius=6)
        pygame.draw.rect(win, c_color, cancel_rect, border_radius=6)
        win.blit(info_font.render('Send', True, WHITE_COLOR), send_rect.move(42,6))
        win.blit(info_font.render('Cancel', True, WHITE_COLOR), cancel_rect.move(32,6))
        # status
        status = globals().get('invite_status','')
        win.blit(info_font.render(status, True, (80,80,80)), (mx + 16, my + modal_h - 96))
    
    # Draw Difficulty Buttons
    diff_font = pygame.font.SysFont("arial", 30)
    
    # Label
    l_surf = diff_font.render("Difficulty:", True, BLACK_COLOR)
    win.blit(l_surf, (BTN_EASY_RECT.x, BTN_EASY_RECT.y - 40))
    
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
    draw_diff_btn(BTN_HARD_RECT, "Hard", 3)

    # If connected to central server, show lobby
    server = globals().get('SERVER')
    mouse_pos = pygame.mouse.get_pos()
    if server and getattr(server, 'connected', False):
        # Status
        s_font = pygame.font.SysFont('arial', 20)
        status = f"Server: {server.host}:{server.port} (Connected)"
        s_surf = s_font.render(status, True, BLACK_COLOR)
        win.blit(s_surf, (20, HEIGHT - 40))

        # Draw Create Game button
        c_color = BUTTON_COLOR if not BTN_CREATE_RECT.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR
        pygame.draw.rect(win, c_color, BTN_CREATE_RECT, border_radius=8)
        pygame.draw.rect(win, BLACK_COLOR, BTN_CREATE_RECT, 2, border_radius=8)
        c_surf = s_font.render('Create Game', True, WHITE_COLOR)
        win.blit(c_surf, c_surf.get_rect(center=BTN_CREATE_RECT.center))

        # Draw a short list of games on right side
        list_x = CENTER_X + BTN_WIDTH + 20
        list_y = CENTER_Y - 140
        gw = 300
        gh = 40
        g_font = pygame.font.SysFont('arial', 18)
        games = globals().get('server_game_list', [])
        win.blit(g_font.render('Server Lobby', True, BLACK_COLOR), (list_x, list_y - 30))
        for i, g in enumerate(games[:8]):
            gy = list_y + i * (gh + 8)
            rect = pygame.Rect(list_x, gy, gw, gh)
            pygame.draw.rect(win, BUTTON_COLOR if not rect.collidepoint(mouse_pos) else BUTTON_HOVER_COLOR, rect, border_radius=6)
            pygame.draw.rect(win, BLACK_COLOR, rect, 2, border_radius=6)
            txt = f"Game {g.get('game_id')} {'W' if g.get('has_white') else '_'}/{ 'B' if g.get('has_black') else '_'} S:{'Yes' if g.get('started') else 'No'}"
            win.blit(g_font.render(txt, True, WHITE_COLOR), (list_x + 8, gy + 8))

    def build_invite_text():
        server = globals().get('SERVER')
        net = globals().get('NETWORK')
        subject = 'Chess Game Invitation'
        body = 'I created a chess game.'
        if server and getattr(server, 'connected', False) and globals().get('current_server_game'):
            gid = globals().get('current_server_game')
            host = server.host
            port = server.port
            body = f"Join my chess game on the server:\nServer: {host}:{port}\nGame ID: {gid}\nRun: python \"chess_game.py\" --server {host}:{port} --name YOURNAME and join game {gid}."
        elif net and getattr(net, 'connected', False) and getattr(net, 'is_host', False):
            try:
                host_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                host_ip = 'HOST_IP'
            body = f"Join my chess game (LAN):\nHost: {host_ip}\nPort: {net.port}\nRun: python \"chess_game.py\" --net-join {host_ip} --net-port {net.port}"
        return subject, body


    def send_invite_email(recipient, smtp_config):
        """Attempt to send invite via SMTP using smtp_config. Returns True on success."""
        subject, body = build_invite_text()
        server = smtp_config.get('server')
        port = smtp_config.get('port')
        user = smtp_config.get('user')
        pwd = smtp_config.get('pass')
        use_tls = smtp_config.get('tls')
        if not server:
            return False
        try:
            msg = EmailMessage()
            msg['From'] = user if user else 'noreply@example.com'
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.set_content(body)

            if use_tls:
                s = smtplib.SMTP(server, port, timeout=10)
                s.ehlo()
                s.starttls()
                s.ehlo()
                if user and pwd:
                    s.login(user, pwd)
                s.send_message(msg)
                s.quit()
            else:
                # try SSL first on 465
                try:
                    s = smtplib.SMTP_SSL(server, port, timeout=10)
                    if user and pwd:
                        s.login(user, pwd)
                    s.send_message(msg)
                    s.quit()
                except Exception:
                    # fallback to plain SMTP
                    s = smtplib.SMTP(server, port, timeout=10)
                    if user and pwd:
                        s.login(user, pwd)
                    s.send_message(msg)
                    s.quit()
            return True
        except Exception as e:
            print('SMTP send error:', e)
            return False


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
    for r in range(8):
        for c in range(8):
            draw_r = 7 - r if flip else r
            draw_c = 7 - c if flip else c
            
            color = LIGHT_SQ if (r + c) % 2 == 0 else DARK_SQ
            if game.selected_pos == (r, c):
                color = SELECTED
            elif (r, c) in game.valid_moves:
                color = HIGHLIGHT
                
            # Checkmate Highlight
            if game.game_over and game.winner:
                # Highlight loser's King in Red
                # If draw, maybe highlight both? Or neither. 
                # For now, only highlight if there is a 'loser' (i.e. not a Draw string)
                if "Draw" not in game.winner:
                     piece = game.board[r][c]
                     loser = 'black' if game.winner == 'white' else 'white'
                     if piece and isinstance(piece, King) and piece.color == loser:
                          pygame.draw.rect(win, CHECKMATE_COLOR, (draw_c * SQ_SIZE, draw_r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                          color = CHECKMATE_COLOR 
            
            pygame.draw.rect(win, color, (draw_c * SQ_SIZE, draw_r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

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
        check_font = pygame.font.SysFont("arial", 60, bold=True)
        check_surf = check_font.render("CHECK!", True, CHECKMATE_COLOR)
        # Position in center of board
        check_rect = check_surf.get_rect(center=(BOARD_WIDTH // 2, BOARD_HEIGHT // 2))
        
        # Add a shadow/outline for better visibility
        shadow_surf = check_font.render("CHECK!", True, BLACK_COLOR)
        shadow_rect = shadow_surf.get_rect(center=(BOARD_WIDTH // 2 + 2, BOARD_HEIGHT // 2 + 2))
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-host', action='store_true', help='Start as network host')
    parser.add_argument('--net-join', metavar='HOST', help='Join network host at HOST')
    parser.add_argument('--net-port', type=int, default=5000, help='Network port')
    parser.add_argument('--net-color', choices=['white','black'], help='Pick color for network host (host only)')
    parser.add_argument('--server', metavar='SERVER', help='Connect to central server (host:port or host)')
    parser.add_argument('--name', default='Player', help='Your player name for server lobby')
    parser.add_argument('--smtp-server', help='SMTP server for sending invites (e.g. smtp.gmail.com)')
    parser.add_argument('--smtp-port', type=int, default=587, help='SMTP port (starttls)')
    parser.add_argument('--smtp-user', help='SMTP username')
    parser.add_argument('--smtp-pass', help='SMTP password')
    parser.add_argument('--smtp-tls', action='store_true', help='Use STARTTLS for SMTP')
    args = parser.parse_args()

    print("Initializing Pygame...")
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Python Chess (vs AI)')
    
    print("Loading images...")
    load_images()
    print("Images loaded successfully.")

    font = pygame.font.SysFont("arial", 40)

    game = None
    clock = pygame.time.Clock()
    
    MENU = "MENU"
    PLAYING = "PLAYING"
    state = MENU
    
    selected_difficulty = 2 # Medium by default
    server_client = None
    globals()['server_game_list'] = []
    globals()['chat_messages'] = []
    chat_messages = globals()['chat_messages']
    chat_input = ''
    chat_active = False
    globals()['invite_mode'] = False
    globals()['invite_input'] = ''
    globals()['invite_status'] = ''
    smtp_config = {'server': args.smtp_server, 'port': args.smtp_port, 'user': args.smtp_user, 'pass': args.smtp_pass, 'tls': args.smtp_tls}
    globals()['current_server_game'] = None
    
    # Setup peer-to-peer networking if requested
    if args.net_host or args.net_join:
        net = None
        if args.net_host:
            net = NetworkHandler(is_host=True, host='', port=args.net_port)
            globals()['AI_ENABLED'] = False
            preferred_color = args.net_color if args.net_color else 'white'
            print('Starting network host...')
            net.start()
            # wait for client to connect
            print('Waiting for client to connect...')
            while not getattr(net, 'connected', False):
                time.sleep(0.1)
            print('Client connected.')
            globals()['NETWORK'] = net
            # auto-start a network game: host plays preferred_color
            game = Game(player_color=preferred_color, ai_depth=selected_difficulty)
            state = PLAYING
            # inform client of host color so client can pick opposite
            try:
                net.send(f"START {preferred_color}")
            except Exception:
                pass
        else:
            host = args.net_join
            net = NetworkHandler(is_host=False, host=host, port=args.net_port)
            globals()['AI_ENABLED'] = False
            print(f'Connecting to host {host}:{args.net_port}...')
            net.start()
            # wait for connection
            while not getattr(net, 'connected', False):
                time.sleep(0.1)
            print('Connected to host.')
            globals()['NETWORK'] = net
            # wait for START message from host to know host color
            host_color = None
            print('Waiting for START message from host...')
            while True:
                try:
                    if not net.recv_queue.empty():
                        msg = net.recv_queue.get_nowait()
                        if msg.startswith('START'):
                            parts = msg.split()
                            if len(parts) >= 2:
                                host_color = parts[1]
                                break
                    time.sleep(0.05)
                except Exception:
                    time.sleep(0.05)

            if host_color:
                client_color = 'black' if host_color == 'white' else 'white'
            else:
                client_color = 'black'

            game = Game(player_color=client_color, ai_depth=selected_difficulty)
            state = PLAYING

    # Setup central server connection if requested
    if args.server:
        host_port = args.server
        host = host_port
        port = 5000
        if ':' in host_port:
            host, port_s = host_port.split(':', 1)
            try:
                port = int(port_s)
            except Exception:
                port = 5000
        server_client = ServerClient(host=host, port=port, name=args.name)
        ok = server_client.connect()
        if ok:
            globals()['SERVER'] = server_client
        else:
            server_client = None

    print("Starting game loop...")
    try:
        while True:
            # Process incoming peer-to-peer network messages (if any)
            net = globals().get('NETWORK')
            if net and getattr(net, 'connected', False) and game:
                try:
                    while not net.recv_queue.empty():
                        msg = net.recv_queue.get_nowait()
                        if msg.startswith('MOVE'):
                            try:
                                parts = msg.split()[1].split(',')
                                r1, c1, r2, c2 = map(int, parts)
                                game.move((r1, c1), (r2, c2))
                                game.switch_turn()
                            except Exception as e:
                                print('Bad MOVE message:', msg, e)
                except Exception:
                    pass

            # Process incoming server messages (if connected to central server)
            server = globals().get('SERVER')
            if server and getattr(server, 'connected', False):
                try:
                    while not server.recv_queue.empty():
                        msg = server.recv_queue.get_nowait()
                        mtype = msg.get('type')
                        if mtype == 'game_list':
                            server_game_list = msg.get('games', [])
                        elif mtype == 'game_created':
                            # request updated list (server will broadcast)
                            pass
                        elif mtype == 'game_start':
                            gid = msg.get('game_id')
                            white = msg.get('white')
                            black = msg.get('black')
                            # determine our color
                            myname = server.name
                            player_color = 'white' if white == myname else ('black' if black == myname else None)
                            if player_color:
                                game = Game(player_color=player_color, ai_depth=selected_difficulty)
                                state = PLAYING
                        elif mtype == 'move':
                            gid = msg.get('game_id')
                            mv = msg.get('move')
                            # add to chat_messages as move for UI
                            cms = globals().get('chat_messages', [])
                            cms.append(f"[move] {mv}")
                            globals()['chat_messages'] = cms[-200:]
                            try:
                                parts = mv.split(',')
                                r1, c1, r2, c2 = map(int, parts)
                                if game:
                                    game.move((r1, c1), (r2, c2))
                                    game.switch_turn()
                            except Exception as e:
                                print('Bad server MOVE', mv, e)
                        elif mtype == 'chat':
                            chat_messages.append(f"{msg.get('from')}: {msg.get('text')}")
                            # keep last 50
                            chat_messages = chat_messages[-50:]
                        elif mtype == 'spectate_ok':
                            moves = msg.get('moves', [])
                            # apply moves to move list only (do not simulate board changes here)
                            for m in moves:
                                mv = m.get('move')
                                chat_messages.append(f"[move] {mv}")
                        elif mtype == 'game_over':
                            if game:
                                game.game_over = True
                                game.winner = msg.get('winner')
                        # other messages ignored for now
                except Exception:
                    pass

            if state == PLAYING and game:
                game.update_timers()
                
                if not game.game_over and not game.promotion_pending:
                     if game.turn != game.player_color and globals().get('AI_ENABLED', True):
                         game.ai_move()
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if state == MENU:
                        # Difficulty Selection
                        if BTN_EASY_RECT.collidepoint(event.pos):
                            selected_difficulty = 1
                            print("Difficulty set to Easy")
                        elif BTN_MED_RECT.collidepoint(event.pos):
                            selected_difficulty = 2
                            print("Difficulty set to Medium")
                        elif BTN_HARD_RECT.collidepoint(event.pos):
                            selected_difficulty = 3
                            print("Difficulty set to Hard")
                        
                        # Game Mode Selection
                        elif BTN_START_RECT.collidepoint(event.pos):
                            print("Starting Random Game...")
                            color = random.choice(['white', 'black'])
                            game = Game(player_color=color, ai_depth=selected_difficulty)
                            state = PLAYING
                        elif BTN_WHITE_RECT.collidepoint(event.pos):
                            print("Starting Game as White...")
                            game = Game(player_color='white', ai_depth=selected_difficulty)
                            state = PLAYING
                        elif BTN_BLACK_RECT.collidepoint(event.pos):
                            print("Starting Game as Black...")
                            game = Game(player_color='black', ai_depth=selected_difficulty)
                            state = PLAYING
                        # Invite button
                        if BTN_INVITE_RECT.collidepoint(event.pos):
                            # Open invite modal to enter recipient email
                            globals()['invite_mode'] = True
                            globals()['invite_input'] = ''
                            globals()['invite_status'] = ''
                        # Server lobby interactions
                        server = globals().get('SERVER')
                        if server and getattr(server, 'connected', False):
                            # Create Game
                            if BTN_CREATE_RECT.collidepoint(event.pos):
                                server.send({'type': 'create_game', 'color': 'white', 'time': TIMER_SECONDS, 'inc': 0})
                            # Click on listed games to join
                            list_x = CENTER_X + BTN_WIDTH + 20
                            list_y = CENTER_Y - 140
                            gw = 300
                            gh = 40
                            games = globals().get('server_game_list', [])
                            for i, g in enumerate(games[:8]):
                                rect = pygame.Rect(list_x, list_y + i * (gh + 8), gw, gh)
                                if rect.collidepoint(event.pos):
                                    gid = g.get('game_id')
                                    server.send({'type': 'join_game', 'game_id': gid})
                                    globals()['current_server_game'] = gid
                                    break
                        # Invite modal click handling
                        if globals().get('invite_mode'):
                            # compute modal rects same as draw
                            modal_w, modal_h = 520, 220
                            mx = (WIDTH - modal_w) // 2
                            my = (HEIGHT - modal_h) // 2
                            in_rect = pygame.Rect(mx + 16, my + 86, modal_w - 32, 36)
                            send_rect = pygame.Rect(mx + 80, my + modal_h - 56, 120, 36)
                            cancel_rect = pygame.Rect(mx + modal_w - 200, my + modal_h - 56, 120, 36)
                            if in_rect.collidepoint(event.pos):
                                globals()['invite_focus'] = True
                            else:
                                globals()['invite_focus'] = False
                            if send_rect.collidepoint(event.pos):
                                # attempt to send email
                                recip = globals().get('invite_input','').strip()
                                if recip:
                                    sent = False
                                    try:
                                        sent = send_invite_email(recip, smtp_config)
                                    except Exception as e:
                                        print('Invite send error', e)
                                        sent = False
                                    if sent:
                                        globals()['invite_status'] = 'Sent successfully.'
                                    else:
                                        globals()['invite_status'] = 'Failed to send; opened mail client instead.'
                                        # fallback to mailto
                                        try:
                                            subject, body = build_invite_text()
                                            mailto = f"mailto:{urllib.parse.quote(recip)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
                                            webbrowser.open(mailto)
                                        except Exception:
                                            pass
                            if cancel_rect.collidepoint(event.pos):
                                globals()['invite_mode'] = False
                                globals()['invite_input'] = ''
                                globals()['invite_status'] = ''
                    
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

                        if game.promotion_pending:
                            promo_options = draw_promotion_menu(win, game)
                            for rect, cls in promo_options:
                                if rect.collidepoint(event.pos):
                                    game.promote_pawn(cls)
                        else:
                            # Chat input focus
                            chat_input_rect = globals().get('CHAT_INPUT_RECT')
                            if chat_input_rect and chat_input_rect.collidepoint(event.pos):
                                globals()['chat_active'] = True
                            else:
                                globals()['chat_active'] = False

                            if not game.game_over and (game.turn == game.player_color or not globals().get('AI_ENABLED', True)):
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
                    # Invite modal typing
                    if globals().get('invite_mode') and globals().get('invite_focus'):
                        s = globals().get('invite_input','')
                        if event.key == pygame.K_BACKSPACE:
                            s = s[:-1]
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # press Enter -> try send
                            recip = s.strip()
                            if recip:
                                try:
                                    sent = send_invite_email(recip, smtp_config)
                                except Exception:
                                    sent = False
                                if sent:
                                    globals()['invite_status'] = 'Sent successfully.'
                                else:
                                    globals()['invite_status'] = 'Failed to send; opened mail client instead.'
                                    try:
                                        subject, body = build_invite_text()
                                        mailto = f"mailto:{urllib.parse.quote(recip)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
                                        webbrowser.open(mailto)
                                    except Exception:
                                        pass
                                globals()['invite_mode'] = False
                                globals()['invite_input'] = ''
                                globals()['invite_focus'] = False
                                continue
                        else:
                            try:
                                ch = event.unicode
                                if ch:
                                    s += ch
                            except Exception:
                                pass
                        globals()['invite_input'] = s
                        continue
                    # Chat input handling
                    if globals().get('chat_active'):
                        c = globals().get('chat_input', '')
                        if event.key == pygame.K_BACKSPACE:
                            c = c[:-1]
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # send chat to server if in a server game
                            gid = globals().get('current_server_game')
                            server = globals().get('SERVER')
                            if gid and server and getattr(server, 'connected', False):
                                text = c.strip()
                                if text:
                                    server.send({'type': 'chat', 'game_id': gid, 'text': text})
                                    cms = globals().get('chat_messages', [])
                                    cms.append(f"{server.name}: {text}")
                                    globals()['chat_messages'] = cms[-200:]
                            globals()['chat_input'] = ''
                            globals()['chat_active'] = False
                        else:
                            try:
                                ch = event.unicode
                                if ch:
                                    c += ch
                            except Exception:
                                pass
                        globals()['chat_input'] = c
                        continue
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
                draw_menu(win, font, selected_difficulty)
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
        net = globals().get('NETWORK')
        if net:
            try:
                net.close()
            except Exception:
                pass
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
