from ursina import *

# Simple 3D Chess demo using primitive shapes (Ursina)
# - Click a piece to select, then click a tile to move.
# - Basic legal moves implemented (no check/checkmate enforcement).

BOARD_OFFSET = Vec3(-3.5, 0, -3.5)
SCALE = 1

class Piece(Entity):
    def __init__(self, color, kind, pos, **kwargs):
        super()._yes_init__(**kwargs)
        self.color_name = color
        self.kind = kind
        self.board_pos = pos
        self.set_visual()
        self.selected = False

    def set_visual(self):
        # Build a composed Staunton-like piece from primitives for nicer visuals
        # Clear any built-in model and create child parts.
        self.model = None
        base_color = color.white if self.color_name == 'white' else color.black
        # Common base
        Entity(parent=self, model='cylinder', scale=Vec3(0.9, 0.12, 0.9), position=Vec3(0, -0.45, 0), color=base_color)
        Entity(parent=self, model='cylinder', scale=Vec3(0.7, 0.08, 0.7), position=Vec3(0, -0.33, 0), color=base_color)

        if self.kind == 'Pawn':
            Entity(parent=self, model='cylinder', scale=Vec3(0.5, 0.4, 0.5), position=Vec3(0, -0.05, 0), color=base_color)
            Entity(parent=self, model='sphere', scale=0.28, position=Vec3(0, 0.35, 0), color=base_color)

        elif self.kind == 'Rook':
            Entity(parent=self, model='cylinder', scale=Vec3(0.6, 0.6, 0.6), position=Vec3(0, 0.05, 0), color=base_color)
            # battlements
            for i, x in enumerate([-0.25, 0, 0.25]):
                Entity(parent=self, model='cube', scale=Vec3(0.18, 0.18, 0.6), position=Vec3(x, 0.45, 0), color=base_color)

        elif self.kind == 'Knight':
            # approximate knight with a sloped head: sphere + cone
            Entity(parent=self, model='cylinder', scale=Vec3(0.55, 0.5, 0.55), position=Vec3(0, -0.05, 0), color=base_color)
            Entity(parent=self, model='sphere', scale=Vec3(0.45, 0.45, 0.35), position=Vec3(0.12, 0.35, 0.05), color=base_color, rotation=Vec3(0,20,0))
            Entity(parent=self, model='cone', scale=Vec3(0.25, 0.35, 0.25), position=Vec3(-0.12, 0.25, -0.05), color=base_color, rotation=Vec3(0,-30,0))

        elif self.kind == 'Bishop':
            Entity(parent=self, model='cylinder', scale=Vec3(0.5, 0.6, 0.5), position=Vec3(0, 0.0, 0), color=base_color)
            Entity(parent=self, model='cone', scale=Vec3(0.35, 0.6, 0.35), position=Vec3(0, 0.55, 0), color=base_color)
            # notch for bishop
            Entity(parent=self, model='cube', scale=Vec3(0.12, 0.18, 0.02), position=Vec3(0, 0.7, 0.14), color=base_color)

        elif self.kind == 'Queen':
            Entity(parent=self, model='cylinder', scale=Vec3(0.6, 0.7, 0.6), position=Vec3(0, 0.0, 0), color=base_color)
            Entity(parent=self, model='sphere', scale=Vec3(0.35, 0.35, 0.35), position=Vec3(0, 0.65, 0), color=base_color)
            # small crowns
            for a in range(6):
                ang = a * (360/6)
                x = 0.35 * math.cos(math.radians(ang))
                z = 0.35 * math.sin(math.radians(ang))
                Entity(parent=self, model='cone', scale=Vec3(0.08, 0.18, 0.08), position=Vec3(x, 0.9, z), color=base_color, rotation=Vec3(0, ang, 0))

        elif self.kind == 'King':
            Entity(parent=self, model='cylinder', scale=Vec3(0.65, 0.75, 0.65), position=Vec3(0, 0.0, 0), color=base_color)
            Entity(parent=self, model='sphere', scale=Vec3(0.28, 0.28, 0.28), position=Vec3(0, 0.7, 0), color=base_color)
            # simple cross
            Entity(parent=self, model='cube', scale=Vec3(0.02, 0.36, 0.02), position=Vec3(0, 1.02, 0), color=base_color)
            Entity(parent=self, model='cube', scale=Vec3(0.18, 0.02, 0.02), position=Vec3(0, 1.08, 0), color=base_color)

        # collider on parent so clicks hit the whole piece
        self.collider = 'box'

    def world_pos_from_board(self, r, c):
        x = c + BOARD_OFFSET.x
        z = r + BOARD_OFFSET.z
        return Vec3(x * SCALE, 0.5, z * SCALE)

    def update_position(self):
        r, c = self.board_pos
        self.position = self.world_pos_from_board(r, c)

    def get_valid_moves(self, board):
        # Minimal legal-move generation (ignores check)
        r, c = self.board_pos
        moves = []
        if self.kind == 'Pawn':
            dir = -1 if self.color_name == 'white' else 1
            nr = r + dir
            if 0 <= nr < 8:
                if board[nr][c] is None:
                    moves.append((nr, c))
                for dc in (-1,1):
                    nc = c + dc
                    if 0 <= nc < 8 and board[nr][nc] and board[nr][nc].color_name != self.color_name:
                        moves.append((nr, nc))
        else:
            # reuse simple linear moves for Rook/Bishop/Queen and knight offsets
            if self.kind == 'Knight':
                offs = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]
                for dr,dc in offs:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<8 and 0<=nc<8 and (board[nr][nc] is None or board[nr][nc].color_name!=self.color_name):
                        moves.append((nr,nc))
            else:
                directions = []
                if self.kind in ('Rook','Queen'):
                    directions += [(1,0),(-1,0),(0,1),(0,-1)]
                if self.kind in ('Bishop','Queen'):
                    directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
                for dr,dc in directions:
                    nr, nc = r+dr, c+dc
                    while 0<=nr<8 and 0<=nc<8:
                        if board[nr][nc] is None:
                            moves.append((nr,nc))
                        else:
                            if board[nr][nc].color_name != self.color_name:
                                moves.append((nr,nc))
                            break
                        nr += dr
                        nc += dc
        return moves


class Chess3DGame:
    def __init__(self):
        self.selected = None
        self.turn = 'white'
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.tiles = {}
        self.create_board()
        self.setup_pieces()

    def create_board(self):
        for r in range(8):
            for c in range(8):
                pos = Vec3(c + BOARD_OFFSET.x, 0, r + BOARD_OFFSET.z)
                t = Entity(model='quad', scale=1, rotation_x=90, position=pos, collider='box')
                if (r+c)%2==0:
                    t.color = color.light_gray
                else:
                    t.color = color.dark_gray
                self.tiles[(r,c)] = t

    def setup_pieces(self):
        # Pawns
        for c in range(8):
            self.spawn_piece('black','Pawn',(1,c))
            self.spawn_piece('white','Pawn',(6,c))
        order = ['Rook','Knight','Bishop','Queen','King','Bishop','Knight','Rook']
        for c, kind in enumerate(order):
            self.spawn_piece('black', kind, (0,c))
            self.spawn_piece('white', kind, (7,c))

    def spawn_piece(self, color_name, kind, pos):
        r,c = pos
        p = Piece(color_name, kind, pos)
        p.update_position()
        p.parent = scene
        p.world_parent = scene
        self.board[r][c] = p

    def input(self, key):
        if key == 'left mouse down':
            hit = mouse.hovered_entity
            if isinstance(hit, Entity):
                # clicked a piece?
                for r in range(8):
                    for c in range(8):
                        p = self.board[r][c]
                        if p and p == hit:
                            if p.color_name == self.turn:
                                self.selected = p
                                print(f"Selected {p.kind} at {p.board_pos}")
                                return
                # clicked a tile
                for (r,c), tile in self.tiles.items():
                    if tile == hit and self.selected:
                        moves = self.selected.get_valid_moves(self.board)
                        if (r,c) in moves:
                            # move
                            sr, sc = self.selected.board_pos
                            self.board[sr][sc] = None
                            # capture if present
                            if self.board[r][c]: destroy(self.board[r][c])
                            self.board[r][c] = self.selected
                            self.selected.board_pos = (r,c)
                            self.selected.update_position()
                            self.selected = None
                            self.turn = 'black' if self.turn == 'white' else 'white'
                            print(f"Moved. Turn: {self.turn}")
                        else:
                            print('Invalid move')


if __name__ == '__main__':
    app = Ursina()
    window.title = '3D Chess (Ursina demo)'
    game = Chess3DGame()

    def input(key):
        game.input(key)

    app.run()
