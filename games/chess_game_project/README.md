# 3D Chess (Ursina demo)

This is a lightweight 3D chess demo using the Ursina engine. Pieces are primitive shapes (no external 3D asset downloads required).

How to run

1. Install dependencies (best inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Run the demo:

```bash
python chess_3d.py
```

Controls
- Left-click a piece to select (only pieces of the current turn).
- Left-click a tile to move if the move is legal (basic move rules implemented; no check/checkmate enforcement).

Notes
- This is a simple demo to get a 3D board and pieces working. If you want full rules (check/checkmate, castling, en-passant, promotions), I can extend it.
- If you prefer higher-quality 3D models, I can add instructions and utilities to download or import .obj/.gltf models and textures.
