#!/usr/bin/env python3
"""
Simple authoritative chess server for lobby/matchmaking, chat, move relay,
spectator support and PGN saving. Uses a line-based JSON protocol.

Run: python chess_server.py --host 0.0.0.0 --port 5000

Protocol (JSON per line):
Client -> Server messages:
  {"type":"auth","name":"Alice"}
  {"type":"create_game","color":"white","time":600,"inc":0}
  {"type":"list_games"}
  {"type":"join_game","game_id":1}
  {"type":"spectate_game","game_id":1}
  {"type":"move","game_id":1,"move":"r1,c1,r2,c2"}
  {"type":"chat","game_id":1,"text":"hello"}
  {"type":"resign","game_id":1}

Server -> Client messages:
  {"type":"auth_ok","client_id":123}
  {"type":"game_created","game_id":1}
  {"type":"game_list","games":[...]}  # brief game info
  {"type":"game_start","game_id":1,"white":"Alice","black":"Bob"}
  {"type":"move","game_id":1,"move":"r1,c1,r2,c2"}
  {"type":"chat","game_id":1,"from":"Alice","text":"hi"}
  {"type":"game_over","game_id":1,"reason":"resign","winner":"black"}

Note: This is a simple educational server, not hardened for production.
"""

import argparse
import json
import socket
import threading
import time
import os
from collections import defaultdict

CLIENT_TIMEOUT = 300
GAME_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'server_games')
if not os.path.exists(GAME_SAVE_DIR):
    os.makedirs(GAME_SAVE_DIR, exist_ok=True)

lock = threading.Lock()
next_client_id = 1
next_game_id = 1
clients = {}  # client_id -> {'sock':sock,'name':name,'thread':t}
client_sockets = {}  # sock -> client_id

# Games: game_id -> dict with host_id, host_color, players {white:cid, black:cid}, spectators set, moves list, start_time
games = {}


def send_line(sock, obj):
    try:
        data = (json.dumps(obj) + '\n').encode('utf-8')
        sock.sendall(data)
    except Exception as e:
        print('send error', e)


def broadcast_game(game_id, obj, include_spectators=True):
    game = games.get(game_id)
    if not game:
        return
    targets = []
    for role in ('white', 'black'):
        cid = game['players'].get(role)
        if cid and cid in clients:
            targets.append(clients[cid]['sock'])
    if include_spectators:
        for cid in list(game['spectators']):
            if cid in clients:
                targets.append(clients[cid]['sock'])
    for s in targets:
        send_line(s, obj)


def save_game_pgn(game_id):
    game = games.get(game_id)
    if not game:
        return
    filename = os.path.join(GAME_SAVE_DIR, f'game_{game_id}_{int(time.time())}.json')
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(game, f, indent=2, ensure_ascii=False)
        print('Saved game', filename)
    except Exception as e:
        print('Failed save game', e)


def handle_client(sock, addr):
    global next_client_id, next_game_id
    print('Client connected', addr)
    client_id = None
    name = None
    sock_file = sock.makefile('rb')
    try:
        while True:
            line = sock_file.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode('utf-8').strip())
            except Exception:
                continue
            mtype = msg.get('type')
            if mtype == 'auth':
                with lock:
                    client_id = next_client_id
                    next_client_id += 1
                    clients[client_id] = {'sock': sock, 'name': msg.get('name', f'Player{client_id}'), 'addr': addr, 'last_seen': time.time()}
                    client_sockets[sock] = client_id
                name = clients[client_id]['name']
                send_line(sock, {'type': 'auth_ok', 'client_id': client_id})
                # send initial game list
                send_game_list(sock)
                print(f'Auth: {client_id} as {name}')

            elif mtype == 'list_games':
                send_game_list(sock)

            elif mtype == 'create_game':
                color = msg.get('color', 'white')
                game_time = int(msg.get('time', 600))
                inc = int(msg.get('inc', 0))
                with lock:
                    gid = next_game_id
                    next_game_id += 1
                    games[gid] = {'game_id': gid, 'host_id': client_id, 'host_color': color, 'players': {}, 'spectators': set(), 'moves': [], 'created': time.time(), 'time': game_time, 'inc': inc}
                    # host reserved as one player (white or black depending on host_color)
                    if color == 'white':
                        games[gid]['players']['white'] = client_id
                    else:
                        games[gid]['players']['black'] = client_id
                send_line(sock, {'type': 'game_created', 'game_id': gid, 'host_color': color})
                broadcast_game_list_update()
                print(f'Game {gid} created by {name} pref {color}')

            elif mtype == 'join_game':
                gid = int(msg.get('game_id'))
                game = games.get(gid)
                if not game:
                    send_line(sock, {'type': 'error', 'message': 'No such game'})
                    continue
                # assign joining player to opposite color if available
                with lock:
                    assigned = None
                    if 'white' not in game['players']:
                        game['players']['white'] = client_id
                        assigned = 'white'
                    elif 'black' not in game['players']:
                        game['players']['black'] = client_id
                        assigned = 'black'
                    else:
                        send_line(sock, {'type': 'error', 'message': 'Game full'})
                        continue
                    # If both players present, start game
                    if 'white' in game['players'] and 'black' in game['players']:
                        game['started'] = True
                        game['start_time'] = time.time()
                        # notify players and spectators
                        white_name = clients[game['players']['white']]['name'] if game['players'].get('white') in clients else 'Unknown'
                        black_name = clients[game['players']['black']]['name'] if game['players'].get('black') in clients else 'Unknown'
                        broadcast_game(gid, {'type': 'game_start', 'game_id': gid, 'white': white_name, 'black': black_name})
                        print(f'Game {gid} started: {white_name} vs {black_name}')
                broadcast_game_list_update()

            elif mtype == 'spectate_game':
                gid = int(msg.get('game_id'))
                game = games.get(gid)
                if not game:
                    send_line(sock, {'type': 'error', 'message': 'No such game'})
                    continue
                with lock:
                    game['spectators'].add(client_id)
                # send existing moves/history
                send_line(sock, {'type': 'spectate_ok', 'game_id': gid, 'moves': game['moves']})
                broadcast_game_list_update()

            elif mtype == 'move':
                gid = int(msg.get('game_id'))
                mv = msg.get('move')
                game = games.get(gid)
                if not game:
                    send_line(sock, {'type': 'error', 'message': 'No such game'})
                    continue
                with lock:
                    game['moves'].append({'by': client_id, 'move': mv, 'ts': time.time()})
                broadcast_game(gid, {'type': 'move', 'game_id': gid, 'move': mv})

            elif mtype == 'chat':
                gid = int(msg.get('game_id'))
                text = msg.get('text', '')
                game = games.get(gid)
                if not game:
                    send_line(sock, {'type': 'error', 'message': 'No such game'})
                    continue
                from_name = clients[client_id]['name']
                broadcast_game(gid, {'type': 'chat', 'game_id': gid, 'from': from_name, 'text': text})

            elif mtype == 'resign':
                gid = int(msg.get('game_id'))
                game = games.get(gid)
                if not game:
                    send_line(sock, {'type': 'error', 'message': 'No such game'})
                    continue
                # determine winner
                with lock:
                    if game['players'].get('white') == client_id:
                        winner = 'black'
                    else:
                        winner = 'white'
                    game['finished'] = True
                    game['winner'] = winner
                broadcast_game(gid, {'type': 'game_over', 'game_id': gid, 'reason': 'resign', 'winner': winner})
                save_game_pgn(gid)
                broadcast_game_list_update()

            elif mtype == 'list_my_games':
                # send games involving this client
                my = []
                with lock:
                    for gid, g in games.items():
                        if client_id in list(g.get('players', {}).values()) or client_id in g.get('spectators', set()):
                            my.append(g)
                send_line(sock, {'type': 'my_games', 'games': my})

            else:
                send_line(sock, {'type': 'error', 'message': 'Unknown command'})

    except Exception as e:
        print('Client handler exception', e)
    finally:
        print('Client disconnected', addr)
        # cleanup
        try:
            if client_id:
                with lock:
                    # remove from games and spectators
                    for gid, g in list(games.items()):
                        removed = False
                        for role in ('white', 'black'):
                            if g['players'].get(role) == client_id:
                                # if game ongoing, mark finished and other player wins
                                if g.get('started') and not g.get('finished'):
                                    winner = 'black' if role == 'white' else 'white'
                                    g['finished'] = True
                                    g['winner'] = winner
                                    broadcast_game(gid, {'type': 'game_over', 'game_id': gid, 'reason': 'disconnect', 'winner': winner})
                                    save_game_pgn(gid)
                                g['players'].pop(role, None)
                                removed = True
                        if client_id in g.get('spectators', set()):
                            g['spectators'].discard(client_id)
                            removed = True
                        if removed:
                            broadcast_game_list_update()
                    clients.pop(client_id, None)
        except Exception as e:
            print('Cleanup error', e)
        try:
            sock.close()
        except Exception:
            pass


def send_game_list(sock):
    with lock:
        lst = []
        for gid, g in games.items():
            info = {'game_id': gid, 'has_white': 'white' in g['players'], 'has_black': 'black' in g['players'], 'started': g.get('started', False), 'spectators': len(g.get('spectators', []))}
            lst.append(info)
    send_line(sock, {'type': 'game_list', 'games': lst})


def broadcast_game_list_update():
    with lock:
        lst = []
        for gid, g in games.items():
            info = {'game_id': gid, 'has_white': 'white' in g['players'], 'has_black': 'black' in g['players'], 'started': g.get('started', False), 'spectators': len(g.get('spectators', []))}
            lst.append(info)
        # send to all connected clients
        for cid, cinfo in list(clients.items()):
            try:
                send_line(cinfo['sock'], {'type': 'game_list', 'games': lst})
            except Exception:
                pass


def accept_loop(server_sock):
    while True:
        try:
            conn, addr = server_sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            print('Accept error', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(32)
    print(f'Chess server listening on {args.host}:{args.port}')
    try:
        accept_loop(server_sock)
    except KeyboardInterrupt:
        print('Shutting down server')
    finally:
        server_sock.close()

if __name__ == '__main__':
    main()
