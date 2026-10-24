import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from nogo_ai.rules import legal_moves, new_board, is_legal_move
from tools.board_platform import Game, make_handler

class BoardTests(unittest.TestCase):
    def test_human_game_and_terminal_result(self):
        game = Game()
        game.start('hh', 2)
        for _ in range(81):
            if game.over:
                break
            game.human(legal_moves(game.board, game.side)[0])
        self.assertTrue(game.over)
        self.assertIn('获胜', game.message)
        self.assertGreater(len(game.history), 10)
        self.assertFalse(legal_moves(game.board, game.side))
        with self.assertRaises(ValueError):
            game.human(0)

    def test_illegal_moves_and_parameters(self):
        game = Game()
        game.start('hh', 2)
        game.human(0)
        for bad in [0, -1, 81, '1', True]:
            with self.assertRaises(ValueError):
                game.human(bad)
        for mode, seconds in [('bad', 2), ('ha', float('nan')), ('ha', 61)]:
            with self.assertRaises(ValueError):
                game.start(mode, seconds)
        # Suicide at corner; and capture of a corner stone are both prohibited.
        board = list(new_board()); board[1] = board[9] = -1
        self.assertFalse(is_legal_move(tuple(board), 1, 0))
        board = list(new_board()); board[0] = -1; board[1] = 1
        self.assertFalse(is_legal_move(tuple(board), 1, 9))

    def test_sau_ai_both_colors_and_cleanup(self):
        game = Game()
        try:
            game.start('aa', 2)
            engines = list(game.engines.values())
            for _ in range(4):
                before = game.board
                side = game.side
                game.step()
                move = game.history[-1]['index']
                self.assertTrue(is_legal_move(before, side, move))
            self.assertEqual(len(game.history), 4)
            game.start('hh', 2)
            self.assertTrue(all(e.process.poll() is not None for e in engines))
        finally:
            game.close()

    def test_complete_sau_ai_game(self):
        game = Game()
        try:
            game.start('aa', .2)
            for _ in range(81):
                if game.over:
                    break
                game.step()
            self.assertTrue(game.over)
            self.assertIn('获胜', game.message)
            self.assertGreater(len(game.history), 10)
            self.assertEqual(game.engines, {})
        finally:
            game.close()

    def test_http_state_token_and_move_validation(self):
        game = Game()
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(game, 'test-token'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        def post(path, data, token='test-token'):
            request = urllib.request.Request(base + path, json.dumps(data).encode(),
                {'Content-Type': 'application/json', 'X-NoGo-Token': token})
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        try:
            with urllib.request.urlopen(base) as response:
                self.assertIn('test-token', response.read().decode())
            with self.assertRaises(urllib.error.HTTPError) as error:
                post('/api/new', {}, 'wrong-token')
            self.assertEqual(error.exception.code, 403)
            post('/api/new', {'mode': 'hh', 'seconds': 2})
            state = post('/api/move', {'move': 0})
            self.assertEqual(state['board'][0], 1)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post('/api/move', {'move': 0})
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            game.close()

if __name__ == '__main__':
    unittest.main()
