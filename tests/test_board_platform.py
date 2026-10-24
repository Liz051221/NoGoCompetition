import json
from datetime import datetime
import re
from pathlib import Path
import random
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from nogo_ai.rules import legal_moves, new_board, is_legal_move
from tools.board_platform import Game, make_handler

class BoardTests(unittest.TestCase):
    def setUp(self):
        records = tempfile.TemporaryDirectory()
        self.addCleanup(records.cleanup)
        self.record_dir = Path(records.name) / '棋谱'

    def game(self):
        return Game(record_dir=self.record_dir)

    def test_human_game_and_terminal_result(self):
        game = self.game()
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
        game = self.game()
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
        game = self.game()
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
        game = self.game()
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
            self.assertTrue(game.record_path.is_file())
            self.assertIn(game.history[-1]['coord'], game.record_path.read_text())
        finally:
            game.close()

    def test_http_state_token_and_move_validation(self):
        game = self.game()
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
            while not game.over:
                state = post('/api/move', {'move': legal_moves(game.board, game.side)[0]})
            self.assertEqual(state['record_filename'], 'A1vsA2，A1胜.txt')
            with urllib.request.urlopen(base + '/api/record') as response:
                self.assertEqual(response.read().decode('utf-8'), game.record_path.read_text())
                self.assertTrue(response.headers['Content-Type'].startswith('text/plain'))
                self.assertIn("filename*=UTF-8''", response.headers['Content-Disposition'])
        finally:
            server.shutdown()
            server.server_close()
            game.close()

    def test_auto_record_complete_positions_and_first_player_result(self):
        for seed, result in ((None, '先手胜'), (1, '先手负')):
            with self.subTest(result=result):
                game = self.game()
                game.start('hh', 2, '我的模型', '我')
                rng = random.Random(seed)
                while not game.over:
                    moves = legal_moves(game.board, game.side)
                    game.human(moves[0] if seed is None else rng.choice(moves))
                winner = '我的模型' if result == '先手胜' else '我'
                self.assertEqual(game.record_path.name, f'我的模型vs我，{winner}胜.txt')
                text = game.record_path.read_text(encoding='utf-8')
                outcome = '先手胜' if result == '先手胜' else '后手胜'
                self.assertTrue(text.startswith(f'([NG][我的模型][我][{outcome}]'))
                rows = re.findall(r';([BW])\[([A-I][1-9])\]', text)
                self.assertEqual(len(rows), len(game.history))
                for row, move in zip(rows, game.history):
                    self.assertEqual(row[0], 'B' if move['side'] == 1 else 'W')
                    self.assertEqual(row[1], move['coord'])
                    self.assertEqual(row[1], f'{chr(65 + move["index"] % 9)}{move["index"] // 9 + 1}')

    def test_auto_record_does_not_overwrite_previous_game(self):
        game = self.game()
        paths = []
        for _ in range(2):
            game.start('hh', 2)
            while not game.over:
                game.human(legal_moves(game.board, game.side)[0])
            paths.append(game.record_path)
            game.state()
            game.save_record()
        self.assertEqual([p.name for p in paths], ['A1vsA2，A1胜.txt', 'A1vsA2，A1胜（2）.txt'])
        self.assertEqual(len(list(self.record_dir.glob('*.txt'))), 2)
        self.assertTrue(all(p.read_text().count(';B[') == 39 for p in paths))

    def test_failed_auto_save_preserves_result_and_text(self):
        self.record_dir.parent.mkdir(parents=True, exist_ok=True)
        self.record_dir.write_text('This file blocks directory creation.')
        game = self.game()
        game.start('hh', 2)
        while not game.over:
            game.human(legal_moves(game.board, game.side)[0])
        self.assertEqual(game.result, '先手胜')
        self.assertIn('获胜', game.message)
        self.assertIsNone(game.record_path)
        self.assertIn('自动保存失败', game.state()['record_error'])
        self.assertIn('[先手胜]', game.record_text())
        self.assertIn(game.history[-1]['coord'], game.record_text())

    def test_player_names_cannot_escape_record_directory(self):
        game = self.game()
        game.start('hh', 2, '../模型/1', '玩家:2')
        while not game.over:
            game.human(legal_moves(game.board, game.side)[0])
        self.assertEqual(game.record_path.parent.resolve(), self.record_dir.resolve())
        self.assertIn('[../模型/1]', game.record_path.read_text())
        with self.assertRaises(ValueError):
            game.start('hh', 2, '名称\n含换行', 'A2')
        self.assertTrue(game.over)

    def test_record_matches_supplied_ng_example(self):
        game = self.game()
        game.start('hh', 2, '参赛队B', '参赛队W', '重庆', '2017 CCGC')
        game.started_at = datetime.fromisoformat('2017-07-29T14:00:00+08:00')
        for move in [39, 40, 31, 30, 49, 41, 50, 32, 42]:
            game.human(move)
        game.history[0]['mark'] = 1
        game.finish(1)
        expected = '([NG][参赛队B][参赛队W][先手胜][2017.07.29 14:00 重庆][2017 CCGC];B[D5]MARK[1];W[E5];B[E4];W[D4];B[E6];W[F5];B[F6];W[F4];B[G5])\n'
        self.assertEqual(game.record_path.read_text(), expected)
        self.assertEqual(game.record_text(), expected)

    def test_ng_metadata_delimiters_cannot_break_header(self):
        game = self.game()
        game.start('hh', 2, '队[1]', '队(2)', '场地;一', '赛]事[\\')
        game.human(0)
        text = game.record_text()
        header, moves = text.strip()[1:-1].split(';', 1)
        fields = re.findall(r'\[([^\[\]]*)\]', header)
        self.assertEqual(len(fields), 6)
        self.assertEqual(fields[:4], ['NG', '队［1］', '队（2）', '进行中'])
        self.assertTrue(fields[4].endswith('场地；一'))
        self.assertEqual(fields[5], '赛］事［＼')
        self.assertEqual(moves, 'B[A1]')

if __name__ == '__main__':
    unittest.main()
