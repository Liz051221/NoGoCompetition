import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from nogo_ai.competition_runtime import CompetitionProtocol
from nogo_ai.protocol import EngineConfig
from tools.board_platform import Game, make_handler


class ManualClock:
    def __init__(self):
        self.value = 1000.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ClockTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.clock = ManualClock()
        self.game = Game(record_dir=Path(directory.name), clock=self.clock)
        self.game.start('hh')

    def test_human_clocks_switch_only_on_confirmed_move(self):
        self.clock.advance(7)
        state = self.game.state()
        self.assertEqual(state['clocks'][1]['used'], 7)
        self.assertEqual(state['clocks'][-1]['used'], 0)
        self.assertEqual(state['history'], [])
        self.game.human(0, 0, state['game_id'])
        self.assertEqual(self.game.history[0]['seconds'], 7)
        self.clock.advance(11)
        self.game.human(1, 1, state['game_id'])
        self.assertEqual(self.game.spent, {1: 7, -1: 11})
        self.assertEqual(self.game.history[1]['seconds'], 11)
        with self.assertRaises(ValueError):
            self.game.human(2, 1, state['game_id'])
        self.assertEqual(len(self.game.history), 2)

    def test_reserve_and_timeout_are_enforced_and_recorded(self):
        self.clock.advance(720)
        state = self.game.state()
        self.assertFalse(state['over'])
        self.assertEqual(state['clocks'][1]['thinking_remaining'], 0)
        self.assertEqual(state['clocks'][1]['reserve_remaining'], 180)
        self.clock.advance(180)
        self.assertTrue(self.game.check_clock())
        self.assertEqual(self.game.result, '先手负')
        self.assertIn('超时判负', self.game.message)
        self.assertEqual(self.game.record_path.name, 'A1vsA2，A2胜.txt')
        self.assertEqual(self.game.state()['clocks'][1]['remaining'], 0)
        with self.assertRaises(ValueError):
            self.game.human(0)
        self.assertEqual(self.game.board[0], 0)

    def test_pause_resume_excludes_paused_time(self):
        self.clock.advance(4)
        self.game.set_paused(True)
        self.clock.advance(1000)
        self.assertFalse(self.game.check_clock())
        self.assertEqual(self.game.state()['clocks'][1]['used'], 4)
        with self.assertRaises(ValueError):
            self.game.human(0)
        self.game.set_paused(False)
        self.clock.advance(6)
        self.game.human(0)
        self.assertEqual(self.game.history[0]['seconds'], 10)

    def test_refresh_does_not_reset_time_and_new_game_resets_clocks(self):
        self.clock.advance(13)
        for _ in range(5):
            self.assertEqual(self.game.state()['clocks'][1]['used'], 13)
        previous_id = self.game.game_id
        self.game.start('hh')
        self.assertNotEqual(previous_id, self.game.game_id)
        self.assertEqual(self.game.state()['clocks'][1]['used'], 0)
        with self.assertRaises(ValueError):
            self.game.human(0, 0, previous_id)

    def test_ai_search_does_not_block_clock_requests_and_pause_defers(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingEngine:
            def __init__(self, *args):
                self.commands = []
            def send(self, line):
                self.commands.append(line)
            def ask(self, line, timeout):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('Test did not release engine')
                return 'move AA'
            def close(self):
                pass
        with patch('tools.board_platform.Engine', WaitingEngine):
            self.game.start('ah')
            self.game.request_step()
            self.assertTrue(entered.wait(2))
            self.clock.advance(5)
            with self.game.lock:
                state = self.game.state()
                self.game.set_paused(True)
            self.assertTrue(state['thinking'])
            self.assertEqual(state['clocks'][1]['used'], 5)
            self.assertTrue(self.game.pause_pending)
            self.clock.advance(3)
            release.set()
            self.game.ai_thread.join(3)
            self.assertFalse(self.game.ai_thread.is_alive())
            self.assertEqual(self.game.history[0]['seconds'], 8)
            self.assertTrue(self.game.paused)
            self.assertFalse(self.game.thinking)
            self.clock.advance(2000)
            self.assertEqual(self.game.state()['clocks'][-1]['used'], 0)

    def test_http_async_search_keeps_state_endpoint_responsive(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingEngine:
            def __init__(self, *args): pass
            def send(self, line): pass
            def ask(self, line, timeout):
                entered.set()
                release.wait(3)
                return 'move AA'
            def close(self): pass
        with patch('tools.board_platform.Engine', WaitingEngine):
            self.game.start('ah')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.game, 'clock-test'))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                request = Request(url+'/api/step', b'{}', {'X-NoGo-Token': 'clock-test'})
                with urlopen(request, timeout=1) as response:
                    self.assertEqual(response.status, 200)
                self.assertTrue(entered.wait(1))
                self.clock.advance(3)
                with urlopen(url+'/api/state', timeout=1) as response:
                    state = json.load(response)
                self.assertTrue(state['thinking'])
                self.assertEqual(state['clocks']['1']['used'], 3)
            finally:
                release.set()
                self.game.ai_thread.join(3)
                server.shutdown()
                server.server_close()

    def test_referee_remaining_time_limits_actual_engine_search(self):
        class Searcher:
            time_limit = 60.
            def search(self, board, side):
                self.received_limit = self.time_limit
                from types import SimpleNamespace
                return SimpleNamespace(move=0, elapsed=0., visits=1, value=0., confidence=0., margin=0., reused_tree=False, reason='test', top_candidates=[])
        searcher = Searcher()
        engine = CompetitionProtocol(searcher, EngineConfig(game_time_limit=900, time_reserve=180))
        engine._log = lambda _: None
        engine.handle_line('clock 41')
        self.assertEqual(engine.handle_line('new black'), 'move AA')
        self.assertAlmostEqual(searcher.received_limit, 1., places=3)

if __name__ == '__main__':
    unittest.main()
