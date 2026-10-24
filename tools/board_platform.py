"""Local browser board; the bundled AI communicates over the SAU protocol."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote
from nogo_ai.rules import new_board, legal_moves, play_move, coord_to_index, index_to_coord

ROOT = Path(__file__).resolve().parents[1]

class Engine:
    def __init__(self, seconds, total_seconds=900, reserve_seconds=180):
        (ROOT / 'logs').mkdir(exist_ok=True)
        self.log = (ROOT / 'logs' / 'platform-engines.log').open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'tools.portable_engine', '--root', str(ROOT), '--move-seconds', str(seconds),
             '--game-seconds', str(total_seconds), '--reserve-seconds', str(reserve_seconds)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', start_new_session=True)
        self.messages = queue.Queue()
        def read():
            for line in self.process.stdout:
                self.messages.put(line.strip())
            self.messages.put(None)
        threading.Thread(target=read, daemon=True).start()
        try:
            if not self.ask('name?', 10).startswith('name '):
                raise RuntimeError('引擎未通过 SAU 握手')
        except Exception:
            self.close()
            raise

    def send(self, line):
        self.process.stdin.write(line + '\n')
        self.process.stdin.flush()

    def ask(self, line, timeout):
        self.send(line)
        try:
            result = self.messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError('引擎响应超时，请检查 logs 后重新开局') from exc
        if result is None:
            raise RuntimeError('引擎已退出，请检查 logs/platform-engines.log')
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.send('quit')
                self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            # Include the search worker if the protocol process failed or hung.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()

class Game:
    def __init__(self, record_dir=None, clock=None, total_seconds=900, reserve_seconds=180):
        self.lock = threading.RLock()
        self.engines = {}
        self.board = new_board()
        self.side = 1
        self.players = {1: 'human', -1: 'human'}
        self.history = []
        self.over = False
        self.message = '选择模式，点击“开始新局”。'
        self.seconds = 60
        self.spent = {1: 0., -1: 0.}
        self.record_dir = Path(record_dir) if record_dir is not None else ROOT / '棋谱'
        self.names = {1: 'A1', -1: 'A2'}
        self.started_at = None
        self.ended_at = None
        self.result = None
        self.record_path = None
        self.record_error = ''
        self.now = clock or time.monotonic
        self.total_seconds = total_seconds
        self.reserve_seconds = reserve_seconds
        self.turn_started = None
        self.turn_spent = 0.
        self.paused = False
        self.pause_pending = False
        self.thinking = False
        self.ai_thread = None
        self.game_id = secrets.token_hex(12)

    def clock_values(self, now=None):
        now = self.now() if now is None else now
        used = dict(self.spent)
        live = max(0., now - self.turn_started) if self.turn_started is not None else 0.
        used[self.side] += live
        return {side: dict(used=used[side], remaining=max(0., self.total_seconds-used[side]),
                          thinking_remaining=max(0., self.total_seconds-self.reserve_seconds-used[side]),
                          reserve_remaining=min(self.reserve_seconds, max(0., self.total_seconds-used[side])),
                          turn_elapsed=self.turn_spent+live if side == self.side else 0.)
                for side in (1, -1)}

    def charge_clock(self, now=None):
        if self.turn_started is not None:
            now = self.now() if now is None else now
            elapsed = max(0., now-self.turn_started)
            self.spent[self.side] += elapsed
            self.turn_spent += elapsed
            self.turn_started = now

    def check_clock(self):
        if self.turn_started is not None and not self.over:
            if self.clock_values()[self.side]['remaining'] <= 0:
                loser = self.side
                self.finish(-loser, reason=('黑方' if loser == 1 else '白方')+'总时间耗尽，超时判负。')
                return True
        return False

    def set_paused(self, paused):
        if self.started_at is None or self.over or self.check_clock():
            return
        if paused and self.thinking:
            self.pause_pending = True
            return
        self.pause_pending = False
        if paused and not self.paused:
            self.charge_clock()
            self.turn_started = None
            self.paused = True
        elif not paused and self.paused:
            self.paused = False
            self.turn_started = self.now()

    def close(self):
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()

    def start(self, mode, seconds=60, first_name='A1', second_name='A2'):
        if mode not in ('hh', 'ha', 'ah', 'aa') or not 0.1 <= seconds <= 60:
            raise ValueError('无效的模式或思考时长')
        if self.thinking:
            raise ValueError('请等待 AI 当前一手结束后再开始新局')
        names = {}
        for side, name in ((1, first_name), (-1, second_name)):
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 24 or any(ord(c) < 32 or ord(c) == 127 for c in name):
                raise ValueError('双方名称需为 1–24 个字符，且不能含换行或控制字符')
            names[side] = name.strip()
        self.close()
        self.board, self.side, self.history, self.over = new_board(), 1, [], False
        self.seconds, self.spent = seconds, {1: 0., -1: 0.}
        self.players = {1: 'human' if mode[0] == 'h' else 'ai', -1: 'human' if mode[1] == 'h' else 'ai'}
        self.message = '新局已开始。'
        self.names = names
        self.started_at = datetime.now().astimezone()
        self.ended_at = self.result = self.record_path = None
        self.record_error = ''
        self.turn_started = None
        self.turn_spent = 0.
        self.paused = self.pause_pending = False
        self.game_id = secrets.token_hex(12)
        try:
            for color in (1, -1):
                if self.players[color] == 'ai':
                    self.engines[color] = Engine(seconds, self.total_seconds, self.reserve_seconds)
                    if color == -1:
                        self.engines[color].send('new white')
        except Exception:
            self.close()
            self.over = True
            self.message = '引擎启动失败，请安装依赖并运行 CheckEngine.command。'
            raise
        self.turn_started = self.now()

    def record_filename(self):
        # Keep names in the record; make separators safe only in the filename.
        def safe(name):
            table = str.maketrans('/\\:*?"<>|', '／＼：＊？＂＜＞｜')
            # The winner's name appears twice; stay below macOS filename limits.
            cleaned = name.translate(table).encode('utf-8')[:76].decode('utf-8', errors='ignore')
            return cleaned.strip('. ') or '未命名'
        first, second = safe(self.names[1]), safe(self.names[-1])
        result = self.result or '进行中'
        if self.result == '先手胜':
            result = first + '胜'
        elif self.result == '先手负':
            result = second + '胜'
        return f'{first}vs{second}，{result}.txt'

    def record_text(self):
        lines = ['NoGo 不围棋棋谱',
                 f'先手（黑方）：{self.names[1]}',
                 f'后手（白方）：{self.names[-1]}',
                 f'黑方类型：{"AI" if self.players[1] == "ai" else "人类"}',
                 f'白方类型：{"AI" if self.players[-1] == "ai" else "人类"}',
                 f'结果：{self.result or "进行中"}',
                 f'对局说明：{self.message}',
                 f'开始时间：{self.started_at.isoformat(timespec="seconds") if self.started_at else "未开始"}',
                 f'结束时间：{self.ended_at.isoformat(timespec="seconds") if self.ended_at else "尚未结束"}',
                 f'总手数：{len(self.history)}',
                 f'每方总时间：{self.total_seconds:g} 秒（思考 {self.total_seconds-self.reserve_seconds:g} 秒＋预留 {self.reserve_seconds:g} 秒）',
                 f'先手累计用时：{self.clock_values()[1]["used"]:.3f} 秒',
                 f'后手累计用时：{self.clock_values()[-1]["used"]:.3f} 秒',
                 '坐标：横纵轴均为 A–I，AA 在左上角；列、行从 1 开始。', '',
                 '手数\t执棋方\t落子坐标\t列\t行\t思考秒数']
        for move in self.history:
            color = '黑（先手）' if move['side'] == 1 else '白（后手）'
            lines.append(f'{move["number"]}\t{color}\t{move["coord"]}\t{move["index"] % 9 + 1}\t{move["index"] // 9 + 1}\t{move["seconds"]:.3f}')
        return '\n'.join(lines) + '\n'

    def save_record(self):
        if self.record_path is not None:
            return
        created = None
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            base = Path(self.record_filename())
            number = 1
            while True:
                filename = base.name if number == 1 else f'{base.stem}（{number}）.txt'
                path = self.record_dir / filename
                try:
                    output = path.open('x', encoding='utf-8', newline='\n')
                except FileExistsError:
                    number += 1
                    continue
                created = path
                with output:
                    output.write(self.record_text())
                    output.flush()
                    os.fsync(output.fileno())
                self.record_path = path
                self.record_error = ''
                break
        except OSError as exc:
            if created is not None:
                try:
                    created.unlink()
                except OSError:
                    pass
            self.record_error = f'自动保存失败：{exc}。请点击“导出棋谱 TXT”保存副本。'

    def finish(self, winner, reason=None):
        self.charge_clock()
        self.turn_started = None
        self.over = True
        self.paused = self.pause_pending = False
        self.ended_at = datetime.now().astimezone()
        self.result = '平局' if winner is None else ('先手胜' if winner == 1 else '先手负')
        self.message = '平局。' if winner is None else ('黑方' if winner == 1 else '白方') + '获胜：对方无合法落点。'
        if reason:
            self.message = ('黑方' if winner == 1 else '白方') + '获胜：' + reason
        self.save_record()
        self.close()

    def place(self, move):
        if self.check_clock():
            return
        next_board = play_move(self.board, self.side, move)
        self.charge_clock()
        self.history.append({'number': len(self.history)+1, 'side': self.side,
                             'coord': index_to_coord(move), 'index': move, 'seconds': round(self.turn_spent, 3)})
        self.board = next_board
        self.side = -self.side
        self.turn_spent = 0.
        self.turn_started = None
        if not legal_moves(self.board, self.side):
            self.finish(-self.side)
        else:
            self.turn_started = self.now()
            self.message = '轮到' + ('黑方' if self.side == 1 else '白方') + '。'
            if self.pause_pending:
                self.set_paused(True)

    def human(self, move, expected_move_no=None, game_id=None):
        if self.check_clock():
            return
        if self.started_at is None or self.over or self.paused or self.thinking or self.players[self.side] != 'human':
            raise ValueError('当前不能由人类落子')
        if (expected_move_no is not None and expected_move_no != len(self.history)) or (game_id is not None and game_id != self.game_id):
            raise ValueError('局面已更新，请重新选择落点并确认')
        if type(move) is not int or move not in legal_moves(self.board, self.side):
            raise ValueError('此处不能落子：不许提子、不许自杀，也不能占用已有棋子的位置。')
        self.place(move)

    def request_step(self):
        if self.ai_thread is not None and self.ai_thread.is_alive():
            return
        self.ai_thread = threading.Thread(target=self.step, daemon=True)
        self.ai_thread.start()

    def step(self):
        with self.lock:
            if self.check_clock() or self.started_at is None or self.over or self.paused or self.thinking or self.players[self.side] != 'ai':
                return
            self.thinking = True
            side, game_id = self.side, self.game_id
            engine = self.engines[side]
            command = 'new black' if not self.history else 'move ' + self.history[-1]['coord']
            clock = self.clock_values()[side]
        try:
            engine.send(f'clock {clock["thinking_remaining"]:.6f}')
            reply = engine.ask(command, min(self.seconds + 10, clock['remaining'] + .05))
            with self.lock:
                if self.over or self.game_id != game_id or self.check_clock():
                    return
                parts = reply.split()
                if len(parts) != 2 or parts[0] != 'move':
                    raise ValueError('引擎回复不符合 SAU move 协议')
                move = coord_to_index(parts[1])
                if move not in legal_moves(self.board, side):
                    raise ValueError('引擎返回非法落子')
                self.thinking = False
                self.place(move)
        except Exception as exc:
            with self.lock:
                if self.over or self.game_id != game_id or self.check_clock():
                    return
                self.charge_clock()
                self.turn_started = None
                self.over = True
                self.ended_at = datetime.now().astimezone()
                self.result = '中止（未判胜负）'
                self.message = '对局中止（未判胜负）：' + str(exc)
                self.close()
        finally:
            with self.lock:
                self.thinking = False

    def state(self):
        self.check_clock()
        return dict(board=self.board, side=self.side, history=list(self.history), over=self.over,
                    message=self.message, players=self.players, spent=dict(self.spent),
                    names=self.names, result=self.result, game_id=self.game_id, server_time=self.now(),
                    clocks=self.clock_values(), total_seconds=self.total_seconds, reserve_seconds=self.reserve_seconds,
                    active_side=self.side if self.turn_started is not None else None,
                    paused=self.paused, pause_pending=self.pause_pending, thinking=self.thinking,
                    move_cap_seconds=self.seconds,
                    record_filename=self.record_path.name if self.record_path else None,
                    record_error=self.record_error,
                    legal=[] if self.over else legal_moves(self.board, self.side))

def make_handler(game, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, content_type='application/json; charset=utf-8', filename=None):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if filename:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(filename, safe=''))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                page = (ROOT / 'tools' / 'board.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
                return self.reply(200, page.encode(), 'text/html; charset=utf-8')
            if self.path == '/api/state':
                with game.lock:
                    return self.reply(200, game.state())
            if self.path == '/api/record':
                with game.lock:
                    return self.reply(200, game.record_text().encode('utf-8'),
                                      'text/plain; charset=utf-8',
                                      game.record_path.name if game.record_path else game.record_filename())
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.headers.get('X-NoGo-Token') != token:
                return self.reply(403, {'error': '请刷新本地棋盘页面'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 4096:
                    raise ValueError('请求过大')
                data = json.loads(self.rfile.read(size) or b'{}')
                with game.lock:
                    if self.path == '/api/new':
                        game.start(data.get('mode', 'ha'), float(data.get('seconds', 60)),
                                   data.get('first_name', 'A1'), data.get('second_name', 'A2'))
                    elif self.path == '/api/move':
                        game.human(data.get('move'), data.get('expected_move_no'), data.get('game_id'))
                    elif self.path == '/api/step':
                        game.request_step()
                    elif self.path == '/api/pause':
                        if type(data.get('paused')) is not bool:
                            raise ValueError('无效的暂停请求')
                        game.set_paused(data['paused'])
                    else:
                        return self.reply(404, {'error': 'Not found'})
                    self.reply(200, game.state())
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                self.reply(400, {'error': str(exc)})
    return Handler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    game = Game()
    stop_clock = threading.Event()
    def watch_clock():
        while not stop_clock.wait(.1):
            with game.lock:
                game.check_clock()
    threading.Thread(target=watch_clock, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(game, secrets.token_hex(24)))
    address = f'http://127.0.0.1:{server.server_port}'
    print('NoGo 棋盘已启动。请在本机浏览器打开：' + address, flush=True)
    print('退出：在此终端按 Control+C。关闭浏览器标签页不会停止服务。', flush=True)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_clock.set()
        server.server_close()
        with game.lock:
            game.close()

if __name__ == '__main__':
    main()
