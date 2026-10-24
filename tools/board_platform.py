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
    def __init__(self, seconds):
        (ROOT / 'logs').mkdir(exist_ok=True)
        self.log = (ROOT / 'logs' / 'platform-engines.log').open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'tools.portable_engine', '--root', str(ROOT), '--move-seconds', str(seconds)],
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
    def __init__(self, record_dir=None):
        self.lock = threading.RLock()
        self.engines = {}
        self.board = new_board()
        self.side = 1
        self.players = {1: 'human', -1: 'human'}
        self.history = []
        self.over = False
        self.message = '选择模式，点击“开始新局”。'
        self.seconds = 2
        self.spent = {1: 0., -1: 0.}
        self.record_dir = Path(record_dir) if record_dir is not None else ROOT / '棋谱'
        self.names = {1: 'A1', -1: 'A2'}
        self.started_at = None
        self.ended_at = None
        self.result = None
        self.record_path = None
        self.record_error = ''

    def close(self):
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()

    def start(self, mode, seconds, first_name='A1', second_name='A2'):
        if mode not in ('hh', 'ha', 'ah', 'aa') or not 0.1 <= seconds <= 60:
            raise ValueError('无效的模式或思考时长')
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
        try:
            for color in (1, -1):
                if self.players[color] == 'ai':
                    self.engines[color] = Engine(seconds)
                    if color == -1:
                        self.engines[color].send('new white')
        except Exception:
            self.close()
            self.over = True
            self.message = '引擎启动失败，请安装依赖并运行 CheckEngine.command。'
            raise

    def record_filename(self):
        # Keep names in the record; make separators safe only in the filename.
        def safe(name):
            table = str.maketrans('/\\:*?"<>|', '／＼：＊？＂＜＞｜')
            return name.translate(table).strip('. ') or '未命名'
        return f'{safe(self.names[1])}：{safe(self.names[-1])}，{self.result or "进行中"}.txt'

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
                 '坐标：横纵轴均为 A–I，AA 在左上角；列、行从 1 开始。', '',
                 '手数\t执棋方\t落子坐标\t列\t行']
        for move in self.history:
            color = '黑（先手）' if move['side'] == 1 else '白（后手）'
            lines.append(f'{move["number"]}\t{color}\t{move["coord"]}\t{move["index"] % 9 + 1}\t{move["index"] // 9 + 1}')
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

    def finish(self, winner):
        self.over = True
        self.ended_at = datetime.now().astimezone()
        self.result = '平局' if winner is None else ('先手胜' if winner == 1 else '先手负')
        self.message = '平局。' if winner is None else ('黑方' if winner == 1 else '白方') + '获胜：对方无合法落点。'
        self.save_record()
        self.close()

    def place(self, move, elapsed=0.):
        self.board = play_move(self.board, self.side, move)
        self.spent[self.side] += elapsed
        self.history.append({'number': len(self.history)+1, 'side': self.side,
                             'coord': index_to_coord(move), 'index': move, 'seconds': round(elapsed, 3)})
        self.side = -self.side
        if not legal_moves(self.board, self.side):
            self.finish(-self.side)
        else:
            self.message = '轮到' + ('黑方' if self.side == 1 else '白方') + '。'

    def human(self, move):
        if self.over or self.players[self.side] != 'human':
            raise ValueError('当前不能由人类落子')
        if type(move) is not int or move not in legal_moves(self.board, self.side):
            raise ValueError('此处不能落子：不许提子、不许自杀，也不能占用已有棋子的位置。')
        self.place(move)

    def step(self):
        if self.over or self.players[self.side] != 'ai':
            return
        try:
            command = 'new black' if not self.history else 'move ' + self.history[-1]['coord']
            started = time.perf_counter()
            reply = self.engines[self.side].ask(command, self.seconds + 10)
            elapsed = time.perf_counter() - started
            parts = reply.split()
            if len(parts) != 2 or parts[0] != 'move':
                raise ValueError('引擎回复不符合 SAU move 协议')
            move = coord_to_index(parts[1])
            if move not in legal_moves(self.board, self.side):
                raise ValueError('引擎返回非法落子')
            self.place(move, elapsed)
        except Exception as exc:
            self.over = True
            self.ended_at = datetime.now().astimezone()
            self.result = '中止（未判胜负）'
            self.message = '对局中止（未判胜负）：' + str(exc)
            self.close()
            raise

    def state(self):
        return dict(board=self.board, side=self.side, history=self.history, over=self.over,
                    message=self.message, players=self.players, spent=self.spent,
                    names=self.names, result=self.result,
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
                        game.start(data.get('mode', 'ha'), float(data.get('seconds', 2)),
                                   data.get('first_name', 'A1'), data.get('second_name', 'A2'))
                    elif self.path == '/api/move':
                        game.human(data.get('move'))
                    elif self.path == '/api/step':
                        game.step()
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
        server.server_close()
        with game.lock:
            game.close()

if __name__ == '__main__':
    main()
