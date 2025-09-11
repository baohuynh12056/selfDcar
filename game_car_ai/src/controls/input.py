# src/controls/input.py
import subprocess
import threading
import queue
import time

class AndroidController:
    def __init__(self, device_id=None, send_interval=0.15):
        self.device_id = device_id
        self.proc = None
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.running = True
        self.send_interval = send_interval  # giãn lệnh 150ms
        self.holding = None
        self._open_shell()
        self._start_worker()

    def _open_shell(self):
        cmd = ["adb"]
        if self.device_id:
            cmd += ["-s", self.device_id]
        cmd.append("shell")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

    def _send_cmd(self, command: str):
        with self.lock:
            try:
                self.proc.stdin.write(command + "\n")
                self.proc.stdin.flush()
            except:
                pass

    def _worker(self):
        while self.running:
            try:
                cmd_tuple = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if cmd_tuple is None:
                continue
            cmd, *args = cmd_tuple
            if cmd == "tap":
                x, y = args
                self._send_cmd(f"input tap {x} {y}")
            elif cmd == "tap_hold":
                x, y, duration = args
                self._send_cmd(f"input swipe {x} {y} {x} {y} {duration}")
            elif cmd == "release":
                if self.holding:
                    x, y = self.holding
                    self._send_cmd(f"input tap {x} {y}")
                    self.holding = None
            time.sleep(self.send_interval)
            self.queue.task_done()

    def _start_worker(self):
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    # API
    def tap(self, x, y):
        self.queue.put(("tap", x, y))

    def tap_hold(self, x, y, duration=100):
        self.queue.put(("tap_hold", x, y, duration))
        self.holding = (x, y)

    def release(self):
        self.queue.put(("release",))

    def clear_queue(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

    def close(self):
        self.running = False
        self.clear_queue()
        try:
            self.proc.terminate()
        except:
            pass
