"""Best-effort live display, independent of training logs and thermal decisions."""
from __future__ import annotations

import codecs
from datetime import datetime
import os
from pathlib import Path
import queue
import re
import sys
import threading

# Keep colors, but remove cursor movement that could overwrite an earlier alert.
CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TrainingConsole:
  def __init__(self, quiet=False, stream=None, capacity=128):
    self.quiet = quiet
    self.stream = stream if stream is not None else sys.stdout
    self.progress = queue.Queue(maxsize=capacity)
    self.notices = queue.Queue(maxsize=32)
    self.closing = threading.Event()
    self.unavailable = threading.Event()
    self.follow_stop = threading.Event()
    self.followers = []
    self.writer = None
    self.dropped = 0
    self.line_open = False
    self.tty = self.stream.isatty()
    self.fd = None
    self.escape = ""
    self.final_message = None

  def _put(self, target, value):
    if self.unavailable.is_set() or self.closing.is_set():
      return
    # No terminal I/O, no waiting for queue space, no lock held by the writer.
    try:
      target.put_nowait(value)
    except queue.Full:
      self.dropped += 1
      try:
        target.get_nowait()
      except queue.Empty:
        pass
      try:
        target.put_nowait(value)
      except queue.Full:
        pass

  def notice(self, message):
    self._put(self.notices, f"[{datetime.now():%H:%M:%S}] {str(message)[:8192]}\n")

  def output(self, text):
    if not self.quiet:
      # Bound item size as well as queue length.
      for offset in range(0, len(text), 4096):
        self._put(self.progress, text[offset:offset + 4096])

  def start(self):
    if self.writer is not None:
      return
    try:
      self.fd = os.dup(self.stream.fileno())
    except (AttributeError, OSError, ValueError):
      self.fd = None
    self.writer = threading.Thread(target=self._write, daemon=True, name="training-console")
    self.writer.start()

  def _emit(self, text):
    if self.fd is None:  # StringIO/test streams; production uses raw fd writes.
      self.stream.write(text)
      self.stream.flush()
    else:
      # Never hold sys.stdout's buffered lock: a stuck daemon must not block
      # Python's interpreter-shutdown flush, either.
      data = text.encode(getattr(self.stream, "encoding", None) or "utf-8", errors="replace")
      while data:
        count = os.write(self.fd, data)
        data = data[count:]

  def _write(self):
    try:
      while not self.closing.is_set() or not self.notices.empty() or not self.progress.empty():
        try:
          text = self.notices.get_nowait()
          notice = True
        except queue.Empty:
          try:
            text = self.progress.get(timeout=.05)
          except queue.Empty:
            continue
          notice = False
        if notice:
          text = ("\n" if self.line_open else "") + text
        else:
          text = self.escape + text
          self.escape = ""
          last_escape = text.rfind("\x1b")
          if last_escape >= 0:
            suffix = text[last_escape:]
            if suffix == "\x1b" or re.fullmatch(r"\x1b\[[0-?]*[ -/]*", suffix):
              self.escape, text = suffix[:64], text[:last_escape]
          text = CSI.sub(lambda match: match[0] if self.tty and match[0].endswith("m") else "", text)
          text = text.replace("\r", "\r\x1b[2K" if self.tty else "\n")
        if text:
          self._emit(text)
          self.line_open = not text.endswith("\n")
      if self.line_open:
        self._emit("\n")
      if self.dropped:
        self._emit("[display] Some live updates were omitted; complete logs remain on disk.\n")
      if self.final_message:
        self._emit(f"[{datetime.now():%H:%M:%S}] {self.final_message}\n")
    except (OSError, ValueError):
      self.unavailable.set()
    finally:
      if self.fd is not None:
        os.close(self.fd)

  def follow(self, path: Path):
    if self.quiet:
      return
    follower = threading.Thread(target=self._follow, args=(path,), daemon=True, name="training-log-view")
    self.followers.append(follower)
    follower.start()

  def _follow(self, path):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
      with path.open("rb") as stream:
        final_budget = 65536
        while True:
          data = stream.read(4096)
          if data:
            self.output(decoder.decode(data))
            if self.follow_stop.is_set():
              final_budget -= len(data)
              if final_budget <= 0:
                break
          elif self.follow_stop.is_set():
            break
          else:
            self.follow_stop.wait(.05)
        self.output(decoder.decode(b"", final=True))
    except OSError as error:
      self.notice(f"Live training display unavailable: {error}; inspect training.log.")

  def close(self, final_message=None):
    self.final_message = final_message
    self.follow_stop.set()
    for follower in self.followers:
      follower.join(timeout=.3)
    self.closing.set()
    self.start()
    self.writer.join(timeout=.3)
