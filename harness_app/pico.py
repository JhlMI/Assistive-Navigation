"""Communication with the Pico over USB serial.

- Resends the current command every PICO_PERIOD_S (works as a heartbeat: if the
  Jetson hangs, the Pico turns the motors off after its timeout).
- If the main loop stops updating the command for PICO_STALE_S, sends S instead of
  repeating the old one (otherwise a frozen main loop would keep the heartbeat alive).
- Reads the telemetry 'B,v_ina,i_ina,p_ina,v_bat_adc,v_jet_adc,mode'.
"""
import glob
import threading
import time

import config as C


class Pico:
    def __init__(self, port=None):
        import serial   # pyserial

        port = port or C.PICO_PORT or self._detect()
        if port is None:
            raise RuntimeError("Pico not found (/dev/ttyACM*)")
        self.port = port
        self._ser = serial.Serial(port, C.PICO_BAUD, timeout=0.05)
        self._lock = threading.Lock()
        self._command = "S"
        self._t_command = time.monotonic()    # last time the main loop gave a decision
        self.stale = False                    # True while forcing S because decisions stopped
        self._sent = None
        self.telemetry = {}           # latest B reading
        self.t_telemetry = 0.0
        self.messages = []            # recent E,... lines
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._write(f"T,{C.PICO_TIMEOUT_MS}")
        self._thread.start()

    @staticmethod
    def _detect():
        ports = sorted(glob.glob("/dev/ttyACM*"))
        return ports[0] if ports else None

    def _write(self, text):
        try:
            self._ser.write((text + "\n").encode())
        except Exception as e:
            self.messages.append(f"write error: {e}")

    def command(self, letter):
        """Set the current command (F, L, R, X or S). Sent immediately if it changed."""
        with self._lock:
            self._command = letter
            self._t_command = time.monotonic()

    def _loop(self):
        t_sent = 0.0
        buffer = b""
        while self._running:
            now = time.monotonic()
            with self._lock:
                letter = self._command
                # Watchdog: if the main loop stops deciding (camera stalled, inference hung),
                # never keep repeating an old direction -> silence the motors.
                self.stale = now - self._t_command > C.PICO_STALE_S
            if self.stale:
                letter = "S"
            if letter != self._sent or now - t_sent >= C.PICO_PERIOD_S:
                self._write(letter)
                self._sent, t_sent = letter, now

            try:
                buffer += self._ser.read(256)
            except Exception as e:
                self.messages.append(f"read error: {e}")
                time.sleep(0.5)
                continue
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._parse(line.decode(errors="ignore").strip())

    def _parse(self, line):
        if line.startswith("B,"):
            parts = line.split(",")
            if len(parts) >= 7:
                try:
                    v, i, p, vbat, vjet = (float(x) for x in parts[1:6])
                except ValueError:
                    return
                self.telemetry = {
                    "v_ina": v if v >= 0 else None,
                    "i_ina": i if v >= 0 else None,
                    "p_ina": p if v >= 0 else None,
                    "v_bat_adc": vbat,
                    "v_jet": vjet,
                    "pico_mode": parts[6],
                }
                self.t_telemetry = time.monotonic()
        elif line.startswith("E,"):
            self.messages = (self.messages + [line[2:]])[-20:]

    def battery_voltage(self):
        """Battery voltage: INA219 if it responds, otherwise the ADC divider."""
        t = self.telemetry
        return t.get("v_ina") or t.get("v_bat_adc")

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self._write("S")
        self._ser.close()
