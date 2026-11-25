"""Communication with the Pico over USB serial.

- Resends the current command every PICO_PERIOD_S (works as a heartbeat: if the
  Jetson hangs, the Pico turns the motors off after its timeout).
- If the main loop stops updating the command for PICO_STALE_S, sends E (fault: all motors
  1 s, then silence) instead of repeating the old one (a frozen main loop would otherwise keep
  the heartbeat alive and the user could not tell "forward" (silence) from "system down").
- Chest channel (route cues): chest(code) -> 'N,<code>' resent with the same heartbeat.
- Reads the telemetry 'B,v_ina,i_ina,p_ina,v_bat_adc,v_jet_adc,mode[,chest]'.
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
        self._ser = serial.Serial(port, C.PICO_BAUD, timeout=0.005)   # short: a new command must not wait for the read
        self._lock = threading.Lock()
        self._command = "S"
        self._t_command = time.monotonic()    # last time the main loop gave a decision
        self._t_change = None                 # when the command last changed (not yet written)
        self.send_delay_ms = None             # command changed -> written to the serial port
        self.stale = False                    # True while forcing S because decisions stopped
        self._sent = None
        self._chest = 0
        self._chest_sent = None
        self.telemetry = {}           # latest B reading
        self.t_telemetry = 0.0
        self.messages = []            # recent E,... lines
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._write(f"T,{C.PICO_TIMEOUT_MS}")
        self._write(f"K,{getattr(C, 'CHEST_DUTY', 80.0):.1f}")
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

    def chest(self, code):
        """Set the chest (route) pattern code 0..11. Sent immediately if it changed."""
        with self._lock:
            self._chest = int(code)

    def command(self, letter):
        """Set the current command (F, L, R, X, W or S). Sent immediately if it changed."""
        with self._lock:
            if letter != self._command:
                self._t_change = time.monotonic()
            self._command = letter
            self._t_command = time.monotonic()

    def _loop(self):
        t_sent = 0.0
        buffer = b""
        while self._running:
            now = time.monotonic()
            with self._lock:
                letter = self._command
                chest = self._chest
                # Watchdog: if the main loop stops deciding (camera stalled, inference hung),
                # never keep repeating an old direction -> silence the motors.
                self.stale = now - self._t_command > C.PICO_STALE_S
            if self.stale:
                letter, chest = "E", 0
            if letter != self._sent or chest != self._chest_sent or now - t_sent >= C.PICO_PERIOD_S:
                self._write(letter)
                self._write(f"N,{chest}")
                with self._lock:
                    if self._t_change is not None and letter == self._command:
                        self.send_delay_ms = (time.monotonic() - self._t_change) * 1000.0
                        self._t_change = None
                self._sent, self._chest_sent, t_sent = letter, chest, now

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
                    "pico_chest": parts[7] if len(parts) > 7 else None,
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
        self._write("N,0")
        self._write("S")
        self._ser.close()
