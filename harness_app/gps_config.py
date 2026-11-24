"""Configures the u-blox M8N over UBX for walking use, and saves it in the module.

What it sets:
  - Only GGA and RMC NMEA sentences (GLL, GSA, GSV and VTG off): less data on the wire.
  - Dynamic model "pedestrian": smoother position at walking speed and when standing still.
  - 5 Hz navigation rate (one fix every 200 ms) instead of 1 Hz.
  - UART at 115200 baud (5 Hz of NMEA does not fit comfortably in 9600).
  - Save to the module's memory (BBR/flash), so it survives a power cycle if the module keeps it.

Usage:
    python gps_config.py                 # /dev/ttyTHS1, finds the current baud rate
    python gps_config.py /dev/ttyUSB0
Then check with: python gps_probe.py   (it should report 115200 baud)
"""
import struct
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyTHS1"
BAUDS = (115200, 9600, 38400, 57600)
NEW_BAUD = 115200
RATE_MS = 200                  # 5 Hz
DYN_PEDESTRIAN = 3


def ubx(cls, mid, payload=b""):
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for x in body:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def wait_ack(s, cls, mid, timeout=1.5):
    """True on ACK-ACK, False on ACK-NAK, None if nothing arrived."""
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        buf += s.read(s.in_waiting or 1)
        i = buf.find(b"\xb5\x62\x05")
        while i >= 0 and len(buf) >= i + 10:
            if buf[i + 6] == cls and buf[i + 7] == mid:
                return buf[i + 3] == 0x01
            i = buf.find(b"\xb5\x62\x05", i + 1)
        buf = buf[-64:]
    return None


def send(s, name, cls, mid, payload):
    s.reset_input_buffer()
    s.write(ubx(cls, mid, payload))
    r = wait_ack(s, cls, mid)
    print(f"  {name:32s} {'OK' if r else 'NAK (rejected)' if r is False else 'no answer'}")
    return r


def find_baud(port):
    for b in BAUDS:
        with serial.Serial(port, b, timeout=0.5) as s:
            t0 = time.time()
            while time.time() - t0 < 2.5:
                if s.readline(256).decode("ascii", "ignore").startswith("$G"):
                    return b
    return None


def main():
    baud = find_baud(PORT)
    if baud is None:
        sys.exit(f"No NMEA on {PORT}: check the wiring (GPS TX on pin 10).")
    print(f"GPS found on {PORT} at {baud} baud")

    with serial.Serial(PORT, baud, timeout=0.2) as s:
        print("Sending configuration:")
        # NMEA sentences off: GLL (0x01), GSA (0x02), GSV (0x03), VTG (0x05). GGA and RMC stay on.
        for name, msg_id in (("GLL", 0x01), ("GSA", 0x02), ("GSV", 0x03), ("VTG", 0x05)):
            send(s, f"CFG-MSG {name} off", 0x06, 0x01, bytes([0xF0, msg_id, 0]))
        # CFG-NAV5: mask 0x0001 = apply only the dynamic model
        nav5 = struct.pack("<HB", 0x0001, DYN_PEDESTRIAN) + bytes(33)
        ok_nav5 = send(s, "CFG-NAV5 pedestrian", 0x06, 0x24, nav5)
        # CFG-RATE: measurement period, 1 measurement per solution, GPS time reference
        ok_rate = send(s, f"CFG-RATE {1000 // RATE_MS} Hz", 0x06, 0x08, struct.pack("<HHH", RATE_MS, 1, 1))

        # CFG-PRT: UART1, 8N1, new baud, in UBX+NMEA, out UBX+NMEA. The ACK comes at the old
        # baud and is often lost while the port switches, so it is not checked.
        prt = struct.pack("<BBHIIHHHH", 1, 0, 0, 0x000008D0, NEW_BAUD, 0x0003, 0x0003, 0, 0)
        s.write(ubx(0x06, 0x00, prt))
        s.flush()
        time.sleep(0.3)
        print(f"  CFG-PRT {NEW_BAUD} baud               sent")

    with serial.Serial(PORT, NEW_BAUD, timeout=0.2) as s:
        time.sleep(0.5)
        # CFG-CFG: save everything to BBR + flash + EEPROM + SPI flash (whatever the board has)
        save = struct.pack("<IIIB", 0, 0x0000FFFF, 0, 0x17)
        ok_save = send(s, "CFG-CFG save", 0x06, 0x09, save)

        # Check: GGA at the new baud, and the real rate
        t0 = time.time()
        n = 0
        while time.time() - t0 < 3.0:
            if "GGA" in s.readline(256).decode("ascii", "ignore")[:7]:
                n += 1
        print(f"\nAfter the change: {n / 3.0:.1f} GGA per second at {NEW_BAUD} baud")

    if n == 0:
        print("No data at the new baud. Run gps_probe.py: it finds the baud the module is using.")
    elif not (ok_nav5 and ok_rate):
        print("Some settings were not confirmed: run this script again.")
    elif not ok_save:
        print("Settings active, but the save was not confirmed: they may be lost on power-off.\n"
              "The navigation app will apply them at every start anyway.")
    else:
        print("Done. Settings active and saved.")


if __name__ == "__main__":
    main()
