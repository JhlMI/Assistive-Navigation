"""GPS probe without the navigation app: one line per second with fix and signal quality.

Uses the same reader as the app (gps.py). Stop the app first (~/stop_test.sh): two programs
cannot read the same serial port.

Usage:
    python gps_probe.py              # /dev/ttyTHS1
    python gps_probe.py --sky        # also lists every satellite and its signal
    python gps_probe.py --log        # also writes logs/gps/gpstest_<time>.csv

cn0_top4 = mean signal of the 4 strongest satellites (dB-Hz): >= 40 clear sky, < 35 blocked sky
or noise near the antenna (USB 3 cable, Jetson).
"""
import argparse
import time

from gps import Gps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None)
    ap.add_argument("--sky", action="store_true", help="list every satellite each second")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    print("Connecting to the GPS (and applying the walking setup)...")
    gps = Gps(args.port)
    print(f"GPS on {gps.port} at {gps.baud} baud. Ctrl+C to stop.\n")
    if args.log:
        gps.start_test()
    try:
        while True:
            time.sleep(1.0)
            s = gps.status()
            pos = (f"lat {s['gps_lat']:.6f} lon {s['gps_lon']:.6f}" if s["gps_lat"] is not None
                   else "no fix            ")
            print(f"{time.strftime('%H:%M:%S')} | {pos} | used {s['gps_sats']} / view {s['gps_sats_view']}"
                  f" | cn0_top4 {s['gps_cn0_top4']} dB-Hz | HDOP {s['gps_hdop']}"
                  f" | {s['gps_speed_kmh']} km/h", flush=True)
            if args.sky:
                sats = sorted(gps.sky_list(), key=lambda x: -(x["snr"] or 0))
                print("   " + "  ".join(f"{x['sys'][:3]}{x['prn']}:{x['snr'] or '-'}" for x in sats))
    except KeyboardInterrupt:
        pass
    finally:
        gps.close()
        print("\nStopped")


if __name__ == "__main__":
    main()
