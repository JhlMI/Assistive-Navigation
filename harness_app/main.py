"""Assisted navigation system - main program (Jetson Orin Nano).

camera -> segmentation + detection (TensorRT) -> decision -> Pico (motors)
                                                    |-> CSV log + web view

Usage:
    python main.py                       # OAK-D Lite camera + Pico + web
    python main.py --video path.avi      # test with a recorded video
    python main.py --video path.avi --video-fps 15   # if the video plays too fast
    python main.py --no-pico             # without the Pico (vision only)
    python main.py --no-web              # without the web view (saves CPU)
    python main.py --record              # start recording right away (also from the web button)
    python main.py --video rec_X.avi --loop   # demo: replay a street video forever, motors respond live
"""
import argparse
import os
import signal
import sys
import time

if "ultralytics" in sys.modules:
    sys.exit("ultralytics is imported: it drops OpenCV to 1 thread. Aborting.")

import config as C
from camera import OakCamera, VideoFile
from decision import Decider, Result
from logger import CsvLogger
from perception import Perception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="use a video file instead of the camera")
    ap.add_argument("--video-fps", type=float, help="real FPS of the video (if the file header is wrong)")
    ap.add_argument("--no-pico", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--port", help="Pico serial port (default /dev/ttyACM*)")
    ap.add_argument("--record", action="store_true", help="save the camera video for later replay")
    ap.add_argument("--loop", action="store_true", help="with --video: restart the video when it ends (demos)")
    args = ap.parse_args()

    running = [True]
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__(0, False))
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))

    # ---- Pico ---------------------------------------------------------------
    pico = None
    if not args.no_pico:
        try:
            from pico import Pico
            pico = Pico(args.port)
            print(f"Pico: {pico.port}")
        except Exception as e:
            print(f"Pico not available ({e}); continuing without motors")

    # ---- Models and camera ----------------------------------------------------
    print("Loading TensorRT models...")
    perception = Perception()
    # Decision: BEV planner (grid + path) if calibrated, otherwise the sidewalk-centroid rule
    import bev
    use_bev = C.DECISION_MODE == "bev" and bev.CALIB_FILE.exists()
    if use_bev:
        from planner import BevPlanner
        make_decider = BevPlanner
    else:
        make_decider = Decider
        if C.DECISION_MODE == "bev":
            print("No bev_calibration.json: using the centroid decision instead")
    decider = make_decider()
    print(f"Decision mode: {'BEV planner' if use_bev else 'centroid'}"
          + (f" (blind zone {decider.y_start:.2f} m)" if use_bev else ""))
    source = (VideoFile(args.video, args.video_fps, loop=args.loop) if args.video
              else OakCamera(C.CAMERA_FPS, C.CAMERA_FOCUS, C.FOCUS_RANGE))
    print(f"Source: {args.video or 'OAK-D Lite'}")
    if hasattr(source, "usb_speed"):
        print(f"Camera USB: {source.usb_speed} | focus: {C.CAMERA_FOCUS}")

    # ---- Logging and web ------------------------------------------------------
    log = CsvLogger()
    print(f"Log: {log.path}")
    from recorder import AnnotatedRecorder, Recorder
    recorder = None                  # raw camera video (for replaying with new algorithms)
    recorder_ann = None              # annotated video: what the system saw and decided (evidence)
    state = None
    server = None
    if not args.no_web:
        from web import SharedState, start_server
        state = SharedState()
        state.recording = args.record
        server = start_server(state)
        print(f"Web view: http://<jetson-ip>:{C.WEB_PORT}")

    def close_recorder(rec):
        rec.close()
        print(f"Video saved to {rec.path} ({rec.written / rec.fps:.0f} s at {rec.fps} FPS, "
              f"{rec.dropped} frames dropped)")
        print(f"Replay it with: python main.py --video {rec.path}")

    def close_annotated(rec):
        rec.close()
        print(f"Annotated video saved to {rec.path} ({rec.frames} frames with guidance ON, "
              f"{rec.dropped} dropped, {rec.written / rec.fps:.0f} s at {rec.fps} FPS)")

    # ---- Main loop ------------------------------------------------------------
    last_id = 0
    fps = 0.0
    t_prev = time.perf_counter()
    t_print = t_prev
    t_low_batt = None
    guidance_was_on = True
    last_loops = 0
    print("Running. Ctrl+C to quit.\n")

    while running[0] and source.active:
        frame, last_id = source.next(last_id, timeout=1.0)
        if frame is None:
            continue

        # Recording: started/stopped from the web (or --record at launch)
        want_rec = state.recording if state is not None else args.record
        if want_rec and recorder is None:
            recorder = Recorder(fps=15)
            if C.REC_ANNOTATED:
                recorder_ann = AnnotatedRecorder(fps=C.REC_ANNOTATED_FPS, size=C.REC_ANNOTATED_SIZE,
                                                 stamp=recorder.path.stem[len("rec_"):])
                print(f"Recording: {recorder.path} + {recorder_ann.path.name} (annotated, guidance ON only)")
            else:
                print(f"Recording: {recorder.path} (raw only)")
        elif not want_rec and recorder is not None:
            close_recorder(recorder)
            if recorder_ann:
                close_annotated(recorder_ann)
            recorder = recorder_ann = None
        if state is not None:
            state.rec_info = recorder.path.name if recorder else None
        if recorder:
            recorder.add(frame)
        # Looping demo video: restart the filters so the end of the clip does not leak into its start
        loops = getattr(source, "loops", 0)
        if loops != last_loops:
            last_loops = loops
            decider = make_decider()
            print(f"Video restarted (loop {loops})")
        guidance = state.guidance if state is not None else True
        t0 = time.perf_counter()
        if guidance:
            if not guidance_was_on:
                decider = make_decider()     # start fresh: no stale smoothing from before
            walkable, boxes, times = perception.process(frame)
            if use_bev:
                res = decider.update(walkable, boxes, frame.shape, perception.last_mask)
            else:
                res = decider.update(walkable, boxes, frame.shape)
        else:                                # camera only: no inference, motors silent
            walkable, boxes, times = None, [], {"seg_ms": 0.0, "det_ms": 0.0}
            res = Result(command="S", reason="guidance off")
        guidance_was_on = guidance
        if pico:
            pico.command(res.command)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000
        dt = t1 - t_prev
        t_prev = t1
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        tel = pico.telemetry if pico else {}
        vbat = pico.battery_voltage() if pico else None

        # Low battery: warning in the console and in the web view
        warning = None
        if vbat is not None and vbat < C.VBAT_CRITICAL:
            t_low_batt = t_low_batt or time.monotonic()
        else:
            t_low_batt = None
        if vbat is not None and vbat < C.VBAT_WARNING:
            warning = f"low battery ({vbat:.2f} V)"
            if t_low_batt and time.monotonic() - t_low_batt > 10:
                warning = f"CRITICAL BATTERY ({vbat:.2f} V): shut the system down"

        row = {"fps": round(fps, 1), "lat_ms": round(lat_ms, 1),
               "seg_ms": round(times["seg_ms"], 1), "det_ms": round(times["det_ms"], 1),
               "command": res.command, "reason": res.reason, "error": round(res.error, 3),
               "sidewalk_frac": round(res.sidewalk_frac, 3), "n_boxes": len(boxes),
               "n_obst": len(res.obstacles), "angle_deg": round(getattr(res, "angle_deg", 0.0), 1),
               "path_len": round(getattr(res, "path_len", 0.0), 2),
               "obst_dist": None if getattr(res, "obst_dist", None) is None else round(res.obst_dist, 2),
               **tel}
        log.write(row)

        # Annotated recording: only frames processed with guidance ON
        if recorder_ann is not None and guidance:
            bat_txt = f"{vbat:.2f} V" if vbat else "-"
            info = (f"{time.strftime('%H:%M:%S')} | FPS {fps:4.1f} | lat {lat_ms:5.1f} ms | "
                    f"seg {times['seg_ms']:.0f} ms det {times['det_ms']:.0f} ms | "
                    f"{res.command} ({res.reason}) | bat {bat_txt}")
            recorder_ann.add(time.monotonic(), frame, walkable, boxes, res, info)

        if state is not None:
            data = {"command": res.command, "reason": res.reason, "fps": round(fps, 1),
                    "latency_ms": round(lat_ms, 1), "error": round(res.error, 2),
                    "sidewalk_%": round(100 * res.sidewalk_frac, 1),
                    "angle_deg": round(getattr(res, "angle_deg", 0.0), 1) if use_bev else None,
                    "path_m": round(getattr(res, "path_len", 0.0), 1) if use_bev else None,
                    "obstacle_m": (None if getattr(res, "obst_dist", None) is None else round(res.obst_dist, 1)),
                    "battery_V": round(vbat, 2) if vbat else None,
                    "current_A": tel.get("i_ina"), "power_W": tel.get("p_ina"),
                    "drok_V": tel.get("v_jet"), "tj_C": log.temperature(),
                    "focus": getattr(source, "lens_position", None),
                    "usb": getattr(source, "usb_speed", None),
                    "pico": "connected" if pico else "no",
                    "guidance": "ON" if guidance else "OFF",
                    "recording": state.rec_info, "photos": state.snapshots, "warning": warning}
            state.update(frame, walkable, boxes, res, data)

        if t1 - t_print >= 2.0:
            t_print = t1
            bat = f"{vbat:5.2f} V" if vbat else "  -  "
            cur = f"{tel['i_ina']:5.3f} A" if tel.get("i_ina") is not None else "  -  "
            print(f"FPS {fps:5.1f} | lat {lat_ms:5.1f} ms | {res.command} ({res.reason:<21s}) | "
                  f"err {res.error:+.2f} | sidewalk {100 * res.sidewalk_frac:4.1f} % | "
                  f"bat {bat} | {cur}" + (f" | {warning}" if warning else ""))

    # ---- Clean shutdown -------------------------------------------------------
    print("\nShutting down...")
    if pico:
        pico.close()                # leaves the motors silent
    source.stop()
    if recorder:
        close_recorder(recorder)
    if recorder_ann:
        close_annotated(recorder_ann)
    if server:
        server.shutdown()
    log.close()
    print(f"Log saved to {log.path}", flush=True)
    # Skip interpreter teardown: TensorRT/CUDA/depthai destructors can abort at exit
    # (everything is already saved and closed at this point).
    os._exit(0)


if __name__ == "__main__":
    main()
