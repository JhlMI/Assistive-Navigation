import static_ffmpeg
static_ffmpeg.add_paths()

import yt_dlp
from pathlib import Path

videos = [
    {
        "url":    "https://www.youtube.com/watch?v=UzgbIAxZgXg",
        "inicio": "29:00",
        "fin":    "34:00",
        "salida": "data/videos/acera_test.mp4"
    },
]

Path("data/videos").mkdir(parents=True, exist_ok=True)

def tiempo_a_segundos(t: str) -> float:
    partes = t.split(":")
    return sum(int(x) * 60**i for i, x in enumerate(reversed(partes)))

def descargar(v: dict):
    opts = {
        "format": "bestvideo[height<=720][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
        "outtmpl": v["salida"],
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(tiempo_a_segundos(v["inicio"]), tiempo_a_segundos(v["fin"]))]
        ),
        "force_keyframes_at_cuts": True,
        "progress_hooks": [
            lambda d: print(f"  {d.get('_percent_str','').strip()}  {d.get('_speed_str','').strip()}", end="\r")
            if d["status"] == "downloading"
            else print(f"\n  ✅ Guardado: {v['salida']}")
            if d["status"] == "finished"
            else None
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([v["url"]])

for v in videos:
    print(f"\nDescargando: {v['salida']}  [{v['inicio']} → {v['fin']}]")
    descargar(v)