#!/usr/bin/env python3
"""Play a fixed walk through the scene, right here on this machine, and measure it.

Nothing goes over the internet: this script talks to the drawing program
(build/viewer) directly, tells it how the camera moves for each picture, and
keeps every picture and every timing it sends back. At the end it writes:

    walk.mp4          the walk as a video (walk.gif if there is no ffmpeg)
    walk_timing.csv   one row per picture: where the camera was and how long it took
    walk_timing.png   a chart of the time per picture

Start it with:
    python3 viewer/replay.py --renderer build/viewer --scene build/scene.bin --out build
"""
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time

# The walk. Each part: (seconds, name, forward m/s, right m/s, up m/s, turn rad/s, look rad/s).
# These are the same five amounts the web page sends, so applyMove sees nothing unusual.
WALK = [
    (2.0, "forward",        2.0, 0.0, 0.0,  0.0,  0.0),
    (2.0, "side-step",      0.0, 2.0, 0.0,  0.0,  0.0),
    (2.0, "turn left",      0.0, 0.0, 0.0, -0.4,  0.0),
    (2.0, "rise, look down", 0.0, 0.0, 1.0,  0.0, -0.2),
    (2.0, "back, turn right", -2.0, 0.0, 0.0,  0.4,  0.0),
]
STEPS_PER_SECOND = 30


def read_exactly(stream, count):
    data = stream.read(count)
    if data is None or len(data) != count:
        raise RuntimeError("the drawing program stopped")
    return data


def play(renderer, scene):
    """Run the walk. Returns (width, height, dots, rows, pictures)."""
    proc = subprocess.Popen([renderer, scene], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        hello = proc.stdout.readline().decode("ascii", "replace").split()
        if len(hello) != 4 or hello[0] != "POINTWALK":
            raise RuntimeError(f"the drawing program did not start properly (it said: {hello!r})")
        width, height, dots = int(hello[1]), int(hello[2]), int(hello[3])

        def one_picture(command):
            started = time.perf_counter()
            proc.stdin.write((command + "\n").encode("ascii"))
            proc.stdin.flush()
            header = proc.stdout.readline().decode("ascii", "replace").split()
            if len(header) != 8 or header[0] != "FRAME":
                raise RuntimeError(f"unexpected reply from the drawing program: {header!r}")
            pixels = read_exactly(proc.stdout, width * height * 3)
            round_trip_ms = (time.perf_counter() - started) * 1000.0
            return [float(v) for v in header[1:]], pixels, round_trip_ms

        # Three throwaway pictures first: the very first ones on a GPU are slower (warm-up),
        # and they would make the walk look worse than it is.
        for _ in range(3):
            one_picture("reset")

        rows, pictures = [], []
        step = 1.0 / STEPS_PER_SECOND
        number = 0
        for seconds, name, forward, right, up, turn, look in WALK:
            for _ in range(round(seconds * STEPS_PER_SECOND)):
                info, pixels, round_trip_ms = one_picture("%f %f %f %f %f" % (
                    forward * step, right * step, up * step, turn * step, look * step))
                x, y, z, yaw, pitch, kernel_ms, frame_ms = info
                rows.append({
                    "picture": number, "seconds": round(number * step, 4), "part": name,
                    "x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch,
                    "kernel_ms": kernel_ms, "gpu_picture_ms": frame_ms,
                    "round_trip_ms": round(round_trip_ms, 4),
                })
                pictures.append(pixels)
                number += 1
        return width, height, dots, rows, pictures
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


def write_video(pictures, width, height, out_dir):
    """Returns the path of the video that was written, or None."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        path = os.path.join(out_dir, "walk.mp4")
        encoder = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(STEPS_PER_SECOND), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)
        for pixels in pictures:
            encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait() == 0:
            return path
        print("ffmpeg could not make the video; falling back to a GIF", file=sys.stderr)
    try:
        from PIL import Image
    except ImportError:
        return None
    # GIF fallback: every second picture, half size, so the file stays small.
    path = os.path.join(out_dir, "walk.gif")
    frames = [Image.frombytes("RGB", (width, height), p).resize((width // 2, height // 2))
              for p in pictures[::2]]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(2000 / STEPS_PER_SECOND), loop=0)
    return path


def write_csv(rows, out_dir):
    path = os.path.join(out_dir, "walk_timing.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_chart(rows, dots, out_dir):
    """A line chart: time per picture across the walk. Returns its path, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    surface, ink, soft_ink, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    series = [("gpu_picture_ms", "Whole picture on GPU", "#2a78d6"),
              ("kernel_ms", "The two kernels", "#eb6834")]
    seconds = [r["seconds"] for r in rows]
    top = max(max(r[key] for r in rows) for key, _, _ in series)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # Where each part of the walk starts, named along the top.
    start = 0.0
    for part_seconds, name, *_ in WALK:
        if start > 0:
            ax.axvline(start, color=grid, linewidth=1, zorder=1)
        ax.text(start + part_seconds / 2, top * 1.22, name, ha="center", va="top",
                fontsize=9, color=soft_ink)
        start += part_seconds

    for key, label, color in series:
        values = [r[key] for r in rows]
        ax.plot(seconds, values, color=color, linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", label=label, zorder=3)
        middle = statistics.median(values)
        ax.annotate(f"{label}\nmedian {middle:.3f} ms", xy=(seconds[-1], values[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=ink)

    ax.set_xlim(0, start)
    ax.set_ylim(0, top * 1.25)
    ax.set_xlabel("Seconds into the walk", color=soft_ink, fontsize=10)
    ax.set_ylabel("Milliseconds per picture", color=soft_ink, fontsize=10)
    ax.set_title(f"Time to draw each picture during the scripted walk ({dots:,} dots)",
                 loc="left", fontsize=12, color=ink, pad=14)
    ax.grid(axis="y", color=grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=soft_ink, labelsize=9, length=0)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(ink)

    fig.subplots_adjust(left=0.08, right=0.78, top=0.86, bottom=0.2)
    path = os.path.join(out_dir, "walk_timing.png")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


def summary(rows, dots, width, height):
    def line(label, key):
        values = [r[key] for r in rows]
        middle = statistics.median(values)
        return (f"  {label:<44} median {middle:8.3f} ms   slowest {max(values):8.3f} ms"
                f"   = {1000.0 / middle:7.0f} pictures per second")

    return "\n".join([
        f"Scripted walk: {len(rows)} pictures, {dots:,} dots, {width}x{height}",
        line("The two kernels", "kernel_ms"),
        line("Whole picture on GPU (wipe, kernels, copy)", "gpu_picture_ms"),
        line("Picture handed to this script (no internet)", "round_trip_ms"),
    ])


def main():
    parser = argparse.ArgumentParser(description="Play and measure a scripted walk")
    parser.add_argument("--renderer", required=True, help="path to the drawing program (build/viewer)")
    parser.add_argument("--scene", required=True, help="path to the scene file (build/scene.bin)")
    parser.add_argument("--out", default=".", help="folder for the video, the table and the chart")
    args = parser.parse_args()

    width, height, dots, rows, pictures = play(args.renderer, args.scene)
    print(summary(rows, dots, width, height), flush=True)
    print("table:", write_csv(rows, args.out))
    print("chart:", write_chart(rows, dots, args.out) or "not made (matplotlib is missing)")
    print("video:", write_video(pictures, width, height, args.out) or "not made (no ffmpeg and no PIL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
