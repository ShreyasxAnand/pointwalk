#!/usr/bin/env python3
"""Stage 4: paint the walls, one camera stop at a time.

For every camera on the route:
  1. draw the scene from there (same camera math and same "glue distance and dot
     number, keep the smallest" trick as the GPU kernels, written here in PyTorch)
  2. find the wall dots this view sees better than any view before it
  3. ask an image model to paint only those boxes, guided by the distance picture
  4. copy the painted colours onto those dots and raise their confidence

Nothing is added or moved. The walls already exist as dots (make_walls.py); this
stage only decides what colour they are.

    python3 paint_route.py --scene quad_walls.bin --route route.txt --out quad_painted.bin

Needs a GPU:  pip install torch diffusers transformers accelerate pillow numpy
"""
import argparse
import math
import os
import shutil

import numpy as np
import torch
from PIL import Image

SIZE = 512              # the picture is SIZE x SIZE boxes, the size the image model was trained on
FOCAL = 400.0           # focal length in boxes. With SIZE 512 the camera sees about 65 degrees across.
DOT_SIZE = 0.3          # metres. Same meaning as in splatKernel.
NEAR_ENOUGH = 15.0      # metres. A wall this close (or closer) is seen as well as distance allows.
HIDDEN_SLACK = 0.25     # metres. A dot this far behind the front surface still counts as visible,
                        # plus two boxes' worth of width, because a slanted wall spans a range of
                        # distances inside one box.
MIN_SHARE = 0.01        # skip a view if less than this share of its boxes would be painted
STEPS = 20              # denoising rounds per picture
PROMPT = ("photograph of a sandstone university building, romanesque arches, "
          "arcade, carved stone, windows, sunny day")
NEGATIVE = "blurry, distorted, text, watermark, people, cars"
EMPTY = torch.iinfo(torch.int64).max       # an empty note: larger than any real glued number

device = "cuda" if torch.cuda.is_available() else "cpu"


def render(points, colours, pose):
    """Draw the scene from one camera.

    Returns, for the SIZE x SIZE boxes:
      picture   colour of each box (sky where no dot landed)
      distance  how far away the winning dot is (0 where empty)
      winner    which dot won each box (-1 where empty)
    and, for every dot:
      box       which box its centre lands in (-1 if it is not in the picture)
      depth     its distance from the camera
    """
    x, y, z, yaw, pitch = pose
    sin_yaw, cos_yaw, sin_pitch, cos_pitch = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    # R and t, exactly as in setPose (src/camera.h).
    R = torch.tensor([[cos_yaw, 0.0, -sin_yaw],
                      [sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch],
                      [sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch]], device=device)
    t = -R @ torch.tensor([x, y, z], device=device)

    in_camera = points @ R.T + t                       # camera point = R * world point + t
    depth = in_camera[:, 2].contiguous()
    in_front = depth > 0.1
    safe_depth = torch.where(in_front, depth, torch.ones_like(depth))
    col = torch.floor(FOCAL * in_camera[:, 0] / safe_depth + SIZE / 2).long()
    row = torch.floor(FOCAL * in_camera[:, 1] / safe_depth + SIZE / 2).long()
    on_picture = in_front & (col >= 0) & (col < SIZE) & (row >= 0) & (row < SIZE)
    box = torch.where(on_picture, row * SIZE + col, torch.full_like(col, -1))

    # Glue: the distance's raw bits on the left, the dot number on the right.
    glued = (depth.view(torch.int32).to(torch.int64) << 32) | torch.arange(len(points), device=device)

    # Each dot is a square, FOCAL * DOT_SIZE / distance boxes wide, between 1 and 8.
    width = torch.clamp(torch.ceil(FOCAL * DOT_SIZE / safe_depth), 1, 8).long()
    close = in_front & (col > -8) & (col < SIZE + 8) & (row > -8) & (row < SIZE + 8)
    notes = torch.full((SIZE * SIZE,), EMPTY, dtype=torch.int64, device=device)
    for down in range(8):
        for across in range(8):
            chosen = torch.nonzero(close & (width > max(down, across))).squeeze(1)
            c = col[chosen] + across - width[chosen] // 2     # the square is centred on the dot
            r = row[chosen] + down - width[chosen] // 2
            inside = (c >= 0) & (c < SIZE) & (r >= 0) & (r < SIZE)
            # "closer? overwrite the note" for all of these dots at once
            notes.scatter_reduce_(0, (r * SIZE + c)[inside], glued[chosen][inside], reduce="amin")

    filled = notes != EMPTY
    winner = torch.where(filled, notes & 0xFFFFFFFF, torch.full_like(notes, -1))
    distance = torch.where(filled, (notes >> 32).to(torch.int32).view(torch.float32),
                           torch.zeros(SIZE * SIZE, device=device))

    # Sky for the empty boxes, the same formula as resolveSkyKernel.
    boxes = torch.arange(SIZE * SIZE, device=device)
    a = ((boxes % SIZE) + 0.5 - SIZE / 2) / FOCAL
    b = ((boxes // SIZE) + 0.5 - SIZE / 2) / FOCAL
    up = -(R[0, 1] * a + R[1, 1] * b + R[2, 1]) / torch.sqrt(a * a + b * b + 1.0)
    blend = torch.where(up > 0, torch.sqrt(up.clamp(min=0)), (-up * 4.0).clamp(max=1.0))[:, None]
    horizon = torch.tensor([200.0, 220.0, 240.0], device=device)
    far = torch.where((up > 0)[:, None], torch.tensor([70.0, 130.0, 210.0], device=device),
                      torch.tensor([120.0, 115.0, 105.0], device=device))
    sky = (horizon + (far - horizon) * blend).to(torch.uint8)

    picture = torch.where(filled[:, None], colours[winner.clamp(min=0)], sky)
    return picture, distance, winner, box, depth


def load_painter():
    """Load the image model and return a function: (picture, mask, distance picture) -> painted picture."""
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
    guide = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", controlnet=guide,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    seed = torch.Generator(device="cpu").manual_seed(0)

    def paint(picture, mask, guide_picture):
        # mask: white boxes are repainted, black boxes are kept.
        return pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=picture, mask_image=mask,
                    control_image=guide_picture, num_inference_steps=STEPS, guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0, strength=1.0, generator=seed).images[0]
    return paint


def main(paint=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", default="paint_log", help="folder for before / mask / after pictures")
    parser.add_argument("--limit", type=int, default=0,
                        help="for a quick try: do only about this many views, spread evenly along the route (0 = all)")
    args = parser.parse_args()
    os.makedirs(args.log, exist_ok=True)

    # ---- Read the scene and its side files ----
    with open(args.scene, "rb") as f:
        assert f.read(8) == b"PWSCENE1", "not a scene file"
        count = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        points = torch.from_numpy(np.frombuffer(f.read(count * 12), dtype=np.float32).reshape(count, 3).copy()).to(device)
        colours = torch.from_numpy(np.frombuffer(f.read(count * 3), dtype=np.uint8).reshape(count, 3).copy()).to(device)
    confidence = torch.from_numpy(np.fromfile(args.scene + ".conf", dtype=np.uint8)).to(device).float()
    normals = torch.from_numpy(np.fromfile(args.scene + ".normals", dtype=np.float32).reshape(count, 2)).to(device)

    # ---- Read the route. Flyover views go first: they see every wall, but badly. The walk
    # comes after and repaints what it sees better, because a better view always wins. ----
    cameras = []
    for line in open(args.route):
        x, y, z, yaw, pitch, kind = line.split()
        cameras.append(((float(x), float(y), float(z), float(yaw), float(pitch)), kind))
    cameras.sort(key=lambda camera: camera[1] != "flyover")
    if args.limit:
        cameras = cameras[::max(1, len(cameras) // args.limit)]

    if paint is None:
        paint = load_painter()

    painted_views = 0
    for number, (pose, kind) in enumerate(cameras):
        picture, distance, winner, box, depth = render(points, colours, pose)

        # ---- How well does this view see each wall dot? ----
        # visible:  the dot is in the picture and not hidden behind something nearer
        # facing:   1 when the camera looks at the wall straight on, 0 when it looks along it
        # near:     1 within NEAR_ENOUGH metres, fading with distance beyond that
        # quality:  20 for the worst useful view, 70 for the best
        on_picture = box >= 0
        visible = on_picture & (depth <= distance[box.clamp(min=0)] * (1 + 2 / FOCAL) + HIDDEN_SLACK)
        to_dot = points[:, [0, 2]] - torch.tensor([pose[0], pose[2]], device=device)    # (east, north)
        flat_distance = torch.linalg.norm(to_dot, dim=1).clamp(min=0.01)
        facing = (-(to_dot * normals).sum(dim=1) / flat_distance).clamp(0, 1)
        near = (NEAR_ENOUGH / flat_distance).clamp(max=1.0)
        quality = 20.0 + 50.0 * facing * near
        better = visible & (confidence < 100) & (quality > confidence) & (facing > 0)

        # The boxes to paint: those whose winning dot is one this view improves.
        mask = better[winner.clamp(min=0)] & (winner >= 0)
        share = mask.float().mean().item()
        if share < MIN_SHARE:
            print(f"view {number + 1}/{len(cameras)} ({kind}): nothing new to paint", flush=True)
            continue

        # ---- The three pictures the model gets ----
        before = Image.fromarray(picture.reshape(SIZE, SIZE, 3).cpu().numpy())
        mask_picture = Image.fromarray((mask.reshape(SIZE, SIZE).cpu().numpy() * 255).astype(np.uint8))
        # Distance picture: near is bright, far is dark, sky is black. That is what the guide expects.
        nearness = torch.where(distance > 0, 1.0 / distance.clamp(min=0.1), torch.zeros_like(distance))
        nearness = (nearness / nearness.max().clamp(min=1e-6) * 255).to(torch.uint8).reshape(SIZE, SIZE)
        guide_picture = Image.fromarray(nearness.cpu().numpy()).convert("RGB")

        after = paint(before, mask_picture, guide_picture).resize((SIZE, SIZE))

        # ---- Copy the painted colours onto the dots ----
        # Every improved dot takes the colour of the box its centre is in, if that box was painted.
        painted = torch.from_numpy(np.asarray(after.convert("RGB")).copy()).to(device).reshape(SIZE * SIZE, 3)
        chosen = torch.nonzero(better & mask[box.clamp(min=0)]).squeeze(1)
        colours[chosen] = painted[box[chosen]]
        confidence[chosen] = quality[chosen]
        painted_views += 1

        strip = Image.new("RGB", (SIZE * 3, SIZE))
        strip.paste(before, (0, 0))
        strip.paste(mask_picture.convert("RGB"), (SIZE, 0))
        strip.paste(after, (SIZE * 2, 0))
        strip.save(os.path.join(args.log, f"{number:03d}_{kind}.jpg"), quality=85)
        print(f"view {number + 1}/{len(cameras)} ({kind}): painted {share:.0%} of the picture, "
              f"{len(chosen):,} dots recoloured", flush=True)

    # ---- Write the painted scene, with the same side files ----
    with open(args.out, "wb") as f:
        f.write(b"PWSCENE1")
        f.write(np.int32(count).tobytes())
        f.write(points.cpu().numpy().tobytes())
        f.write(colours.cpu().numpy().tobytes())
    confidence.to(torch.uint8).cpu().numpy().tofile(args.out + ".conf")
    for ending in (".labels", ".normals", ".pose", ".ground.npz"):
        shutil.copy(args.scene + ending, args.out + ending)

    # ---- A second scene that shows confidence as colour, to fly through and inspect ----
    # Measured dots keep their colour, dimmed. Wall dots go from red (never painted)
    # through yellow to green (painted from a close, straight-on view).
    level = ((confidence - 10.0) / 60.0).clamp(0, 1)[:, None]
    red, yellow, green = (torch.tensor(c, device=device) for c in ([220.0, 50.0, 40.0], [235.0, 200.0, 50.0], [60.0, 190.0, 80.0]))
    ramp = torch.where(level < 0.5, red + (yellow - red) * level * 2, yellow + (green - yellow) * (level - 0.5) * 2)
    shown = torch.where((confidence >= 100)[:, None], colours.float() * 0.45, ramp).to(torch.uint8)
    confidence_scene = args.out.rsplit(".", 1)[0] + "_confidence.bin"
    with open(confidence_scene, "wb") as f:
        f.write(b"PWSCENE1")
        f.write(np.int32(count).tobytes())
        f.write(points.cpu().numpy().tobytes())
        f.write(shown.cpu().numpy().tobytes())
    shutil.copy(args.scene + ".pose", confidence_scene + ".pose")

    walls = confidence < 100
    print(f"painted {painted_views} of {len(cameras)} views")
    print(f"wall dots: {int(walls.sum()):,}; painted at least once: {int((walls & (confidence > 10)).sum()):,}; "
          f"painted well (confidence 50 or more): {int((walls & (confidence >= 50)).sum()):,}")
    print("saved", args.out, "and", confidence_scene)


if __name__ == "__main__":
    main()
