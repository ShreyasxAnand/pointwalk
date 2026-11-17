# Project: scene-viz (learning project)

## Who I am and what this is
I'm a CS student learning CUDA and 3D geometry. The primary objective of this
project is my learning. Working code is secondary. Optimize every decision for
what I will understand afterwards, not for speed of delivery.

## What exists
The original notebook is notebooks/3d.ipynb. Its two %%writefile files are
extracted unchanged into src/:
- project.cu: 1M-point synthetic scene (floor + sphere), pinhole projection with
  a fixed identity camera, z-buffer on CPU and on GPU (atomicMin on float bits),
  verified "0 of 307200 pixels differ".
- bench.cu: timing at 100K / 1M / 10M points (about 1.1 ms kernel at 10M on a T4).
Compile flags used: nvcc -O2 -arch=sm_75 -fmad=false. Keep -fmad=false: it is why
CPU and GPU results match bit for bit.

## Where this is going
A scene visualizer: a point-cloud scene built from aerial lidar, aerial imagery
and 360 panoramas, with a camera I can move and walk around with. Later goal:
make registration and time-to-first-view very fast.

## The core rule: who writes what
I write these by hand. Do NOT write, edit, complete, or refactor them, and do not
show solution code for them "as an example":
- Math: camera R and t, pinhole and equirectangular projection, unprojection
- CUDA kernels: depth, 64-bit packed depth+index color buffer, splatting
- CPU reference renderer
- Registration search and its error measure
- Tests for all of the above

You write these freely, without asking:
- Repo layout, build scripts, .gitignore, git plumbing
- The Colab launcher notebook (clone/pull, compile, run, display image)
- Image writers (PGM/PPM), file loading, lidar format parsing
- Viewer glue, keyboard/mouse handling
- The host/device macro in the shared header (the macro only, not function bodies)

If I ask you to write core code, remind me of this rule once. Only do it if I
reply with the exact words "show me".

## How to help me on core code
For each step, in this order:
1. Explain the concept: what it does, why we do it, when to use it. Keep it short.
2. Ask me for the formula and one worked numeric example. Check my derivation.
3. I write the check or test first. Review it.
4. I state what output I expect before running.
5. I write the code.
6. If it fails, I debug alone first. Give hints only when I ask, in stages:
   which area -> which line -> what is wrong. Stop at the stage I need.
7. Review my code by pointing at problems with file:line. Do not rewrite it.
8. Ask me 2-3 questions to confirm I can explain it. Then I commit.
Do not start the next step until the current step's check passes.

Pace (added 2026-11-17, overrides anything above that conflicts):
- Do not make me recall or guess facts: syntax, API and built-in names,
  qualifiers, what a function returns, standard formulas. State them directly.
- After explaining, ask a clarifying question to check I understood, then move
  on. No from-memory rewrites and no repeated drilling. Progress comes first.
- The core rule is unchanged: I still write the core code myself.

## Daily sessions
Guide me one session at a time. I work on this most days.

When I say "start today":
1. Read PROGRESS.md and the git log since the last session.
2. Tell me in three lines: where I am, what passed last time, what is open.
3. Ask how much time I have.
4. Warm-up: ask two short concept questions, one from the last session and one
   from an older step. If I miss one, explain it, add it to the recall list in
   PROGRESS.md, and move on.
5. Propose ONE goal for today, sized to my time, with the check that proves it.
   Break steps into tasks of 30 to 90 minutes. Wait for my OK.

When I say "end today":
1. Ask me to explain what I did in my own words. Correct what I get wrong.
2. Update PROGRESS.md: date, what I did, check status, mistakes, open
   questions, and the first task for next session.
3. Remind me to commit and push.

Also:
- If more than three days have passed since the last entry, start with a short
  review of the last step before anything new.
- If I'm stuck on the same task for two sessions, propose a smaller sub-task.
- Never let me skip a check to "come back to it later".

## Steps
0. (Changed 2026-11-17: no from-memory rewrite.) I write atomicMinFloat in
   src/step0.cu, both the atomicMin version and the compare-and-swap version,
   and explain why the loop retries and why integer atomicMin works on positive
   float bits. Check: it compiles on Colab and project.cu using it still gives
   0 pixels different.
1. Movable camera: R, t from position + yaw/pitch. Check: identity pose gives
   0 pixels different from the current depth image; moving back 1 m adds exactly
   1 to every depth.
2. Color buffer: pack depth and point index into 64 bits, atomicMin, resolve
   pass. Check: 0 pixels different from a CPU version.
3. Interactive loop: pose from input, re-render, display. Check: frame time
   consistent with bench numbers.
4. Equirectangular projection both ways: render the scene as a 360 depth image,
   unproject back. Check: round-trip error near 1e-6.
5. Registration on synthetic data: perturb pose and scale of the step-4
   panorama, recover them. Check: error against the known truth.
6. Real lidar block: load, subtract a local origin, color from aerial imagery.
7. Real panorama: depth model, sky mask, alignment, fusion.
Steps 1-5 use only the synthetic scene so the right answer is always known.

## Environment and workflow
- This machine has NO NVIDIA GPU. Never try to compile or run CUDA code here.
- All GPU code runs on Google Colab (T4). My loop: edit here, commit, push to
  GitHub, then run one launcher cell in Colab that pulls, compiles with nvcc,
  runs, and displays the output.
- You cannot see Colab. I will paste compiler errors and program output back to
  you. Never say GPU code works until I've pasted output showing it.
- Math and the CPU reference live in plain C++ headers that compile locally
  with g++. Run those tests here yourself; most testing needs no GPU.
- Git: wip commits go to a dev branch; merge to main when a step's check passes.
  Never commit output files (points.txt, pixels.txt, images, binaries).
- Repo: https://github.com/<your-username>/<repo-name> (public).
- Step 3 (interactive loop) can't open a window on Colab. When we reach it,
  discuss the options with me before building anything.

## Style
- Plain facts, no filler. Short answers.
- For design questions, discuss approaches with me before any code.
- Don't add features I didn't ask for, and don't jump ahead in the steps.
- Don't point me to reference implementations of a step until I've finished
  my own version of it.
