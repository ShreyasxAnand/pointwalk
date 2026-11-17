# Progress

## Status

Legend: `todo` not started, `wip` in progress, `pass` check passed and merged to main.

| Step | What | Check | Status |
| --- | --- | --- | --- |
| setup | Repo, launcher, sources extracted from notebook | Launcher prints "0 of 307200 pixels differ" on Colab | wip: sources extracted to src/ from notebooks/3d.ipynb; launcher not yet run on Colab (needs repo URL and push) |
| 0 | Write atomicMinFloat both ways (atomicMin, compare-and-swap) and explain them | Compiles on Colab; project.cu using it still gives 0 pixels different | wip: both versions written in src/step0.cu and read correct; NOT compiled or run (Colab run pending) |
| 1 | Movable camera: R, t from position + yaw/pitch | Identity pose: 0 pixels differ from current depth image; back 1 m adds exactly 1 to every depth | wip: started 2026-11-17 (math on paper first) |
| 2 | Color buffer: 64-bit packed depth + index, atomicMin, resolve | 0 pixels differ from CPU version | todo |
| 3 | Interactive loop: pose from input, re-render, display | Frame time consistent with bench numbers | todo |
| 4 | Equirectangular projection both ways | Round-trip error near 1e-6 | todo |
| 5 | Registration on synthetic data | Error against known truth | todo |
| 6 | Real lidar block: load, local origin, color from imagery | (define when reached) | todo |
| 7 | Real panorama: depth model, sky mask, alignment, fusion | (define when reached) | todo |

## Session log

### 2026-11-16 (setup)
- Did: created CLAUDE.md, repo layout (src/, tests/, notebooks/), .gitignore,
  README.md, notebooks/launcher.ipynb, this file. `git init` on main, nothing
  committed yet.
- Check status: baseline NOT confirmed. project.cu and bench.cu are not in src/
  yet: no .ipynb was found in the project folder.
- Mistakes: none recorded.
- Open questions:
  - Where is the original notebook? (Answered 2026-11-17: notebooks/3d.ipynb.)
  - GitHub repo URL (placeholder in CLAUDE.md and in the launcher's REPO_URL).
- First task next session: put the notebook in the folder, extract the two
  files, push, run the launcher, paste the output to confirm
  "0 of 307200 pixels differ". Then step 0.

### 2026-11-17
- Did: notebook added as notebooks/3d.ipynb; project.cu and bench.cu extracted
  to src/. Wrote src/step0.cu: atomicMinFloat (integer atomicMin on the bits)
  and atomicMinFloatGeneric (compare-and-swap loop, float comparison). Written
  with line-by-line review, notebook available.
- Check status: both functions read correct. Not compiled, not run. Baseline
  launcher run on Colab still not done (no GitHub repo URL yet).
- Mistakes: used `__global__` on a helper (needs `__device__`); treated
  atomicMin's return value as the new value (it returns the old one) and added
  an unprotected write after it; compared against a second read of the cell
  instead of the snapshot `old`; tried `__float_as_int` on a pointer.
- Explain-it questions: 2 of 3 right. Missed why the loop re-checks after a
  failed swap (on the recall list).
- Rule change: no memorization drills or from-memory rewrites. State facts
  directly, check understanding with a question, move on. Step 0 reworded.
- Open questions: GitHub repo URL; Colab run of the baseline and of step0.cu.
- Next: step 1, starting with t for a camera moved back 1 m with no rotation.

## Recall list

Questions missed in warm-ups. Each is asked again in a later session and
removed after two correct answers in a row.

| Added | Question | Times missed | Last asked |
| --- | --- | --- | --- |
| 2026-11-17 | Without `-fmad=false`, what does the compiler do, and why does that change the bits of a result? (Got "combines operations"; missed which operations and the rounding reason.) | 1 | 2026-11-17 |
| 2026-11-17 | Pinhole: x and y fixed, depth doubles. What happens to the pixel's distance from the image centre? (Got the direction; the amount was wrong.) | 1 | 2026-11-17 |
| 2026-11-17 | In the compare-and-swap loop, the swap fails. What does `prev` hold, why is it saved into `old`, and why must the loop re-check the condition before swapping again? (Thought `old` then held an unwanted value and that `val` could be lost.) | 1 | 2026-11-17 |
