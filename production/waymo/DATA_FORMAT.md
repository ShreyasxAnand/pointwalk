# Waymo data: what is in it, and what the converter writes

Two parts:

1. **Waymo's side**: the Parquet tables of the Waymo Open Dataset v2 (Perception).
2. **Our side**: the plain binary files `convert.py` writes for the C++/CUDA code.

Every fact is marked with where it comes from:

| Mark | Meaning |
| --- | --- |
| **[data]** | Printed from real data on 2026-11-24 by `inspect_segment.py`, segment `10023947602400723454_1120_000_1140_000`. One segment only. |
| **[notebook]** | Stated in Waymo's `tutorial_v2.ipynb` (in `references/`). |
| **[memory]** | From Waymo's documentation as remembered. Not in the notebook, not proven by the data. Treat as unconfirmed. |

To check this file against another segment:

```
python waymo/inspect_segment.py --input gs://waymo_open_dataset_v_2_0_1/training --segment <id>
```

## 1. Waymo's side

### Where the files are

One folder per table, one Parquet file per segment **[notebook]**:

```
<dataset>/<table>/<segment>.parquet
gs://waymo_open_dataset_v_2_0_1/training/lidar/10023947602400723454_1120_000_1140_000.parquet
```

The bucket is `waymo_open_dataset_v_2_0_1` (an underscore after the `v`). It
holds `training/`, `validation/`, `testing/` and `testing_location/` **[data]**.

A segment is one drive of about 20 seconds. A frame is one moment in it; frames
are 0.1 s apart (99905 to 100099 microseconds in the checked segment) **[data]**.

### Key columns

Rows in different tables belong together when their key columns are equal **[notebook]**.

| Column | Type | Meaning |
| --- | --- | --- |
| `key.segment_context_name` | string | the segment id (same as the file name) |
| `key.frame_timestamp_micros` | int64 | the frame's time, in microseconds |
| `key.laser_name` | int8 | which laser, 1 to 5 |

Most tables also carry a string column `index` (`<segment>;<timestamp>`). It is
not in the notebook's column list **[data]**. The converter does not use it.

The five lasers **[memory]** for the names, **[data]** for the positions (the
last column of each laser's extrinsic, in metres):

| `laser_name` | Name | Position on the car (x, y, z) |
| --- | --- | --- |
| 1 | TOP | 1.43, 0, 2.184 |
| 2 | FRONT | 4.07, 0, 0.689 |
| 3 | SIDE_LEFT | 3.245, 1.025, 0.979 |
| 4 | SIDE_RIGHT | 3.245, -1.025, 0.979 |
| 5 | REAR | -1.155, 0, 0.464 |

Waymo's vehicle axes are x forward, y left, z up **[memory]**. The positions
above agree with that.

### Table `lidar`: the range images

One row per (frame, laser). The checked segment has 199 frames x 5 lasers = 995
rows, sorted by time and then by laser, in 4 row groups. The file is 169 MB **[data]**.

| Column | Type | Meaning |
| --- | --- | --- |
| `[LiDARComponent].range_image_return1.values` | list of float32 | every number of the image, as one flat list |
| `[LiDARComponent].range_image_return1.shape` | 3 x int32 | `[H, W, C]`: rows, columns, numbers per box |
| `[LiDARComponent].range_image_return2.values` | list of float32 | the same for the second return |
| `[LiDARComponent].range_image_return2.shape` | 3 x int32 | |

A laser pulse can come back more than once; return 1 is the first echo. The
converter writes return 1 only.

**How the flat list is laid out.** The list has `H * W * C` numbers. It is
stored row after row, and inside a row box after box, with a box's `C` numbers
next to each other **[data]**:

```
position in the list = (row * W + col) * C + channel
```

**Sizes [data]**: the same in all 199 frames.

| Laser | H | W | C |
| --- | --- | --- | --- |
| 1 (TOP) | 64 | 2650 | 4 |
| 2, 3, 4, 5 | 200 | 600 | 4 |

**The 4 channels.** Names are **[memory]**; the value ranges are **[data]** from
the first frame.

| Channel | Name | Seen in the data |
| --- | --- | --- |
| 0 | range: straight-line distance, in metres | up to 74.97 (TOP), up to about 20 (the others) |
| 1 | intensity: how strong the echo was | up to 14656, most values small |
| 2 | elongation: how stretched the echo was | 0 to 1.49 |
| 3 | no-label-zone flag | -1 in every box of the first frame |

**A box with no return** holds -1 in every channel **[data]**: the count of
`-1` is identical in channels 0, 1 and 2 of every laser in the first frame.

### Table `lidar_calibration`: one row per laser

5 rows, in the file order 2, 5, 3, 4, 1 **[data]**. It has no timestamp: the
calibration is the same for the whole segment.

| Column | Type | Meaning |
| --- | --- | --- |
| `[LiDARCalibrationComponent].extrinsic.transform` | 16 x double | a 4x4 matrix, see below |
| `[LiDARCalibrationComponent].beam_inclination.min` | double | smallest beam angle, radians |
| `[LiDARCalibrationComponent].beam_inclination.max` | double | largest beam angle, radians |
| `[LiDARCalibrationComponent].beam_inclination.values` | list of double, or null | one angle per beam, radians |

**Beam inclinations [data].** A beam's inclination is its tilt above (+) or
below (-) level **[memory]** for the sign.

- **Laser 1 (TOP)** has a table of 64 angles, one per image row. The table is
  stored rising: first -0.3073, last +0.0406. The steps between them are not equal.
  Its `min` is -0.3128 and its `max` +0.0422, a little outside the table.
- **Lasers 2 to 5** have `null` for the table. They give only `min` =
  -1.5708 (-pi/2) and `max` = +0.5236 (pi/6) for their 200 rows.

**Extrinsic matrix layout [data].** The 16 numbers are the matrix read row by
row. The top-left 3x3 block is a rotation, the last column is a shift in
metres, and the last row is 0 0 0 1. Laser 1:

```
 -0.850487  -0.525982  -0.003966   1.430
  0.525971  -0.850495   0.003455   0.000
 -0.005190   0.000852   0.999986   2.184
  0.000000   0.000000   0.000000   1.000
```

It takes a point measured from the laser to the same point measured from the
vehicle **[memory]**. The shift (1.43, 0, 2.184) being a believable spot on the
roof agrees with that direction.

### Table `vehicle_pose`: one row per frame

199 rows, sorted by time, with the same timestamps as `lidar` **[data]**.

| Column | Type | Meaning |
| --- | --- | --- |
| `[VehiclePoseComponent].world_from_vehicle.transform` | 16 x double | a 4x4 matrix: vehicle -> world |

**Pose matrix layout [data].** Same layout as the extrinsic: read row by row,
rotation in the top-left 3x3, shift in the last column, last row 0 0 0 1. The
first frame:

```
  0.699059   0.715033   0.006630     860.594771
 -0.715058   0.698983   0.010772   15568.674927
  0.003068  -0.012271   0.999920    -651.441000
  0.000000   0.000000   0.000000       1.000000
```

The shifts are thousands of metres. A float holds about 7 digits: near 15,000
it can only step in units of about 1 mm, and the error grows each time
matrices are multiplied. That is why the converter keeps every matrix in
doubles.

### Tables for later (not converted yet)

Columns were printed from the real files; the contents were not read.

| Table | What it is | Checked |
| --- | --- | --- |
| `lidar_pose` | a pose for every box of the TOP laser's image (the car moves during one spin). 199 rows, laser 1 only, shape `[64, 2650, 6]`, float32. What the 6 numbers are: not checked. | **[data]** |
| `lidar_box` | labelled 3D boxes around objects: centre, size, heading, type, speed. 18633 rows. For removing moving objects. | **[data]** |
| `camera_image` | JPEG bytes per (frame, camera), with the camera's pose and shutter timing. 5 cameras, 343 MB. | **[data]** |
| `camera_calibration` | per camera: `f_u`, `f_v`, `c_u`, `c_v`, distortion `k1 k2 p1 p2 k3`, a 4x4 extrinsic, width, height. | **[data]** |

### Not in this file, on purpose

These belong to the range-image-to-points step and are not settled by anything
above:

- Which image row goes with which entry of the beam table.
- Which image column looks forward, and which way the columns turn.
- How to get beam angles for lasers 2 to 5 from only `min` and `max`.
- How Waymo's axes map to this project's axes (x right, y down, z forward).

## 2. Our side: what `convert.py` writes

```
python waymo/convert.py --input <folder> --segment <id> --out <dir>
```

`<folder>` holds `lidar/`, `lidar_calibration/` and `vehicle_pose/`. It can be
a local folder, `gs://...` or `s3://...`.

Add `--frames N` to write range images for the first N frames only (frame
numbers 0 to N - 1). `calibration.bin` and `poses.bin` are always written
whole. Two frames are 20.8 MB and take a few seconds.

The full output for the checked segment is 2.07 GB in 997 files:

```
<out>/calibration.bin
<out>/poses.bin
<out>/range/frame0000_laser1.bin
<out>/range/frame0000_laser2.bin
...
<out>/range/frame0198_laser5.bin
```

Rules for all three kinds of file:

- **Nothing is changed.** Every number is the one in the Parquet file: Waymo's
  order, Waymo's axes, metres and radians. Checked on 2026-11-24: all 995
  range images, 5 calibrations and 199 poses read back equal to the source.
- **Byte order** is little-endian, the native order on x86 and ARM. `int` is 4
  bytes, `float` 4, `double` 8, `long long` 8.
- **Read field by field** with `fread`, in the order of the tables below. Do
  not read a header into a struct in one go: the compiler may pad the struct.
- **Each file starts with 8 letters** naming its kind, like `PWSCENE1` in
  `scene.bin`.
- **Frame number** = the frame's place in time order, starting at 0. It is the
  same number in `poses.bin` and in the range file names.

### `range/frameFFFF_laserL.bin`: one range image

`FFFF` is the frame number with 4 digits, `L` the laser name.

| Bytes | Type | Field |
| --- | --- | --- |
| 8 | char[8] | the letters `PWRANGE1` |
| 8 | long long | timestamp in microseconds |
| 4 | int | frame number |
| 4 | int | laser name (1 to 5) |
| 4 | int | H, rows |
| 4 | int | W, columns |
| 4 | int | C, numbers per box (4) |
| H * W * C * 4 | float | the image, laid out as in Waymo's flat list |

The header is 36 bytes. Channel `c` of the box at (`row`, `col`) is float
number `(row * W + col) * C + c`. A box with no return holds -1.

File sizes: 2,713,636 bytes for laser 1, 1,920,036 bytes for lasers 2 to 5.

### `calibration.bin`: every laser

| Bytes | Type | Field |
| --- | --- | --- |
| 8 | char[8] | the letters `PWCALIB1` |
| 4 | int | number of lasers (5) |

Then, for each laser, smallest laser name first:

| Bytes | Type | Field |
| --- | --- | --- |
| 4 | int | laser name |
| 4 | int | N, the number of beam angles in the table (64 for laser 1, 0 when Waymo gives no table) |
| 16 * 8 | double | the extrinsic matrix, row by row |
| 8 | double | beam inclination min, radians |
| 8 | double | beam inclination max, radians |
| N * 8 | double | the beam angle table, in Waymo's stored order |

The checked segment gives 1284 bytes.

### `poses.bin`: every frame

| Bytes | Type | Field |
| --- | --- | --- |
| 8 | char[8] | the letters `PWPOSES1` |
| 4 | int | number of frames |

Then, for each frame in time order:

| Bytes | Type | Field |
| --- | --- | --- |
| 8 | long long | timestamp in microseconds |
| 16 * 8 | double | `world_from_vehicle`, row by row |

Each frame takes 136 bytes. The checked segment gives 27,076 bytes.

### Reading the files in C++

`io/waymo/waymofile.h` has one reader per kind of file: `readWaymoRange`,
`readWaymoCalibration` and `readWaymoPoses`. They return the numbers exactly
as stored: nothing is reordered, flipped or turned into angles. Each one
checks the 8 letters, the sizes in the header and the size of the file. On a
problem it prints one line to stderr naming the file and the reason, and
returns false.

`io/waymo/read_frame.cpp` reads one range image and prints its shape, its first
ranges and two counts. `inspect_segment.py` prints the same lines for the first
frame, so the two can be compared by eye:

```
g++ -std=c++17 -O2 -Wall -Wextra io/waymo/read_frame.cpp -o build/read_frame
./build/read_frame <out>/range/frame0000_laser1.bin
```
