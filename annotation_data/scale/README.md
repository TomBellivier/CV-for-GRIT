# Scale annotations

`scale_annotations.csv` is filled **by hand**: no tool writes it. It is one of
the three sources merged into `annotation_data/annotation_data.csv` by
`annotation_tools/build_annotation_data.py`.

One row per image, every column optional but `image_name`. Leave a cell empty
when the information does not exist — an empty cell means "not annotated", and
is carried through to the merged table as such.

| Column | Applies to | Meaning |
|---|---|---|
| `image_name` | all | file name with its extension, as in the pose annotations (e.g. `EB6.0OU.jpg`) |
| `scale_type` | all | `ruler` or `scale_bar` |
| `scale_px_per_mm` | all | the true scale of the image, in pixels per millimetre |
| `scale_bar_bbox_x` | scale bar | x of the bar's box, in pixels |
| `scale_bar_bbox_y` | scale bar | y of the bar's box, in pixels |
| `scale_bar_text` | scale bar | the text printed next to the bar, as read (e.g. `5 mm`) |
| `ruler_direction` | ruler | `horizontal` or `vertical` |
| `ruler_line_min` | ruler | first row (horizontal) or column (vertical) covered by the ruler |
| `ruler_line_max` | ruler | last row or column covered by the ruler |

`ruler_line_min` / `ruler_line_max` bound the band in which a detected ruler
line counts as correct; they are what `modules/ruler_detection` evaluates
against. They come from the older `../ruler_detection/ruler_db.csv`
(`Min`/`Max`), which this file replaces.

## Current content

Seeded from what already existed, so only the gaps are left to fill by hand:

- 160 rows from `../ruler_detection/ruler_db.csv` — direction and line range;
- 36 rows from `../whole_pipeline/pipeline_gt.csv` — `scale_px_per_mm`
  (`length [px] / value [mm]`), with no scale type recorded there;
- 3 images appear in both.

After editing this file, rebuild the merged table:

```bash
python annotation_tools/build_annotation_data.py
```
