# CAD_Engraver

**Generate a printable housing with an ArUco tag extruded into the geometry — give it a tag ID, get an STL.**

Fiducial-tracked robotics has a boring failure mode: printed paper tags peel, curl, glare, and shift by a millimetre at exactly the wrong moment. When your pose estimate needs to be good to a tenth of a millimetre, a sticker is a liability.

The fix is to stop attaching tags and start *building* them. This tool takes an ArUco tag ID and produces a mesh with that tag's pattern extruded directly into the part surface — so the fiducial is printed as part of the object, dimensionally exact and physically incapable of moving relative to the thing it marks.

---

## Usage

```bash
pip install -r requirements.txt
python engraver.py        # enter an ArUco tag ID when prompted
```

The generated STL is written to the files folder. `app.py` provides a UI wrapper over the same generator.

## Files

| file | role |
|---|---|
| `engraver.py` | the generator — tag pattern to extruded mesh |
| `app.py` | interface wrapper |
| `highousing.stl`, `housingchanged.stl`, `new housing 1-29.stl` | base housing geometry, successive revisions |
| `Printing_Tag9/10/11.stl` | example outputs, tags 9–11 |

## Related

Built for the fiducial-tracking work in **[AprilTag_Detection](https://github.com/ShrishChou/AprilTag_Detection)** and **[AUTO-Arm](https://github.com/ShrishChou/AUTO-Arm)**, where the tagged parts are picked at 0.1 mm precision.
