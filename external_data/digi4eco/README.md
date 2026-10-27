# DIGI4ECO fish tracking data

Source: [DIGI4ECO Underwater Fish Tracking Dataset](https://doi.org/10.5281/zenodo.20701225), Zenodo record 20701225. The record identifies the dataset as CC BY 4.0. Cite the dataset and preserve its attribution when using it.

The user approved downloading the archive for a bounded tracking validation. The archive contains four sequences; only the OBSEA video and its matching CVAT annotations were extracted for this project.

## Provenance and integrity

| Item | Value |
|---|---|
| Archive | `DIGI4ECO_tracking_dataset.zip` |
| Source size | 138,667,050 bytes |
| Source checksum | MD5 `e91d8df3e9f6a18527bcb854f98df962` |
| Local archive SHA-256 | `037c709c2ce059277c669430b83e9b1fad521c98773be90adc874c91cd0e910d` |
| Checksum verification | Local MD5 matched the Zenodo API record |
| Selected sequence | OBSEA |
| Video | `extracted/OBSEA/video.mp4`, 796,025 bytes; SHA-256 `f204115305485f7e5d681eaf41f7a632f9f135833c3b68863f3f3c37eb28408f` |
| Annotations | `extracted/OBSEA/annotations.xml`, 191,327 bytes; SHA-256 `658673a192ab6251c9a7213fc09dfc161f1f2ad1d5b544c4cda4add4903d7dcc` |
| CVAT annotation inventory | 12 tracks, 1,384 box entries, latest annotated frame index 149 |
| Default gate reference | 6 unique crossing tracks; computed from unoccluded annotation boxes at the tracker’s default gate |

The archive is retained so the extracted files can be independently checked. The other sequence files were not extracted. The source record and the project’s existing OBSEA image dataset share the OBSEA name/domain; this tracking clip is therefore not an independent second geographic domain.

## Validation state

The video was processed in an approved isolated CPU environment at `.venv-video` (Python 3.11.16, PyTorch 2.12.0+cpu, Ultralytics 8.4.190, OpenCV 5.0.0). The clip decodes to 148 frames although CVAT annotations span frames 0–149; 14 visible, non-occluded annotation boxes beyond the decoded video were excluded, leaving 1,327 boxes for frame-level comparison. The default run (`conf=0.25`) produced zero detections and zero track IDs, versus six reference gate crossings. A sensitivity run at `conf=0.001` produced 450 raw boxes (mean confidence 0.0041), but every box was below ByteTrack's `track_low_thresh=0.1`; it produced zero IDs and zero counts. Both runs therefore have 100% absolute count error on this clip. This is a failed model-tracking validation, not evidence of persistent-ID performance. See `docs/video-tracking-validation.md` and the JSON artifacts `outputs/digi4eco_obsea_tracking_primary_v2.json` and `outputs/digi4eco_obsea_tracking_conf001_thresholds.json`.

The archive and extracted data are excluded from Git by the root `.gitignore`. Keep them local; do not push these files to the project repository.

The small provenance file (`external_data/digi4eco/README.md`) is intentionally allowed through `.gitignore`; the archive, video, and XML remain excluded. The ground-truth-only gate extraction is saved as `outputs/digi4eco_obsea_gate_reference.json` and can be regenerated with `python scratch/analyze_cvat_gate_reference.py external_data/digi4eco/extracted/OBSEA/annotations.xml --output outputs/digi4eco_obsea_gate_reference.json` (the command refuses to overwrite an existing result).
