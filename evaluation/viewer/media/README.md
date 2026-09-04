# Local demo media

Keep the five frozen source videos in this directory under the names below.
The MP4 files are intentionally covered by the repository-wide `*.mp4` ignore
rule; only this README is versioned.

| Filename | Expected SHA-256 |
|---|---|
| `full_0001.mp4` | `c3243c46bad68d3b2772e82648e45b68e75a1893b0ce27edecd450226464c1e9` |
| `full_0002.mp4` | `6741b6184b847b6096e4282b1e0f76142714870b258ca16a19606e0441f0973f` |
| `full_0024.mp4` | `a7a696bcdd835c083b27ca3705d13a2f22e069ebec9038581354fed39e6fbbe8` |
| `full_0021.mp4` | `1cd6b0752bd3b7f1ca987d470d8e65d9247a4f0751c7f891c6824b63751eff05` |
| `full_0004.mp4` | `34bc1833f713419c694c97af544f5c4148f0b03e02913415f4a3eb2be8660d09` |

Verify a local copy from the repository root with:

```bash
shasum -a 256 evaluation/viewer/media/*.mp4
```

Do not force-add these files. Anyone without the authorized media can still use
the viewer's local file picker; the selected file is not persisted or uploaded.
