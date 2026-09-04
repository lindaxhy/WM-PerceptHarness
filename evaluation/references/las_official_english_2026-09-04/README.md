# Official English LAS reference set (2026-09-04)

This directory freezes a new, separately generated reference set for the five
previously selected diagnostic videos. It does not replace the earlier Chinese
reference set used by the historical baseline in the comparison report.

The files under `references/` are the successful official LAS `final_summary`
JSON values, parsed and pretty-printed without changing their data. `prompt.txt`
is the exact English query used for every submission. `manifest.json` records
the immutable hashes, generation settings, validation result, and the one input
media exception. `comparison_with_postfix_local.json` stores the description-free
per-sample and aggregate metrics against the repaired five-sample local run.

Generation used official LAS operator `las_video_understanding`, version `v1`,
and model `doubao-seed-2-1-pro-260628`. All five outputs are valid JSON and use
English free text with no Han characters. They contain 31 semantic events, of
which 13 use an occlusion event type, plus 9 `occlusions` records. A separate
prompt-compliance audit found that `full_0004` omits enter/exit semantic events
for its two occlusion records and `full_0021` has three occlusion records but no
occlusion semantic event. These machine-only references are therefore retained
without manual correction and must not be treated as human ground truth.

`full_0002` is the only reference not generated from the byte-identical frozen
video. LAS failed to download the 35,714,259-byte original from each of three
temporary public endpoints. A video-only H.264 transcode preserved its complete
14.7-second, 441-frame timeline at 30 fps while reducing resolution from
1920x1080 to 1280x720 and removing audio. The manifest records both hashes. This
reference is useful for diagnosis but should be reported with that caveat.

Temporary public media URLs, LAS task identifiers, raw service envelopes, and
credentials are intentionally excluded. The temporary repository used to serve
the media was made private immediately after the successful outputs were
preserved; deletion is pending a GitHub credential with `delete_repo` scope.
