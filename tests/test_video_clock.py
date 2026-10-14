"""Video and SAM share the playback clock, including delayed video streams."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from percept_harness.media import probe_video, extract_frames, TimeSpan
from percept_harness.cv.timeline import probe_frame_timeline, materialize_sampled_frames

class VideoClockTests(unittest.TestCase):
    def test_delayed_video_keeps_initial_gap_and_includes_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); path=root/'delayed.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=10:duration=1',
                '-f','lavfi','-i','anullsrc=r=48000:cl=mono','-vf','setpts=PTS+0.2/TB',
                '-fps_mode','passthrough','-c:v','mpeg4','-c:a','aac','-t','1.3',str(path)],check=True)
            metadata=probe_video(path); timeline=probe_frame_timeline(path)
            self.assertAlmostEqual(timeline.frames[0].timestamp_seconds,.2,places=5)
            self.assertAlmostEqual(metadata.duration,1.2,places=5)
            self.assertLess(timeline.frames[-1].timestamp_seconds,metadata.duration)
            frames=extract_frames(path,TimeSpan(0,metadata.duration),5,root/'vlm')
            self.assertAlmostEqual(frames[0].timestamp,.2,places=5)
            self.assertGreaterEqual(frames[-1].timestamp,1.)
            self.assertTrue(all(any(abs(f.timestamp - t.timestamp_seconds) < 1e-6
                                    for t in timeline.frames) for f in frames))

    def test_nonzero_container_origin_is_removed_without_renumbering_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); path=root/'offset.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=10:duration=1',
                '-c:v','mpeg4','-output_ts_offset','5',str(path)],check=True)
            metadata=probe_video(path); timeline=probe_frame_timeline(path)
            self.assertEqual(timeline.frames[0].timestamp_seconds,0.)
            self.assertLess(timeline.frames[-1].timestamp_seconds,metadata.duration)
            sampled=materialize_sampled_frames(path,timeline,(2,7),root/'sam')
            self.assertEqual([f.source_frame_index for f in sampled.frames],[2,7])
            self.assertEqual([f.source_timestamp_seconds for f in sampled.frames],[.2,.7])
