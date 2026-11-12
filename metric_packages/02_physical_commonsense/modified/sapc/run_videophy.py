"""SA/PC with pinned VideoPhy weights/templates and selectable sampling density."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from sampling import (experiment_output, official_rating,
                      open_video_plan, save_json, sha256)


def main(fixed_task=None):
    description = __doc__ if fixed_task is None else {
        'sa': 'Semantic Adherence with the current VideoPhy sampling protocol.',
        'pc': 'Physical Commonsense with the current VideoPhy sampling protocol.',
    }[fixed_task]
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input-csv', type=Path, required=True)
    if fixed_task is None:
        p.add_argument('--task', choices=['sa', 'pc'], required=True)
    else:
        p.set_defaults(task=fixed_task)
    p.add_argument('--sampling', choices=['official', 'half'], default='half',
                   help='half: ceil(N/2) frames, minimum 32 (author indices), maximum 256 over the whole clip (default)')
    p.add_argument('--frame-cap', type=int, choices=[256],
                   help='User-approved sampling cap for half mode: 256')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'official/videophy/VIDEOPHY2')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--preprocess-only', action='store_true')
    args = p.parse_args()
    if args.sampling != 'half' and args.frame_cap is not None:
        p.error('--frame-cap applies only to --sampling half')
    frame_cap = (256 if args.frame_cap is None else args.frame_cap) if args.sampling == 'half' else None
    if not os.environ.get('TMUX'):
        p.error('Enter a verified persistent tmux session before remote execution')
    output = experiment_output(args.output)
    with args.input_csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(args.source))
    import torch
    from PIL import Image
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
    from template import PROMPT_PHYSICS, PROMPT_SA
    tokenizer = LlamaTokenizer.from_pretrained(str(args.checkpoint))
    processor = MplugOwlProcessor(MplugOwlImageProcessor.from_pretrained(str(args.checkpoint)), tokenizer)
    generate_kwargs = dict(do_sample=False, top_k=1, temperature=0.001, max_length=256)
    checkpoint_config = json.loads((args.checkpoint / 'config.json').read_text())
    expected_visual_tokens = checkpoint_config['num_query_tokens'] + 1
    protocol = dict(version='sa_pc_min32_cap256_20261016_v1' if args.sampling == 'half' else 'sampling_experiment_20261012_v1', metric=args.task,
                    sampling=args.sampling,
                    frame_floor=32 if args.sampling == 'half' else None,
                    frame_floor_policy='When ceil(N/2) < 32, use exact author 32-frame indices, including duplicates for short clips',
                    frame_cap=frame_cap,
                    frame_cap_policy='When half sampling exceeds cap, resample whole clip with endpoints',
                    uncapped_source_frame_stride=2 if args.sampling == 'half' else None,
                    requested_frame_ratio=0.5 if args.sampling == 'half' else None,
                    expected_visual_tokens=expected_visual_tokens,
                    input_sha256=sha256(args.input_csv), entrypoint_sha256=sha256(__file__),
                    sampler_sha256=sha256(Path(__file__).with_name('sampling.py')),
                    official_inference_sha256=sha256(args.source / 'inference.py'),
                    official_template_sha256=sha256(args.source / 'template.py'),
                    official_processing_sha256=sha256(args.source / 'mplug_owl_video/processing_mplug_owl.py'),
                    checkpoint=str(args.checkpoint), device='cuda',
                    dtype='bfloat16',
                    preprocess_only=args.preprocess_only, generate_kwargs=generate_kwargs,
                    caption_policy='use input caption verbatim; no output-dependent rewriting',
                    sampling_matches_official=args.sampling == 'official')
    save_json(output / 'protocol.json', protocol)
    model = None
    visual_evidence = {}

    def observe_abstractor(module, inputs, outputs):
        visual_evidence['visual_token_count'] = int(outputs['last_hidden_state'].shape[1])

    results = []
    for index, row in enumerate(rows):
        record = dict(index=index, **row, status='pending', score=None)
        started = time.monotonic()
        try:
            reader, plan = open_video_plan(row['videopath'], family='videophy', mode=args.sampling,
                                          sample_fps=None, resize=224,
                                          frame_cap=frame_cap)
            record.update(video_sha256=sha256(row['videopath']), sampling=plan.to_dict())
            prompt = PROMPT_SA.format(caption=row['caption']) if args.task == 'sa' else PROMPT_PHYSICS
            inputs = processor(text=[prompt], return_tensors='pt')
            frames = [Image.fromarray(reader[i].asnumpy()).convert('RGB') for i in plan.indices]
            pixels = processor.image_processor(frames, return_tensors='pt')['pixel_values']
            inputs['video_pixel_values'] = pixels.unsqueeze(0).permute(0, 2, 1, 3, 4)
            record.update(video_tensor_shape=list(inputs['video_pixel_values'].shape),
                          prompt=prompt, prompt_tokens=int(inputs['input_ids'].shape[-1]))
            if args.preprocess_only:
                record['status'] = 'preprocess_verified'
            else:
                if model is None:
                    model = MplugOwlForConditionalGeneration.from_pretrained(str(args.checkpoint),
                                torch_dtype=torch.bfloat16, device_map={'': 'cpu'})
                    model.eval()
                    model = model.to('cuda').to(torch.bfloat16)
                    model.abstractor.register_forward_hook(observe_abstractor)
                inputs = {k: (v.bfloat16() if v.dtype == torch.float else v).to(model.device) for k, v in inputs.items()}
                visual_evidence.clear()
                with torch.no_grad():
                    generated = model.generate(**inputs, **generate_kwargs)
                record.update(visual_evidence)
                record['visual_tokens_match_checkpoint'] = record.get('visual_token_count') == expected_visual_tokens
                text = tokenizer.decode(generated.tolist()[0], skip_special_tokens=True)
                rating = official_rating(text)
                record.update(raw_output=text, official_numeric_score=rating,
                              parse_is_single_rating=bool(re.fullmatch(r'[1-5]|one|two|three|four|five', text.lower().strip())),
                              score=rating, status='complete')
        except Exception:
            record.update(status='failed', error=traceback.format_exc())
            record['elapsed_s'] = time.monotonic() - started
            results.append(record)
            save_json(output / 'rows.json', results)
            raise
        record['elapsed_s'] = time.monotonic() - started
        results.append(record)
        save_json(output / 'rows.json', results)
        print(json.dumps({k:record.get(k) for k in ('index','status','score','elapsed_s')}), flush=True)
    save_json(output / 'complete.json', dict(total=len(results),
              successful=sum(r['status'] in ('complete','preprocess_verified') for r in results)))
    return int(any(r['status'] not in ('complete','preprocess_verified') for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
