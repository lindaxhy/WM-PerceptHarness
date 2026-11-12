"""CPU-only paired source-reference PSNR with explicit, frozen alignment rules."""
import argparse, ast, bisect, concurrent.futures as cf, hashlib, io, json, math, os
from pathlib import Path
import subprocess, time
from official_runtime import PACKAGE_ROOT, verify_official, read_manifest, require_persistent_remote_session

ROOT=None
SOURCES=PACKAGE_ROOT/'official'
REV='18dd7a19694e94aac21019170e3f5e63d6b4e19e'
MANIFEST=None
GLOBALS={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def command(args,timeout=120):
    return subprocess.run(args,capture_output=True,check=True,timeout=timeout).stdout

def first_png(path):
    return command(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),
                    '-map','0:v:0','-an','-sn','-dn','-frames:v','1','-c:v','png',
                    '-pix_fmt','rgb24','-threads','1','-f','image2pipe','pipe:1'])

def timing(path):
    data=json.loads(command(['ffprobe','-v','error','-select_streams','v:0',
              '-show_streams','-show_frames','-show_entries',
              'stream=width,height,avg_frame_rate,sample_aspect_ratio,color_transfer:frame=best_effort_timestamp_time',
              '-of','json',str(path)],timeout=240))
    pts=[float(f['best_effort_timestamp_time']) for f in data['frames']]
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Non-increasing/missing display timestamps')
    stream=data['streams'][0]
    if stream.get('sample_aspect_ratio') not in (None,'1:1','0:1','N/A'):raise ValueError('Non-square source pixels')
    if stream.get('color_transfer') in ('smpte2084','arib-std-b67'):raise ValueError('HDR reference requires a separate protocol')
    return stream,pts

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    verify_official()
    pinned=SOURCES/'pyiqa/psnr_arch.py'
    import importlib.metadata
    protocol={'metric':'psnr','name':'RGB source-reference PSNR appendix',
        'manifest_sha256':sha(MANIFEST),'adapter_sha256':sha(__file__),
        'official_psnr_revision':REV,'official_psnr_source_sha256':sha(pinned),
        'geometry_sources':{p.name:sha(p) for p in [SOURCES/'lingbot/pipeline_lingbot_video_i2v.py',SOURCES/'wan/utils.py',SOURCES/'wan/textimage2video.py']},
        'first_frame_verification':'FFmpeg first displayed frame, default autorotate, RGB24 PNG; exact historical conditioning PNG SHA256 required',
        'source_identity':'Every original source and generated video SHA256 verified against frozen manifest',
        'temporal_alignment':'Normalize each video to its first displayed PTS. For each generated frame choose nearest source PTS; tie selects earlier source. Accept only within source PTS support and absolute error <= 1/48 seconds. No extrapolation, retiming or optical-flow warping.',
        'spatial_alignment':'Exact pinned per-model I2V input geometry: Wan best_output_size(32,32,1280*704), PIL LANCZOS cover and floor center crop; LingBot original uint8 Torch bilinear cover and rounded center crop to832x480.',
        'conditioning_frame':'Exclude generated frame0 from primary mean; retain its score separately',
        'metric_parameters':{'test_y_channel':False,'crop_border':0,'data_range':1.0,'eps':1e-8},
        'aggregation':'Mean official per-frame dB within each video, then equal mean over successful videos; paired comparisons use common success set',
        'scope':'New frozen source-reference diagnostic, not a recovered historical gold protocol. Per-model native geometry differs; PSNR is sensitive to valid motion timing and is not perceptual quality.',
        'runtime':{n:importlib.metadata.version(n) for n in ['torch','numpy','pillow']},
        'ffmpeg':command(['ffmpeg','-version']).decode().splitlines()[0]}
    protocol['adapter_helper_sha256']=sha(Path(__file__).with_name('official_runtime.py'))
    protocol['fingerprint']=digest(protocol)
    existing=ROOT/'protocol.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=protocol:raise ValueError('Frozen PSNR protocol differs; use a new version')
    save(existing,protocol)
    return protocol

def initialize(output_dir):
    global ROOT
    ROOT=Path(output_dir)
    verify_official()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch, numpy as np
    from PIL import Image
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env={'torch':torch,'nn':torch.nn,'np':np,'Image':Image,'F':F,'math':math}
    tree=ast.parse((SOURCES/'pyiqa/psnr_arch.py').read_text(encoding='utf-8'))
    keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('psnr','PSNR')]
    if len(keep)!=2:raise ValueError('Pinned PSNR module structure differs')
    for node in keep:node.decorator_list=[]
    exec(compile(ast.Module(body=keep,type_ignores=[]),'official_psnr','exec'),env)
    tree=ast.parse((SOURCES/'lingbot/pipeline_lingbot_video_i2v.py').read_text(encoding='utf-8'))
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image')
    exec(compile(ast.Module(body=[method],type_ignores=[]),'official_lingbot_geometry','exec'),env)
    tree=ast.parse((SOURCES/'wan/utils.py').read_text(encoding='utf-8'))
    method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='best_output_size')
    exec(compile(ast.Module(body=[method],type_ignores=[]),'official_wan_size','exec'),env)
    env['scorer']=env['PSNR'](test_y_channel=False,crop_border=0,data_range=1.0,eps=1e-8)
    GLOBALS.update(env)

class Reader:
    def __init__(self,path,size,log):
        self.width,self.height=size
        self.log=log.open('wb')
        self.proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','1','-filter_threads','1',
             '-i',str(path),'-map','0:v:0','-an','-sn','-dn','-vsync','0','-pix_fmt','rgb24',
             '-threads','1','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=self.log)
        self.index=-1
        self.current=None
    def get(self,index):
        if index<self.index:raise ValueError('Decoder target moved backward')
        while self.index<index:
            payload=self.proc.stdout.read(self.width*self.height*3)
            if len(payload)!=self.width*self.height*3:raise ValueError('Unexpected decoded frame count')
            self.current=GLOBALS['np'].frombuffer(payload,dtype='uint8').reshape(self.height,self.width,3)
            self.index+=1
        return self.current
    def close(self):
        self.proc.stdout.close()
        if self.proc.poll() is None:self.proc.terminate()
        try:self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        self.log.close()

def transform(rgb,model,size):
    image=GLOBALS['Image'].fromarray(rgb)
    torch=GLOBALS['torch']
    width,height=size
    if model=='lingbot-dense':
        if size!=(832,480):raise ValueError('Unexpected LingBot native dimensions')
        return GLOBALS['preprocess_image'](None,image,height,width).squeeze(2)
    wanted=GLOBALS['best_output_size'](image.width,image.height,32,32,1280*704)
    if wanted!=size:raise ValueError(f'Wan geometry differs from pinned transform: {wanted} != {size}')
    scale=max(width/image.width,height/image.height)
    image=image.resize((round(image.width*scale),round(image.height*scale)),GLOBALS['Image'].Resampling.LANCZOS)
    left=(image.width-width)//2;top=(image.height-height)//2
    image=image.crop((left,top,left+width,top+height))
    return torch.from_numpy(GLOBALS['np'].array(image)).permute(2,0,1).unsqueeze(0).float()/255.0

def score_pair(pair,protocol):
    started=time.time();sid=pair[0]['sample_id'];folder=ROOT/'evidence'/sid
    folder.mkdir(parents=True,exist_ok=True)
    readers=[]
    try:
        done=[]
        for row in pair:
            p=ROOT/'results'/row['model']/(sid+'.json')
            if p.exists():
                value=json.loads(p.read_text(encoding='utf-8'))
                if value.get('protocol_fingerprint')!=protocol['fingerprint']:raise ValueError('Result protocol changed')
                if value.get('status')=='complete':
                    if sha(row['video_path'])!=row['recorded_final_sha256'] or sha(row['source_video_path'])!=row['source_sha256']: raise ValueError('Resume input bytes changed')
                    done.append(row['model'])
        if len(done)==len(pair):return {'sample_id':sid,'status':'already_complete'}
        source=pair[0]['source_video_path']
        if sha(source)!=pair[0]['source_sha256']:raise ValueError('Original source hash mismatch')
        png=first_png(source);png_hash=hashlib.sha256(png).hexdigest()
        if any(png_hash!=r['first_frame_sha256'] for r in pair):raise ValueError('Historical conditioning PNG cannot be exactly reproduced')
        image=GLOBALS['Image'].open(io.BytesIO(png)).convert('RGB')
        source_stream,source_pts=timing(source)
        source_time=[p-source_pts[0] for p in source_pts]
        source_reader=Reader(source,image.size,folder/'source_decode.log');readers.append(source_reader)
        models={};all_steps=[]
        for row in pair:
            if row['model'] in done:continue
            video=Path(row['video_path'])
            if sha(video)!=row['recorded_final_sha256']:raise ValueError('Generated video hash mismatch')
            stream,pts=timing(video)
            if len(pts)!=row['recorded_evaluation_frames']:raise ValueError('Recorded generated frame count differs')
            elapsed=[p-pts[0] for p in pts]
            if any(abs(t-i/24)>1e-5 for i,t in enumerate(elapsed)):raise ValueError('Generated video is not the recorded24fps timeline')
            size=(stream['width'],stream['height'])
            transform(GLOBALS['np'].array(image),row['model'],size)
            reader=Reader(video,size,folder/(row['model']+'_decode.log'));readers.append(reader)
            mappings=[]
            for j,t in enumerate(elapsed):
                k=bisect.bisect_left(source_time,t)
                choices=[x for x in (k-1,k) if 0<=x<len(source_time)]
                source_index=min(choices,key=lambda x:(abs(source_time[x]-t),x))
                error=source_time[source_index]-t
                accepted=(source_time[0]-1e-9<=t<=source_time[-1]+1e-9 and abs(error)<=1/48+1e-9)
                mappings.append({'generated_frame':j,'generated_time':t,'source_frame':source_index,'source_time':source_time[source_index],
                                  'error_seconds':error,'accepted':accepted,'conditioning_frame':j==0})
                if accepted:all_steps.append((source_index,j,row['model']))
            models[row['model']]={'row':row,'reader':reader,'size':size,'mappings':mappings,'scores':{},'pts':pts}
        for source_index,j,model in sorted(all_steps):
            item=models[model]
            ref=transform(source_reader.get(source_index),model,item['size'])
            gen=GLOBALS['torch'].from_numpy(item['reader'].get(j).copy()).permute(2,0,1).unsqueeze(0).float()/255.0
            with GLOBALS['torch'].no_grad():value=float(GLOBALS['scorer'](gen,ref).item())
            if not math.isfinite(value):raise ValueError('Non-finite official PSNR')
            item['scores'][j]=value
        for model,item in models.items():
            row=item['row'];scores=[v for j,v in item['scores'].items() if j>0]
            if not scores:raise ValueError('No valid non-conditioning frame pair')
            mapping_file=folder/(model+'_frames.json')
            save(mapping_file,{'source_pts':source_pts,'generated_pts':item['pts'],'frames':[{**m,'psnr_db':item['scores'].get(m['generated_frame'])} for m in item['mappings']]})
            value={'metric':'psnr','model':model,'sample_id':sid,'status':'complete','score':sum(scores)/len(scores),
                'conditioning_frame_psnr':item['scores'].get(0),'matched_scored_frames':len(scores),'generated_frames':len(item['pts']),
                'unmatched_frames':sum(not m['accepted'] for m in item['mappings']),
                'source_video_sha256':row['source_sha256'],'video_sha256':row['recorded_final_sha256'],
                'historical_first_frame_sha256':png_hash,'protocol_fingerprint':protocol['fingerprint'],
                'frame_evidence':str(mapping_file),'frame_evidence_sha256':sha(mapping_file),'elapsed_s':time.time()-started}
            save(ROOT/'results'/model/(sid+'.json'),value)
        save(folder/'source.json',{'source_path':source,'source_sha256':pair[0]['source_sha256'],'source_first_frame_sha256':png_hash,
             'source_stream':source_stream,'first_frame_size':list(image.size),'protocol_fingerprint':protocol['fingerprint']})
        return {'sample_id':sid,'status':'complete','elapsed_s':time.time()-started}
    except Exception as error:
        for row in pair:
            if row['model'] in locals().get('done',[]):continue
            save(ROOT/'results'/row['model']/(sid+'.json'),{'metric':'psnr','model':row['model'],'sample_id':sid,
                 'status':'failed','score':None,'video_sha256':row['recorded_final_sha256'],
                 'protocol_fingerprint':protocol['fingerprint'],'error':repr(error)})
        return {'sample_id':sid,'status':'failed','error':repr(error)}
    finally:
        for reader in readers:reader.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error('--limit must be positive')
    require_persistent_remote_session()
    if args.workers<1: parser.error('--workers must be positive')
    global ROOT,MANIFEST
    ROOT=args.output_dir.resolve()
    MANIFEST=args.manifest.resolve(strict=True)
    rows=read_manifest(MANIFEST,'psnr')
    protocol=prepare()
    pairs={}
    for row in rows:pairs.setdefault(row['sample_id'],[]).append(row)
    selected=list(pairs.values())[:args.limit] if args.limit else list(pairs.values())
    state={'expected':len(rows),'complete':0,'failed':0}
    with cf.ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(str(ROOT),)) as pool:
        futures=[pool.submit(score_pair,pair,protocol) for pair in selected]
        for future in cf.as_completed(futures):
            result=future.result();print(json.dumps(result),flush=True)
            values=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'results').glob('*/*.json')]
            state={'expected':len(rows),'complete':sum(v['status']=='complete' for v in values),
                   'failed':sum(v['status']=='failed' for v in values),'updated_unix':time.time()}
            save(ROOT/'progress.json',state)
    save(ROOT/'batch_finished.json',state)

if __name__=='__main__':main()
