"""Seven frozen Vulkan winding gates; explicit external baseline and executable.

Requires VULKAN_WINDING_BINARY, VULKAN_WINDING_SHADER and FIELD_REFERENCE_SOURCE.
Pass --allow-software only for diagnosis; hardware acceptance remains required.
Main writes reports/vulkan_winding.json and matching full synthetic arrays.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy import ndimage
import trimesh
ROOT=Path(__file__).resolve().parents[4]


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ray_sum(corners,origins):
    counts=np.zeros(len(origins),np.int32);direction=np.array([0.,0.,1.])
    for a,b,c in corners.astype(np.float64):
        e1=b-a;e2=c-a;h=np.cross(direction,e2);det=np.dot(e1,h)
        if det==0:continue
        q=origins.astype(np.float64)-a;u=q@h/det;k=np.cross(q,e1)
        v=k[:,2]/det;t=k@e2/det
        hit=(u>=0)&(v>=0)&(u+v<=1)&(t>0)&(t<=1e16)
        nz=np.cross(e1,e2)[2];counts[hit]+=int(nz>0)-int(nz<0)
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-software',action='store_true');args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_field',os.environ['FIELD_REFERENCE_SOURCE'])
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    binary=os.environ['VULKAN_WINDING_BINARY'];shader=os.environ['VULKAN_WINDING_SHADER']
    first=trimesh.creation.box(extents=[4,4,4]);offset=first.copy();offset.apply_translation([.13,-.21,.37])
    overlap=first.copy();overlap.apply_translation([1,.5,.25]);nested=trimesh.creation.box(extents=[2,2,2])
    reverse=first.copy();reverse.faces=reverse.faces[:,::-1]
    cases=dict(box=first,offset=offset,reversed=reverse,overlap=trimesh.util.concatenate([first,overlap]),nested=trimesh.util.concatenate([first,nested]))
    rows=[];arrays={}
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory)
        for name,mesh in cases.items():
            pitch=.25;origin=np.array([-4.,-4.,-4.])
            gmin,shape,_,solid,distance=baseline.surface_raster_and_flood(mesh.vertices,mesh.faces,pitch,origin)
            points=origin+gmin*pitch+np.indices(shape).reshape(3,-1).T*pitch
            points[:,0]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[0]
            points[:,1]+=pitch*baseline.JITTER_FRAC*baseline._GYLLENE[1]
            points=np.ascontiguousarray(points,dtype='<f4');corners=np.asarray(mesh.vertices[mesh.faces],dtype='<f4')
            oracle=ray_sum(corners,points).reshape(shape)
            payload=np.array([len(corners),len(points)],dtype='<u4').tobytes()+corners.tobytes()+points.tobytes()
            source=directory/(name+'.bin');source.write_bytes(payload)
            row=dict(case=name,voxels=int(solid.size),oracle_hash=digest(oracle),legs=[])
            arrays[name+'_oracle']=oracle;arrays[name+'_reference_distance']=distance
            for leg in range(2):
                output=directory/(name+str(leg)+'.out')
                cmd=[binary,shader,str(source),str(output)]+(['--allow-software'] if args.allow_software else [])
                child=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
                item=dict(exit=child.returncode,stderr=child.stderr)
                if child.returncode:
                    row['legs'].append(item);break
                metadata=json.loads(child.stdout)
                counts=np.fromfile(output,dtype='<i4').reshape(shape);mask=counts!=0
                dout=ndimage.distance_transform_edt(~mask,sampling=(pitch,)*3).astype(np.float32)
                din=ndimage.distance_transform_edt(mask,sampling=(pitch,)*3).astype(np.float32)
                sd=(dout-din).astype(np.float32);sd=(sd-np.sign(sd)*np.float32(.5*pitch)).astype(np.float32)
                item.update(software=metadata['software'],counts_hash=digest(counts),distance_hash=digest(sd),
                            count_mismatches=int(np.count_nonzero(counts!=oracle)),
                            occupancy_mismatches=int(np.count_nonzero(mask!=solid)),distance_mismatches=int(np.count_nonzero(sd!=distance)))
                row['legs'].append(item);arrays[f'{name}_counts_{leg}']=counts;arrays[f'{name}_distance_{leg}']=sd
            rows.append(row)
            if child.returncode:break
    legs=[q for r in rows for q in r['legs']]
    success=bool(legs) and all(q['exit']==0 for q in legs)
    gates=dict(runtime_success=success,
               full_repeat=success and all(len(r['legs'])==2 and r['legs'][0]['counts_hash']==r['legs'][1]['counts_hash'] and r['legs'][0]['distance_hash']==r['legs'][1]['distance_hash'] for r in rows),
               exact_counts=success and all(q['count_mismatches']==0 for q in legs),
               exact_occupancy=success and all(q['occupancy_mismatches']==0 for q in legs),
               exact_distance=success and all(q['distance_mismatches']==0 for q in legs),
               five_fixtures=len(rows)==5 and all(len(r['legs'])==2 for r in rows),
               hardware_execution=success and all(not q['software'] for q in legs))
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    report=dict(rows=rows,gates=gates,baseline_sha256=hashlib.sha256(Path(os.environ['FIELD_REFERENCE_SOURCE']).read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),shader_sha256=hashlib.sha256(Path(shader).read_bytes()).hexdigest(),
                scope='Normal-coordinate synthetic winding only; no throughput or subnormal portability claim.')
    (folder/'vulkan_winding.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'vulkan_winding_arrays.npz',**arrays)
    print(json.dumps(report));return 0 if all(gates.values()) else 2


if __name__=='__main__':raise SystemExit(main())
