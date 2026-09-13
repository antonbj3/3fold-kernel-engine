"""Execute only explicitly declared recipes; report incomplete repository coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from verification_inventory_v1 import inventory


def pointer(value,path):
    if not path.startswith('/'):raise ValueError('JSON pointer must start with /')
    for token in path[1:].split('/'):
        token=token.replace('~1','/').replace('~0','~')
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value


def check_expected(root,expected,code,stdout):
    checks={'exit_code':code==expected['exit_code']}
    if 'report' in expected:
        path=(root/expected['report']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Report escapes repository')
        report=json.loads(path.read_text())
        for key,want in expected.get('json_equal',{}).items():
            actual=pointer(report,key);checks['equal:'+key]=actual==want and type(actual)==type(want)
        for key,want in expected.get('json_length',{}).items():checks['length:'+key]=len(pointer(report,key))==want
    if 'pytest_passed' in expected:
        found=re.findall(r'\b(\d+) passed\b',stdout);checks['pytest_passed']=found==[str(expected['pytest_passed'])]
    return checks


def run_recipes(root,selected,runtime):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if runtime=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    cache={};rows=[]
    for recipe in selected:
        signature=json.dumps({k:recipe.get(k) for k in ('argv','cwd','runtime','pythonpath')} | {'report':recipe['expected'].get('report')},sort_keys=True)
        reused=signature in cache
        if not reused:
            if 'report' in recipe['expected']:
                target=(root/recipe['expected']['report']).resolve()
                if not target.is_relative_to(root/'reports'):raise ValueError('Fresh report must stay under reports')
                target.unlink(missing_ok=True)
            argv=list(recipe['argv'])
            if argv[0]=='python':argv[0]=sys.executable
            child_env=dict(env)
            if recipe.get('pythonpath'):
                child_env['PYTHONPATH']=os.pathsep.join(str((root/v).resolve()) for v in recipe['pythonpath'])
            if recipe['runtime']=='cpu':child_env['CUDA_VISIBLE_DEVICES']=''
            child=subprocess.Popen(argv,cwd=root/recipe['cwd'],env=child_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            try:stdout,stderr=child.communicate(timeout=900);code=child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);stdout,stderr=child.communicate();code=124
            report_path=root/recipe['expected']['report'] if 'report' in recipe['expected'] else None
            report_hash=hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path and report_path.is_file() else None
            cache[signature]=(code,stdout,stderr,report_hash,recipe['key'])
        code,stdout,stderr,report_hash,executed_by=cache[signature]
        try:checks=check_expected(root,recipe['expected'],code,stdout.decode(errors='replace'));error=None
        except (KeyError,ValueError,FileNotFoundError,TypeError,IndexError) as exc:checks={'evidence_read':False};error=type(exc).__name__
        report_path=root/recipe['expected']['report'] if 'report' in recipe['expected'] else None
        current_hash=hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path and report_path.is_file() else None
        checks['fresh_execution_evidence']=current_hash==report_hash
        rows.append(dict(key=recipe['key'],executed_by=executed_by,reused_execution=reused,checks=checks,returncode=code,error=error,
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest(),
            report_sha256=hashlib.sha256((root/recipe['expected']['report']).read_bytes()).hexdigest() if 'report' in recipe['expected'] and (root/recipe['expected']['report']).is_file() else None))
        print(recipe['key']+' '+('PASS' if all(checks.values()) else 'FAIL'),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',choices=['cpu','modal'],default='cpu');p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1];state=inventory(root)
    recipes=json.loads((root/'docs/VERIFY_RECIPES.json').read_text());valid={r['key']:r['verification_recipe']=='DECLARED' for r in state['rows']}
    selected=[r for r in recipes if r['runtime']==args.runtime or args.runtime=='modal']
    out=root/'reports'/('verification_complete_v1.json' if args.require_complete else 'declared_verification_v1.json');out.unlink(missing_ok=True);rows=[]
    if args.require_complete and (not all(state['gates'].values()) or len(selected)!=state['declared_verified_rows']):
        out.write_text(json.dumps(dict(status='OWN-GATE-FAIL',complete_repository_coverage=False,numerical_execution='NOT_RUN',declared_rows=state['declared_verified_rows'],selected_recipes=len(selected),reason='Missing, stale or unavailable-runtime recipes'),indent=2)+'\n')
        print('Full verification refused: incomplete recipe coverage',flush=True);return 1
    if any(not valid.get(recipe['key']) for recipe in selected):raise ValueError('Stale or invalid declared recipe')
    rows=run_recipes(root,selected,args.runtime)
    gates=dict(nonempty_selected=bool(selected),selected_gates=bool(rows) and all(all(r['checks'].values()) for r in rows))
    result=dict(schema=2,executions=sum(not r['reused_execution'] for r in rows),runtime=args.runtime,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),recipes_sha256=hashlib.sha256((root/'docs/VERIFY_RECIPES.json').read_bytes()).hexdigest(),gates=gates,rows=rows,declared_verified_rows=state['declared_verified_rows'],
                complete_repository_coverage=state['gates']['explicit_recipe_coverage'] and len(rows)==state['declared_verified_rows'],
                status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                scope='Fresh selected declared recipes only. Unmapped rows and recipes for another runtime are NOT RUN, never implicitly passed.')
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');return int(not all(gates.values()))

if __name__=='__main__':raise SystemExit(main())
