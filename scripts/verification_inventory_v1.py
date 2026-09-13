"""Inventory declared verified rows without importing or executing their modules."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def python_entry(path):
    if path.suffix!='.py':return dict(kind='non_python',main_lines=[])
    tree=ast.parse(path.read_text(),filename=path.name);lines=[]
    for node in tree.body:
        if not isinstance(node,ast.If) or not isinstance(node.test,ast.Compare):continue
        test=node.test
        if len(test.ops)!=1 or not isinstance(test.ops[0],ast.Eq):continue
        pair=[test.left,*test.comparators]
        if len(pair)==2 and any(isinstance(x,ast.Name) and x.id=='__name__' for x in pair) and any(isinstance(x,ast.Constant) and x.value=='__main__' for x in pair):lines.append(node.lineno)
    return dict(kind='python_main_candidate' if lines else 'python_library',main_lines=lines)


def resolve(root,package,label):
    # Support the existing explicit directory/member status row as one declaration.
    match=re.fullmatch(r'(.+/) \(([^)]+)\)',label)
    labels=[match[1]+name.strip() for name in match[2].split(',')] if match else [label]
    paths=[]
    for item in labels:
        found=None
        for base in (root,root/'src'/package):
            candidate=(base/item).resolve()
            if not candidate.is_relative_to(root.resolve()):raise ValueError('Status path escapes repository')
            if candidate.is_file():found=candidate;break
        if found is None:return []
        paths.append(found)
    return paths


def inventory(root,recipes=None):
    root=root.resolve();doc=root/'docs/RUNNING.md'
    packages=[p.name for p in (root/'src').iterdir() if p.is_dir() and p.name.endswith('_engine')]
    if len(packages)!=1:raise ValueError('One engine package required')
    package=packages[0];rows=[];seen=set();duplicates=[]
    for number,line in enumerate(doc.read_text().splitlines(),1):
        cells=line.split('|')
        if not line.startswith('|') or len(cells)<4 or cells[2].strip()!='VERIFIED-FRESH':continue
        label=cells[1].strip().strip('`');paths=resolve(root,package,label)
        key=tuple(str(p.relative_to(root)) for p in paths) if paths else (label,)
        if key in seen:duplicates.append(dict(line=number,label=label))
        seen.add(key);members=[]
        for path in paths:
            entry=python_entry(path);members.append(dict(path=str(path.relative_to(root)),sha256=digest(path),**entry))
        if len(members)!=1:kind='group_requires_recipe' if members else 'unresolved'
        else:kind=members[0]['kind']
        note='|'.join(cells[3:-1]).strip()
        rows.append(dict(key=';'.join(key),label=label,line=number,note_sha256=hashlib.sha256(note.encode()).hexdigest(),members=members,kind=kind,
                         verification_recipe='UNSPECIFIED',numerical_execution='NOT_RUN'))
    recipe_path=root/'docs/VERIFY_RECIPES.json'
    recipes=recipes if recipes is not None else (json.loads(recipe_path.read_text()) if recipe_path.exists() else [])
    if not isinstance(recipes,list):raise ValueError('Recipes must be a list')
    mapping={r['key']:r for r in recipes};recipe_errors=[]
    for row in rows:
        recipe=mapping.get(row['key'])
        if recipe is None:continue
        expected_sources={m['path']:m['sha256'] for m in row['members']}
        cwd=(root/recipe.get('cwd','')).resolve()
        pythonpath=recipe.get('pythonpath',[])
        path_ok=isinstance(pythonpath,list) and all(isinstance(v,str) and (root/v).resolve().is_relative_to(root) and (root/v).is_dir() for v in pythonpath)
        valid=(path_ok and recipe.get('sources')==expected_sources and bool(expected_sources) and
               recipe.get('note_sha256')==row['note_sha256'] and cwd.is_relative_to(root) and cwd.is_dir() and
               recipe.get('runtime') in ('cpu','modal') and isinstance(recipe.get('argv'),list) and bool(recipe['argv']) and
               all(isinstance(a,str) and a for a in recipe['argv']) and isinstance(recipe.get('expected'),dict) and bool(recipe['expected']))
        row['verification_recipe']='DECLARED' if valid else 'INVALID'
        if not valid:recipe_errors.append(row['key'])
    recipe_inventory_ok=len(mapping)==len(recipes) and set(mapping)<=set(r['key'] for r in rows)
    counts={kind:sum(r['kind']==kind for r in rows) for kind in sorted({r['kind'] for r in rows})}
    gates=dict(nonempty_declared_inventory=bool(rows),all_source_members_resolve=all(r['members'] for r in rows),
               no_duplicate_declarations=not duplicates,explicit_recipe_coverage=bool(rows) and recipe_inventory_ok and all(r['verification_recipe']=='DECLARED' for r in rows))
    return dict(schema=1,package=package,document_sha256=digest(doc),declared_verified_rows=len(rows),counts=counts,
        gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',duplicates=duplicates,recipe_errors=recipe_errors,rows=rows,
        scope='Readiness inventory only. A Python main guard does not establish correct arguments, working directory, dependencies, hardware, gates or expected numbers. No numerical module was imported, run or skipped as passing. Explicit execution/evidence recipes remain required for every declared row.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=inventory(args.root);output=args.output or args.root/'reports/verification_inventory_v1.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('package','declared_verified_rows','counts','gates','status')}))
    return int(not all(result['gates'].values()))

if __name__=='__main__':raise SystemExit(main())
