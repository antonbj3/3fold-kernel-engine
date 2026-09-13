"""Readiness scanning never mistakes imports, comments or libraries for tests."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verification_inventory_v1 import inventory


def fixture(tmp_path,rows):
    (tmp_path/'src/example_engine').mkdir(parents=True);(tmp_path/'docs').mkdir()
    (tmp_path/'docs/RUNNING.md').write_text('| module | status | note |\n'+rows)
    return tmp_path


def test_real_guard_library_and_no_execution(tmp_path):
    root=fixture(tmp_path,'| a.py | VERIFIED-FRESH | |\n| b.py | VERIFIED-FRESH | |\n| c.py | OWN-GATE-FAIL | |\n')
    (root/'src/example_engine/a.py').write_text('raise RuntimeError("must not execute")\nif __name__ == "__main__":\n    pass\n')
    (root/'src/example_engine/b.py').write_text('# if __name__ == "__main__":\nx=1\n')
    report=inventory(root)
    assert report['counts']=={'python_main_candidate':1,'python_library':1}
    assert report['status']=='OWN-GATE-FAIL' and not report['gates']['explicit_recipe_coverage']
    assert all(r['numerical_execution']=='NOT_RUN' for r in report['rows'])


def test_alias_duplicates_and_missing_source(tmp_path):
    root=fixture(tmp_path,'| a.py | VERIFIED-FRESH | |\n| src/example_engine/a.py | VERIFIED-FRESH | |\n| missing.py | VERIFIED-FRESH | |\n')
    (root/'src/example_engine/a.py').write_text('x=1')
    report=inventory(root)
    assert not report['gates']['no_duplicate_declarations'] and not report['gates']['all_source_members_resolve']


def test_explicit_group_and_escape_refusal(tmp_path):
    root=fixture(tmp_path,'| src/example_engine/ (a.py, b.cu) | VERIFIED-FRESH | |\n')
    for name in ('a.py','b.cu'):(root/'src/example_engine'/name).write_text('')
    report=inventory(root);assert report['counts']=={'group_requires_recipe':1}
    (root/'docs/RUNNING.md').write_text('| ../escape.py | VERIFIED-FRESH | |\n')
    with pytest.raises(ValueError,match='escapes'):inventory(root)


def test_declared_recipe_binds_sources_and_status_note(tmp_path):
    root=fixture(tmp_path,'| a.py | VERIFIED-FRESH | decisive value1 |\n')
    (root/'src/example_engine/a.py').write_text('if __name__ == "__main__":\n    pass\n')
    row=inventory(root)['rows'][0]
    recipe=dict(key=row['key'],sources={m['path']:m['sha256'] for m in row['members']},
                note_sha256=row['note_sha256'],cwd='.',runtime='cpu',argv=['python','src/example_engine/a.py'],expected={'decisive_value':1})
    assert inventory(root,[recipe])['gates']['explicit_recipe_coverage']
    (root/'src/example_engine/a.py').write_text('changed=1')
    assert not inventory(root,[recipe])['gates']['explicit_recipe_coverage']
