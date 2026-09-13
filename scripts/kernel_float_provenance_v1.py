"""Catalogue observed cross-device mismatches without guessing instruction causes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import kernel_certificates_v1 as certificates
import kernel_matrix_local_report_v1 as matrix


def catalogue(report):
    differences=[]
    for row in report['rows']:
        for pair in row['comparisons']:
            if not pair['exact']:
                differences.append(dict(row=row['module'],left=pair['left'],right=pair['right'],
                    first_difference=pair['first_difference'],max_absolute_delta=pair['max_absolute_delta'],
                    operation_class='unresolved',
                    required_reproducer='Freeze inputs at the first differing record and capture intermediate operations on both devices; output hashes alone cannot distinguish FMA, transcendental implementations or reduction order'))
    integrity=all(v for k,v in report['gates'].items() if k!='cross_sku_exact')
    gates=dict(matrix_integrity=integrity,complete_pair_inventory=len(report['rows'])==20 and all(len(r['comparisons'])==6 for r in report['rows']),
               no_unattributed_difference=not differences)
    return dict(schema=1,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,
        compared_rows=len(report['rows']),pair_comparisons=sum(len(r['comparisons']) for r in report['rows']),
        source_matrix_sha256=certificates.sha(certificates.canonical(report)),differences=differences,
        operation_classes=dict(fma=[],transcendental=[],reduction_order=[]),
        scope='Kernel-family certified fixed-order observations only. Empty means no observed differing record in these fixtures; it does not prove that every floating-point operation is portable. Float-atomic diagnostic controls and other repositories are excluded.')


def main():
    folders=[ROOT/'reports/kernel_matrix_v1'/s for s in ('l4','a10g','h100')]+[ROOT/'reports/kernel_matrix_local_v1']
    report=matrix.compare([certificates.audit.load(p) for p in folders]);result=catalogue(report)
    out=ROOT/'reports/kernel_float_provenance_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'catalogue.json').write_bytes(certificates.canonical(result)+b'\n')
    print(json.dumps(dict(gates=result['gates'],differences=len(result['differences']))))
    return int(not all(result['gates'].values()))

if __name__=='__main__':raise SystemExit(main())
