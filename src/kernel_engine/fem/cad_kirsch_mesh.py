#!/usr/bin/env python3
"""CAD -> KIRSCH PLATE MESH (gmsh) - a graded mesh that resolves the stress concentration.

stress3d reports Kt as a LOWER BOUND (a coarse mesh + cell means under-resolve the peak). To
resolve the TRUE concentration against analytic Kirsch, a FINE mesh at the hole is needed. The gmsh
curvature-based mesh grading (MeshSizeFromCurvature) gives ~40 elements around the hole and a coarse mesh far away.
A small hole (2a/W~0.13) approaches the infinite-plate Kirsch value (Kt=3). Writes the npz contract for warpfem_kirsch.py.

  python3 cad_kirsch_mesh.py
"""
import sys
from pathlib import Path

import numpy as np

W, H, A = 6.0, 6.0, 0.4
OUT = Path("data/cad_meshes")


def main():
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("kirsch")
    gmsh.model.occ.addRectangle(0, 0, 0, W, H)
    gmsh.model.occ.addDisk(W / 2, H / 2, 0, A, A)
    gmsh.model.occ.cut([(2, 1)], [(2, 2)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 48)     # ~48 elements around the hole (fine angular resolution)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
    gmsh.model.mesh.generate(2)

    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(ncoords).reshape(-1, 3)[:, :2]
    tag2idx = {int(t): i for i, t in enumerate(ntags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
    tris = None
    for et, en in zip(etypes, enodes):
        if et == 2:
            tt = np.array(en, dtype=np.int64).reshape(-1, 3)
            tris = np.vectorize(lambda t: tag2idx[int(t)])(tt)
    gmsh.finalize()

    r = np.sqrt((coords[:, 0] - W / 2) ** 2 + (coords[:, 1] - H / 2) ** 2)
    n_edge = int(np.sum(np.abs(r - A) < 0.02))                  # nodes on the hole edge
    a_ = coords[tris[:, 0]]; b_ = coords[tris[:, 1]]; c_ = coords[tris[:, 2]]
    area = float(0.5 * np.abs((b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1])
                              - (c_[:, 0] - a_[:, 0]) * (b_[:, 1] - a_[:, 1])).sum())
    rel = abs(area - (W * H - np.pi * A ** 2)) / (W * H - np.pi * A ** 2)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cad_kirsch.npz", positions=coords, tris=tris, W=W, H=H, A=A)
    ok = rel < 5e-2 and n_edge > 30
    print(f"  cad_kirsch: {len(coords)} nodes, {len(tris)} triangles, {n_edge} hole-edge nodes, area relative error {rel:.2e} "
          f"→ {OUT/'cad_kirsch.npz'}")
    print(f"\nVERDICT: Kirsch plate mesh {'GENERATED (graded, fine at the hole)' if ok else 'NOT OK'} - "
          f"{n_edge} nodes around the hole (angular resolution ~{360//max(n_edge,1)} deg) -> the sigma_tt(theta) peak can be resolved. "
          "Konsumeras av warpfem_kirsch.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
