#!/usr/bin/env python3
"""CAD -> 3D TET MESH (gmsh) - a real 3D CAD part made simulation-ready as a volume mesh.

The 2D bridge (`cad_to_femmesh.py`) handles planar faces (a BRepMesh surface shell). REAL parts are 3D SOLIDS and
need a VOLUME tet mesh (BRepMesh does not produce tet volumes; gmsh/netgen are required). This builds a 3D CAD solid
(box minus a cylindrical hole) in the gmsh OpenCASCADE kernel, generates a tet VOLUME mesh, and writes the mesh contract
(npz: positions + tetrahedra) that warp.fem consumes for 3D physics. The same hot-swap contract
som 2D, men 3D-volym.

  python3 cad_to_tetmesh.py
"""
import sys
from pathlib import Path

import numpy as np

LX, LY, LZ = 2.0, 1.0, 1.0
HOLE_R = 0.25
MESH_SIZE = 0.12
OUT = Path("data/cad_meshes")


def main():
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box_hole")
    gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)                                  # 3D CAD-solid
    gmsh.model.occ.addCylinder(LX / 2, LY / 2, -0.1, 0, 0, LZ + 0.2, HOLE_R)    # through hole
    gmsh.model.occ.cut([(3, 1)], [(3, 2)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    gmsh.model.mesh.generate(3)

    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(ncoords).reshape(-1, 3)
    tag2idx = {int(t): i for i, t in enumerate(ntags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
    tets = None
    for et, en in zip(etypes, enodes):
        if et == 4:        # 4 = linear tetrahedron
            tt = np.array(en, dtype=np.int64).reshape(-1, 4)
            tets = np.vectorize(lambda t: tag2idx[int(t)])(tt)
    gmsh.finalize()

    # tet-volym (Cayley-Menger via |det|/6) vs analytisk (box − cylinder)
    p = coords[tets]
    vol = float(np.abs(np.einsum("ij,ij->i",
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])).sum() / 6.0)
    vexp = LX * LY * LZ - np.pi * HOLE_R ** 2 * LZ
    rel = abs(vol - vexp) / vexp
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cad_box3d.npz", positions=coords.astype(np.float64), tets=tets.astype(np.int32),
             LX=LX, LY=LY, LZ=LZ, hole_r=HOLE_R)
    ok = rel < 5e-2 and tets is not None
    print(f"  cad_box3d: {len(coords)} noder, {len(tets)} tetraedrar, volym={vol:.4f} "
          f"(expected box minus cylinder {vexp:.4f}, relative error {rel:.2e}) -> {OUT/'cad_box3d.npz'}")
    print(f"\nVERDICT: CAD -> 3D tet mesh {'GENERATED + volume verified' if ok else 'VOLUME MISMATCH'} - a real "
          + ("3D CAD solid (box minus cylindrical hole, OCC kernel) tet-VOLUME meshed, the volume matches the analytic value -> "
             "consumed by warpfem_mms_cad3d.py. " if ok else "tet-volym ≠ geometri. ")
          + "CAVEAT: linear tet connectivity (warp.fem builds the P2 basis); gmsh Delaunay.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
