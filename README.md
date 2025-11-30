# spherical-tiling-3d

Python tools for generating 3D-printable models of **spherically tiled surfaces**.

The current focus is on **regular `{p,q}` tilings** (Platonic solids) on a spherical surface, intended as a basis for 3D-printable models (e.g. puzzle spheres, educational objects, etc.).

> ⚠️ **Status:** Work in progress / not stable.  
> API, directory structure, and functionality may change at any time.

## Goal

- Construct regular `{p,q}` tilings on the sphere via Wythoff construction  
- Derive polygonal faces and spherical triangle meshes with configurable resolution  
- Prepare geometry for 3D printing (e.g. individual tiles with defined wall thickness, indentations, colors)

## Installation (dev)

```bash
git clone https://github.com/maxthematics/spherical-tiling-3d.git
cd spherical-tiling-3d
pip install -e .
