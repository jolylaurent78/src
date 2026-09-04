# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.building.datastruct import Tree


# ``COLLECT`` place les DATA sous ``_internal``. Ces deux arbres doivent au
# contraire rester au niveau de l'executable, selon ApplicationPaths.
root_data = [
    (destination, source, "EXECUTABLE")
    for tree in (Tree("resources", prefix="resources"), Tree("defaults", prefix="defaults"))
    for destination, source, _typecode in tree
]


a = Analysis(
    ['src\\assembleur_tk.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AssembleurTriangles',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    root_data,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AssembleurTriangles',
)
