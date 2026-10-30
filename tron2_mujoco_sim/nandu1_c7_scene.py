"""Display-only MuJoCo checker floor/sky; the frozen asset is never rewritten."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


@contextmanager
def standard_scene_xml(source):
    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    # The temporary XML lives elsewhere; retain original asset resolution.
    for field in ("meshdir", "texturedir"):
        compiler.set(field, str((source.parent / compiler.get(field, ".")).resolve()))
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="c7_view_sky", type="skybox", builtin="gradient",
                  rgb1=".3 .5 .7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(assets, "texture", name="c7_view_grid", type="2d", builtin="checker",
                  rgb1=".2 .3 .4", rgb2=".1 .2 .3", mark="edge", markrgb=".8 .8 .8",
                  width="300", height="300")
    ET.SubElement(assets, "material", name="c7_view_floor", texture="c7_view_grid",
                  texuniform="true", texrepeat="5 5", reflectance=".15")
    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is None or floor.get("type") != "plane":
        raise ValueError("Standard scene requires the existing plane named floor")
    floor.set("material", "c7_view_floor")
    floor.set("rgba", "1 1 1 1")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(diffuse=".6 .6 .6", ambient=".3 .3 .3", specular="0 0 0")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", ".15 .25 .35 1")
    with tempfile.TemporaryDirectory(prefix="nandu-c7-display-") as directory:
        target = Path(directory) / "scene.xml"
        tree.write(target, encoding="utf-8", xml_declaration=True)
        yield target
