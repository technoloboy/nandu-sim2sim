"""Build audited MJCF from the training USD physics and URDF mesh assembly."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd
import mujoco
import trimesh
import nandu1_contract as c

def fmt(v):
    return ' '.join(format(float(x), '.15g') for x in np.atleast_1d(v))

def quat(q):
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=float)

def wxyz(r):
    q = r.as_quat()
    return q[[3,0,1,2]]

def usd_rotation(q):
    return Rotation.from_quat(quat(q)[[1,2,3,0]])

def build_asset(source, dest):
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/'meshes', dest/'meshes', dirs_exist_ok=True)
    for mesh_path in (dest/'meshes').glob('*.STL'):
        trimesh.load_mesh(mesh_path, process=False).export(mesh_path.with_suffix('.obj'))
    shutil.copy2(source/'urdf/robot.urdf', dest/'robot.urdf')
    urdf = ET.parse(source/'urdf/robot.urdf').getroot()
    # Copied URDF stays usable next to its meshes.
    deployed_urdf = ET.parse(dest/'robot.urdf')
    for mesh in deployed_urdf.findall('.//mesh'):
        mesh.set('filename', 'meshes/'+Path(mesh.get('filename')).name)
    deployed_urdf.write(dest/'robot.urdf', encoding='utf-8', xml_declaration=True)
    stage = Usd.Stage.Open(str(source/'usd/robot.usd'))
    root_path = str(stage.GetDefaultPrim().GetPath())
    links = {x.get('name'): x for x in urdf.findall('link')}
    children = {x.find('child').get('link'): x for x in urdf.findall('joint')}
    xml = ET.Element('mujoco', model='NANDU1')
    ET.SubElement(xml,'compiler',angle='radian',meshdir='meshes',balanceinertia='false')
    ET.SubElement(xml,'option',timestep='0.001',gravity='0 0 -9.81',integrator='implicitfast')
    asset = ET.SubElement(xml,'asset')
    world = ET.SubElement(xml,'worldbody')
    ET.SubElement(world,'light',pos='0 0 4',dir='0 0 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='20 20 .1',rgba='.2 .25 .3 1',contype='1',conaffinity='2',friction='1 .005 .0001')
    actuators = ET.SubElement(xml,'actuator')
    sensors = ET.SubElement(xml,'sensor')
    for i,name in enumerate(c.JOINTS):
        ET.SubElement(actuators,'motor',name=name,joint=name,gear='1',ctrllimited='true',ctrlrange=fmt([-c.EFFORT[i],c.EFFORT[i]]))
        for tag,suffix in [('jointpos','pos'),('jointvel','vel'),('jointactuatorfrc','frc')]:
            ET.SubElement(sensors,tag,name=name+'_'+suffix,joint=name)

    def add_link(name,parent):
        prim = stage.GetPrimAtPath(root_path+'/'+name)
        args = {'name':name}
        joint = children.get(name)
        if joint is None:
            args['pos']='0 0 .65337'
        else:
            jp = stage.GetPrimAtPath(root_path+'/joints/'+joint.get('name'))
            if not jp:
                raise ValueError('USD joint not found: '+joint.get('name'))
            r0 = usd_rotation(jp.GetAttribute('physics:localRot0').Get())
            r1 = usd_rotation(jp.GetAttribute('physics:localRot1').Get())
            r = r0*r1.inv()
            p0 = np.array(jp.GetAttribute('physics:localPos0').Get())
            p1 = np.array(jp.GetAttribute('physics:localPos1').Get())
            args.update(pos=fmt(p0-r.apply(p1)),quat=fmt(wxyz(r)))
        body = ET.SubElement(parent,'body',**args)
        if joint is None:
            ET.SubElement(body,'freejoint',name='root')
        elif joint.get('type') != 'fixed':
            jname=joint.get('name'); i=c.JOINTS.index(jname)
            axis=np.eye(3)['XYZ'.index(jp.GetAttribute('physics:axis').Get())]
            lo=jp.GetAttribute('physics:lowerLimit').Get(); hi=jp.GetAttribute('physics:upperLimit').Get()
            ET.SubElement(body,'joint',name=jname,type='hinge',axis=fmt(r1.apply(axis)),pos=fmt(p1),
                limited='true',range=fmt(np.deg2rad([lo,hi])),damping='0',frictionloss='0',
                armature=str(c.ARMATURE[i]),actuatorfrclimited='true',actuatorfrcrange=fmt([-c.EFFORT[i],c.EFFORT[i]]))
        mass=prim.GetAttribute('physics:mass').Get()
        if mass is None: raise ValueError('USD missing mass for '+name)
        ET.SubElement(body,'inertial',mass=str(mass),pos=fmt(prim.GetAttribute('physics:centerOfMass').Get()),
            quat=fmt(quat(prim.GetAttribute('physics:principalAxes').Get())),diaginertia=fmt(prim.GetAttribute('physics:diagonalInertia').Get()))
        for kind in ['visual','collision']:
            for gidx,g in enumerate(links[name].findall(kind)):
                mesh=g.find('geometry/mesh')
                if mesh is None: raise ValueError('Unsupported geometry '+name)
                meshname=f'{name}_{kind}_{gidx}'
                ET.SubElement(asset,'mesh',name=meshname,file=Path(mesh.get('filename')).with_suffix('.obj').name,scale=mesh.get('scale','1 1 1'))
                origin=g.find('origin')
                rp=Rotation.from_euler('xyz',[float(v) for v in origin.get('rpy','0 0 0').split()])
                ET.SubElement(body,'geom',name=meshname,type='mesh',mesh=meshname,pos=origin.get('xyz','0 0 0'),quat=fmt(wxyz(rp)),
                    contype='2' if kind=='collision' else '0',conaffinity='1' if kind=='collision' else '0',
                    group='3' if kind=='collision' else '1',rgba='.7 .75 .8 1',friction='1 .005 .0001')
        if name in ['base_Link','up3']:
            prefix='base_imu' if name=='base_Link' else 'up3_imu'
            ET.SubElement(body,'site',name=prefix,pos='0 0 0',size='.005')
            ET.SubElement(sensors,'framequat',name=prefix+'_quat',objtype='site',objname=prefix)
            ET.SubElement(sensors,'gyro',name=prefix+'_gyro',site=prefix)
            ET.SubElement(sensors,'accelerometer',name=prefix+'_acc',site=prefix)
        for child,j in children.items():
            if j.find('parent').get('link')==name: add_link(child,body)
    add_link('base_Link',world)
    ET.indent(xml)
    path=dest/'robot.xml'
    ET.ElementTree(xml).write(path,encoding='utf-8',xml_declaration=True)
    return path

def audit_asset(source,path):
    """Independent URDF FK against USD-derived MJCF, plus exact USD inertials."""
    root=ET.parse(Path(source)/'urdf/robot.urdf').getroot()
    model=mujoco.MjModel.from_xml_path(str(path)); data=mujoco.MjData(model)
    rng=np.random.default_rng(12)
    max_p=max_r=max_mass=max_inertia=max_com=0.
    stage=Usd.Stage.Open(str(Path(source)/'usd/robot.usd'))
    limit_differences=[]
    for joint in root.findall('joint'):
        if joint.get('type')=='fixed':continue
        jp=stage.GetPrimAtPath('/nandu1/joints/'+joint.get('name'))
        usd_limit=np.deg2rad([jp.GetAttribute('physics:lowerLimit').Get(),jp.GetAttribute('physics:upperLimit').Get()])
        old=np.array([float(joint.find('limit').get(k)) for k in ['lower','upper']])
        if np.max(np.abs(old-usd_limit))>1e-9:
            limit_differences.append({'joint':joint.get('name'),'urdf_rad':old.tolist(),'usd_rad':usd_limit.tolist()})
    for n in range(6):
        q=c.DEFAULT_Q.copy() if n==0 else c.DEFAULT_Q+rng.uniform(-.1,.1,13)
        data.qpos[:7]=[0,0,.65337,1,0,0,0]
        for i,name in enumerate(c.JOINTS):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            data.qpos[model.jnt_qposadr[jid]]=q[i]
        mujoco.mj_forward(model,data)
        transforms={'base_Link':np.eye(4)};transforms['base_Link'][2,3]=.65337
        pending=list(root.findall('joint'))
        while pending:
            progress=False
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in transforms: continue
                o=j.find('origin');t=np.eye(4)
                t[:3,3]=[float(x) for x in o.get('xyz','0 0 0').split()]
                rot=Rotation.from_euler('xyz',[float(x) for x in o.get('rpy','0 0 0').split()])
                if j.get('type')!='fixed':
                    axis=np.array([float(x) for x in j.find('axis').get('xyz').split()])
                    rot=rot*Rotation.from_rotvec(axis*q[c.JOINTS.index(j.get('name'))])
                t[:3,:3]=rot.as_matrix();transforms[child]=transforms[parent]@t
                pending.remove(j);progress=True
            if not progress: raise ValueError('URDF disconnected')
        for name,t in transforms.items():
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
            max_p=max(max_p,float(np.linalg.norm(t[:3,3]-data.xpos[bid])))
            err=Rotation.from_matrix(t[:3,:3].T@data.xmat[bid].reshape(3,3)).magnitude()
            max_r=max(max_r,float(np.rad2deg(err)))
            mass=stage.GetPrimAtPath('/nandu1/'+name).GetAttribute('physics:mass').Get()
            max_mass=max(max_mass,abs(model.body_mass[bid]-mass))
            prim=stage.GetPrimAtPath('/nandu1/'+name)
            usd_r=usd_rotation(prim.GetAttribute('physics:principalAxes').Get()).as_matrix()
            mj_r=c.rotation(model.body_iquat[bid])
            usd_i=usd_r@np.diag(prim.GetAttribute('physics:diagonalInertia').Get())@usd_r.T
            mj_i=mj_r@np.diag(model.body_inertia[bid])@mj_r.T
            max_inertia=max(max_inertia,float(np.max(np.abs(usd_i-mj_i))))
            max_com=max(max_com,float(np.max(np.abs(model.body_ipos[bid]-prim.GetAttribute('physics:centerOfMass').Get()))))
    if max_p>.001 or max_r>.1 or max_mass>1e-6 or max_inertia>1e-6 or max_com>1e-6:
        raise ValueError(f'Asset audit failed {max_p}, {max_r}, {max_mass}, {max_inertia}, {max_com}')
    return {'max_position_error_m':max_p,'max_rotation_error_deg':max_r,'max_mass_error_kg':max_mass,
            'max_full_inertia_error':max_inertia,'max_center_of_mass_error_m':max_com,
            'actuators':model.nu,'urdf_vs_usd_limit_differences':limit_differences,
            'collision_note':'MuJoCo mesh convex hull versus PhysX convex decomposition; contact behavior requires sim2sim validation.'}
