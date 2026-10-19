"""Simulator-independent robustness schedule and batched reset geometry."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

_COLUMNS = {
    'tilt_deg': (1.,2.,3.,4.,5.), 'com': (.005,.01,.02,.035,.05),
    'root_v': (.05,.1,.2,.35,.5), 'root_w': (.05,.1,.2,.35,.5),
    'joint_offset': (.03,.08,.15,.3,.5), 'joint_pos_noise': (.002,.005,.01,.015,.02),
    'joint_vel_noise': (.05,.15,.4,.8,1.5), 'gyro_noise': (.01,.025,.05,.1,.2),
    'gravity_noise': (.005,.01,.02,.035,.05), 'force': (0.,2.,4.,7.,10.),
    'torque': (0.,.5,1.,2.,4.), 'z_kick': (0.,.03,.06,.1,.15),
    'angular_kick': (0.,.04,.08,.12,.2),
}
STAGES = tuple({k: v[i] for k,v in _COLUMNS.items()} for i in range(5))

def uniform_noise(value, amplitude):
    return value + torch.empty_like(value).uniform_(-amplitude, amplitude)

def pulse_gain(elapsed, duration):
    """20%-duration cosine ramps, exactly zero before/after a pulse."""
    fraction = (elapsed / duration.clamp_min(1e-6)).clamp(0.,1.)
    ramp = torch.minimum(fraction,1.-fraction).div(.2).clamp(0.,1.)
    return .5 - .5*torch.cos(math.pi*ramp)

def wheel_sensor_indices(names):
    return [list(names).index(n) for n in ('wheel_L_Link','wheel_R_Link')]

def obb_collisions(centers, rotations, half, pairs):
    """Exact separating-axis test for the supplied conservative OBB proxies."""
    if not pairs:
        return torch.zeros(len(centers),dtype=torch.bool,device=centers.device)
    p = torch.as_tensor(pairs,device=centers.device)
    a,b=p[:,0],p[:,1]
    ra,rb=rotations[:,a],rotations[:,b]
    axes_a=ra.transpose(-1,-2);axes_b=rb.transpose(-1,-2)
    cross=torch.linalg.cross(axes_a[:,:,:,None,:],axes_b[:,:,None,:,:],dim=-1).flatten(2,3)
    axes=torch.cat((axes_a,axes_b,cross),dim=2)
    distance=((centers[:,b]-centers[:,a]).unsqueeze(2)*axes).sum(-1).abs()
    extent_a=((axes@ra).abs()*half[a][None,:,None,:]).sum(-1)
    extent_b=((axes@rb).abs()*half[b][None,:,None,:]).sum(-1)
    separated=(distance>extent_a+extent_b+1e-6).any(-1)
    return (~separated).any(-1)

def rpy_rotation(rpy):
    r,p,y=rpy.unbind(-1);cr,sr,cp,sp,cy,sy=r.cos(),r.sin(),p.cos(),p.sin(),y.cos(),y.sin()
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
                        sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(-1,3,3)

class ResetGeometry:
    """URDF FK plus CAD-mesh OBBs, constructed once and evaluated on GPU.

    Tree-distance <=2 pairs are mechanical interfaces, not collision candidates.
    OBB intersections are narrowed by Pinocchio mesh collision at reset only.
    """
    def __init__(self, urdf, joint_names, device='cpu'):
        import trimesh
        root=ET.parse(urdf).getroot(); self.device=device
        links={x.get('name'):x for x in root.findall('link')}
        self.names=['base_Link'];pending=list(root.findall('joint'));self.joints=[]
        while pending:
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in self.names: continue
                o=j.find('origin'); xyz=[float(v) for v in o.get('xyz','0 0 0').split()]
                rpy=[float(v) for v in o.get('rpy','0 0 0').split()]
                axis=[float(v) for v in j.find('axis').get('xyz').split()] if j.find('axis') is not None else [1,0,0]
                idx=list(joint_names).index(j.get('name')) if j.get('type')!='fixed' else None
                self.joints.append((self.names.index(parent),idx,torch.tensor(xyz,device=device),
                                    rpy_rotation(torch.tensor([rpy],device=device))[0],torch.tensor(axis,dtype=torch.float,device=device)))
                self.names.append(child);pending.remove(j)
        centers=[];rots=[];halves=[];self.vertices=[]
        for name in self.names:
            vertices=[]
            for collision in links[name].findall('collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None: raise ValueError('NANDU reset geometry requires mesh collisions')
                loaded=trimesh.load(Path(urdf).parent/mesh.get('filename'),force='mesh')
                verts=np.asarray(loaded.vertices)*np.array([float(v) for v in mesh.get('scale','1 1 1').split()])
                o=collision.find('origin')
                xyz=[float(v) for v in o.get('xyz','0 0 0').split()] if o is not None else [0,0,0]
                rpy=[float(v) for v in o.get('rpy','0 0 0').split()] if o is not None else [0,0,0]
                rotation=rpy_rotation(torch.tensor([rpy],dtype=torch.float))[0].numpy()
                vertices.append(verts@rotation.T+xyz)
            verts=np.concatenate(vertices)
            from scipy.spatial import ConvexHull
            self.vertices.append(torch.tensor(verts[ConvexHull(verts).vertices],dtype=torch.float,device=device))
            # PCA box is deterministic, cheap, and encloses all CAD vertices.
            axes=np.linalg.eigh(np.cov(verts.T))[1]
            projected=verts@axes;lo,hi=projected.min(0),projected.max(0)
            centers.append(axes@((lo+hi)/2));rots.append(axes);halves.append((hi-lo)/2)
        self.centers=torch.tensor(np.array(centers),dtype=torch.float,device=device)
        self.rotations=torch.tensor(np.array(rots),dtype=torch.float,device=device)
        self.half=torch.tensor(np.array(halves),dtype=torch.float,device=device)
        parents={i+1:j[0] for i,j in enumerate(self.joints)}
        def distance(a,b):
            chain={a:0};n=a
            while n in parents: chain[parents[n]]=chain[n]+1;n=parents[n]
            steps=0;n=b
            while n not in chain: n=parents[n];steps+=1
            return steps+chain[n]
        self.pairs=[(a,b) for a in range(len(self.names)) for b in range(a+1,len(self.names)) if distance(a,b)>2]
        self.wheels=wheel_sensor_indices(self.names)
        import pinocchio as pin
        self.pin=pin;self.joint_names=list(joint_names)
        self.model=pin.buildModelFromUrdf(str(urdf))
        self.geom=pin.buildGeomFromUrdf(self.model,str(urdf),pin.GeometryType.COLLISION,
                                      package_dirs=[str(Path(urdf).parent)])
        mapping={x.name.rsplit('_',1)[0]:i for i,x in enumerate(self.geom.geometryObjects)}
        for a,b in self.pairs:
            self.geom.addCollisionPair(pin.CollisionPair(mapping[self.names[a]],mapping[self.names[b]]))
        self.data=self.model.createData();self.gdata=pin.GeometryData(self.geom)

    def exact_collisions(self,q):
        """Narrow phase at reset only; CPU mesh BVH, no dynamics stepping."""
        result=[]
        for row in q.detach().cpu().numpy():
            pq=self.pin.neutral(self.model)
            for name,value in zip(self.joint_names,row):
                joint=self.model.joints[self.model.getJointId(name)]
                if joint.nq==1:pq[joint.idx_q]=value
                else:pq[joint.idx_q:joint.idx_q+2]=[math.cos(value),math.sin(value)]
            result.append(self.pin.computeCollisions(self.model,self.data,self.geom,self.gdata,pq,True))
        return torch.tensor(result,device=q.device,dtype=torch.bool)

    def forward(self,q,root_rotation):
        positions=[torch.zeros(len(q),3,device=q.device)];rotations=[root_rotation]
        eye=torch.eye(3,device=q.device).expand(len(q),3,3)
        for parent,idx,offset,origin,axis in self.joints:
            base=rotations[parent]@origin
            positions.append(positions[parent]+(rotations[parent]@offset[:,None]).squeeze(-1))
            if idx is None: rotations.append(base);continue
            x,y,z=axis;zero=x*0
            skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
            theta=q[:,idx,None,None]
            motion=eye+theta.sin()*skew+(1-theta.cos())*(skew@skew)
            rotations.append(base@motion)
        return torch.stack(positions,1),torch.stack(rotations,1)

    def evaluate(self,q,root_rotation):
        pos,rot=self.forward(q,root_rotation)
        centers=pos+(rot@self.centers[None,:,:,None]).squeeze(-1)
        box_rotation=rot@self.rotations
        bottom=torch.stack([pos[:,i,2]+(rot[:,i,2,:]@v.T).min(-1).values for i,v in enumerate(self.vertices)],1)
        root_z=.003-bottom[:,self.wheels].min(-1).values
        nonwheel=[i for i in range(len(self.names)) if i not in self.wheels]
        legal=(bottom[:,nonwheel]+root_z[:,None]>.008).all(-1)
        legal &= (bottom[:,self.wheels].max(-1).values-bottom[:,self.wheels].min(-1).values)<.08
        broad=legal & obb_collisions(centers,box_rotation,self.half,self.pairs)
        ids=broad.nonzero().flatten()
        if len(ids):legal[ids] &= ~self.exact_collisions(q[ids])
        return legal,root_z
