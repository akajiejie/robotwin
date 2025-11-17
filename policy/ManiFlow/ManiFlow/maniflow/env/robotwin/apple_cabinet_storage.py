from .base_task import Base_task
from .utils import *
import sapien


class apple_cabinet_storage(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags,table_static=False)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors()
        self.step_lim = 600

    def load_actors(self, **kwargs):
        self.cabinet, _ = rand_create_urdf_obj(
            self.scene,
            modelname="036_cabine",
            xlim=[-0.01,0],
            ylim=[0.18,0.19],
            zlim=[0.985],
            rotate_rand=False,
            qpos=[1,0,0,1],
            scale=0.3
        )

        self.cabinet_active_joints = self.cabinet.get_active_joints()
        for joint in self.cabinet_active_joints:
            joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")
        self.cabinet_all_joints = self.cabinet.get_joints()

        self.apple,_ = rand_create_obj(
            self.scene,
            xlim=[0.18,0.3],
            ylim=[-0.2,-0.05],
            zlim=[0.78],
            modelname="035_apple",
            rotate_rand=False,
            convex=True
        )
        self.apple.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        
    def play_once(self):
        pass
        
    def check_success(self):
        cabinet_pos = self.cabinet.get_pose().p
        eps = 0.03
        apple_pose = self.apple.get_pose().p
        left_endpose = self.get_left_endpose_pose()
        target_pose = [-0.05,-0.1,0.89,-0.505,-0.493,-0.512,-0.488]
        eps1 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01])
        return abs(apple_pose[0]-cabinet_pos[0])<eps and abs(apple_pose[1]+0.06-cabinet_pos[1])<eps and apple_pose[2] > 0.79 and np.all(abs(np.array(left_endpose.p.tolist() + left_endpose.q.tolist()) - target_pose) < eps1)