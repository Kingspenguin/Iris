from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.g1.g1_21 import G1_21

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

class G1Crawl(G1_21):
    pass

    def step(self, actions):
        actions[:, 1] = torch.clamp(actions[:, 1], min=0.3)
        actions[:, 7] = torch.clamp(actions[:, 7], max=-0.3)
        return super().step(actions)
    

    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        termination_height = torch.any(self.rigid_body_states[:, self.termination_height_indices, 2] < 0.3, dim = 1)

        self.reset_buf |= termination_height

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= roll_cutoff
        # self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    def _reward_tracking_vx(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, 1:3]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        # print(self.commands)
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 0])
        rew = torch.clamp(rew, max=4)
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, 1:3]), dim=1)
    