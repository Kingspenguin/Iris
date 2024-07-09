# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.g1.g1_21_config import G1_21Cfg, G1_21CfgPPO

class G1CrawlCfg( G1_21Cfg ):
    class env( G1_21Cfg.env ):
        num_envs = 500
        num_actions = 21

        n_demo_steps = 2
        n_demo = 9 + 3 + 3 + 3 +6 * 3  #observe height
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 3
        n_priv_latent = 4 + 1 + num_actions * 2
        n_proprio = 3 + 2 + 2 + num_actions * 3 + 2 # one hot
        history_len = 10

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        n_command = 3

        # num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv # with motion
        num_observations = 1 + n_command + n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 20 # episode length in seconds
        
        num_policy_actions = 21
        randomize_start_pos = True
    
    class init_state( G1_21Cfg.init_state ):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        rot = [0.0, 1.0, 0.0, 0.707] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 4.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # joint two four six must be non-zero (see urdf)

        hip_pitch = -2.35
        hip_roll = 0.3
        hip_yaw = 0.

        knee = 1.6

        ankle_pitch = -0.68
        ankle_roll = 0.

        shoulder_pitch = -1.4
        shoulder_roll = 0.3
        shoulder_yaw = 0.2

        elbow_pitch = 1.0
        elbow_roll = 0.

        zero_joint = 0.
        one_joint = 0.6
        two_joint = 0.9

        three_joint = 1.2
        four_joint = 1.2

        five_joint = 1.2
        six_joint = 1.2

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': hip_pitch,
            'left_hip_roll_joint': hip_roll,
            'left_hip_yaw_joint': hip_yaw,
            'left_knee_joint': knee,
            'left_ankle_pitch_joint': ankle_pitch,
            'left_ankle_roll_joint': ankle_roll,
            'right_hip_pitch_joint': hip_pitch,
            'right_hip_roll_joint': -hip_roll,
            'right_hip_yaw_joint': -hip_yaw,
            'right_knee_joint': knee,
            'right_ankle_pitch_joint': ankle_pitch,
            'right_ankle_roll_joint': -ankle_roll,
            ################################
            'torso_joint': 0,
            'left_shoulder_pitch_joint': shoulder_pitch,
            'left_shoulder_roll_joint': shoulder_roll,
            'left_shoulder_yaw_joint': shoulder_yaw,
            'left_elbow_pitch_joint': elbow_pitch,
            'right_shoulder_pitch_joint': shoulder_pitch,
            'right_shoulder_roll_joint': -shoulder_roll,
            'right_shoulder_yaw_joint': -shoulder_yaw,
            'right_elbow_pitch_joint': elbow_pitch,
        }

    class control( G1_21Cfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        hand_stiffness = 0.5
        hand_damping = 0.003
        stiffness = {'hip_yaw': 80,
                     'hip_roll': 120,
                     'hip_pitch': 120,
                     'knee': 120,
                     'ankle_pitch': 15,
                     'ankle_roll': 10,
                     'torso': 150,
                     'shoulder_pitch': 120,
                     'shoulder_roll': 120,
                     'shoulder_yaw': 40,
                     "elbow_pitch": 80,
                     "elbow_roll": 80,
                     'zero': hand_stiffness,
                     'one': hand_stiffness,
                     'two': hand_stiffness,
                     'three': hand_stiffness,
                     'four': hand_stiffness,
                     'five': hand_stiffness,
                     'six': hand_stiffness,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 1,
                     'hip_roll': 2,
                     'hip_pitch': 3,
                     'knee': 2,
                     'ankle_pitch': 0.05,
                     'ankle_roll': 0.05,
                     'torso': 4,
                     'shoulder_pitch': 3,
                     'shoulder_roll': 3,
                     'shoulder_yaw': 1,
                     "elbow_pitch":2,
                     "elbow_roll":1.5,
                     'zero': hand_damping,
                     'one': hand_damping,
                     'two': hand_damping,
                     'three': hand_damping,
                     'four': hand_damping,
                     'five': hand_damping,
                     'six': hand_damping,
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( G1_21Cfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_blue_red_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_custom_21.urdf'
        torso_name = "torso_link"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["shoulder", "elbow", "hip", "torso_link"]
        terminate_after_contacts_on = ["shoulder_roll"] #"torso_link", ]#, "thigh", "calf"]
        height_termination_rigid_body_name = ["torso", "pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( G1_21Cfg.rewards ):
        base_height_target = 0.75
        class scales:
            # tracking rewards
            # alive = 5
            height = 10
            # base_height = 0
            tracking_vx = 3
            tracking_ang_vel = 3
            # tracking_vx_sqr = -0.1
            # tracking_ang_vel_sqr = -0.1
            # tracking_demo_goal_vel = 1.0
            # tracking_mul = 6
            # tracking_lin_vel = 6
            # stand_still = 3
            # tracking_goal_vel = 4


            # tracking_demo_yaw = 1
            # tracking_demo_roll_pitch = 1
            # tracking_demo_dof_pos = 3
            # tracking_demo_dof_vel = 1.0
            # tracking_demo_key_body = 2
            # tracking_demo_height = 1  # useful if want better height tracking
            # tracking_demo_lin_vel = 1
            # tracking_demo_ang_vel = 0.5

            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.4
            orientation = -2
            # dof_acc = -3e-7
            collision = -10.
            action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -1e-5
            energy = -1e-3
            # hip_pos = -0.5
            # dof_error = -0.1
            # feet_edge = -1
            dof_pos_limits = -10.0
            feet_stumble = -2
            feet_drag = -0.1
            feet_air_time = 15
            feet_height = 2
            feet_force = 0.03
            # termination = 0
            # hip_roll = -3
            # hip_yaw = -3
            # zerocommand_action_rate = -1

            # periodic action movement
            hip_periodic_sqr = -0.002
            knee_periodic_sqr = -0.002
            # hip_periodic_norm = -1e-4
            # knee_periodic_norm = -1e-4
            # arm_periodic_sqr = -0.002
            arm_action = -0.3
            # arm_action_sqr = -10
            # vx = 3

        only_positive_rewards = False
        clip_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2
        class ranges:
            lin_vel_x = [0.4, 1.0] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]#[0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-1.6, 1.6]

class G1orzCfg( G1_21Cfg ):
    class env( G1_21Cfg.env ):
        num_envs = 500
        num_actions = 21

        n_demo_steps = 2
        n_demo = 9 + 3 + 3 + 3 +6 * 3  #observe height
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 3
        n_priv_latent = 4 + 1 + num_actions * 2
        n_proprio = 3 + 2 + 2 + num_actions * 3 + 2 # one hot
        history_len = 10

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        n_command = 3

        # num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv # with motion
        num_observations = 1 + n_command + n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 20 # episode length in seconds
        
        num_policy_actions = 21
        randomize_start_pos = True
    
    class init_state( G1_21Cfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        rot = [0.0, 1.0, 0.0, 0.707] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # joint two four six must be non-zero (see urdf)

        hip_pitch = -1.6
        hip_roll = 0.
        hip_yaw = 0.

        knee = 1.6

        ankle_pitch = 0.73
        ankle_roll = 0.

        shoulder_pitch = -1.3
        shoulder_roll = 0.3
        shoulder_yaw = 0.

        elbow_pitch = 0.8
        elbow_roll = 0.

        zero_joint = 0.
        one_joint = 0.6
        two_joint = 0.9

        three_joint = 1.2
        four_joint = 1.2

        five_joint = 1.2
        six_joint = 1.2

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': hip_pitch,
            'left_hip_roll_joint': hip_roll,
            'left_hip_yaw_joint': hip_yaw,
            'left_knee_joint': knee,
            'left_ankle_pitch_joint': ankle_pitch,
            'left_ankle_roll_joint': ankle_roll,
            'right_hip_pitch_joint': hip_pitch,
            'right_hip_roll_joint': -hip_roll,
            'right_hip_yaw_joint': -hip_yaw,
            'right_knee_joint': knee,
            'right_ankle_pitch_joint': ankle_pitch,
            'right_ankle_roll_joint': -ankle_roll,
            ################################
            'torso_joint': 0,
            'left_shoulder_pitch_joint': shoulder_pitch,
            'left_shoulder_roll_joint': shoulder_roll,
            'left_shoulder_yaw_joint': shoulder_yaw,
            'left_elbow_pitch_joint': elbow_pitch,
            'left_elbow_roll_joint': -elbow_roll,
            'left_five_joint': -five_joint,
            'left_six_joint': -six_joint,
            'left_three_joint': -three_joint,
            'left_four_joint': -four_joint,
            'left_zero_joint': zero_joint,
            'left_one_joint': one_joint,
            'left_two_joint': two_joint,
            'right_shoulder_pitch_joint': shoulder_pitch,
            'right_shoulder_roll_joint': -shoulder_roll,
            'right_shoulder_yaw_joint': -shoulder_yaw,
            'right_elbow_pitch_joint': elbow_pitch,
            'right_elbow_roll_joint': elbow_roll,
            'right_five_joint': five_joint,
            'right_six_joint': six_joint,
            'right_three_joint': three_joint,
            'right_four_joint': four_joint,
            'right_zero_joint': zero_joint,
            'right_one_joint': -one_joint,
            'right_two_joint': -two_joint,
        }

    class control( G1_21Cfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        hand_stiffness = 0.5
        hand_damping = 0.003
        stiffness = {'hip_yaw': 80,
                     'hip_roll': 200,
                     'hip_pitch': 160,
                     'knee': 200,
                     'ankle_pitch': 15,
                     'ankle_roll': 10,
                     'torso': 150,
                     'shoulder': 80,
                     "elbow_pitch": 120,
                     "elbow_roll": 80,
                     'zero': hand_stiffness,
                     'one': hand_stiffness,
                     'two': hand_stiffness,
                     'three': hand_stiffness,
                     'four': hand_stiffness,
                     'five': hand_stiffness,
                     'six': hand_stiffness,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 1,
                     'hip_roll': 3,
                     'hip_pitch': 10,
                     'knee': 3,
                     'ankle_pitch': 0.05,
                     'ankle_roll': 0.05,
                     'torso': 4,
                     'shoulder': 1.5,
                     "elbow_pitch":1.5,
                     "elbow_roll":1.5,
                     'zero': hand_damping,
                     'one': hand_damping,
                     'two': hand_damping,
                     'three': hand_damping,
                     'four': hand_damping,
                     'five': hand_damping,
                     'six': hand_damping,
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( G1_21Cfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_blue_red_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_custom_21.urdf'
        torso_name = "torso_link"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["shoulder", "elbow", "hip", "torso_link"]
        terminate_after_contacts_on = []#"torso_link", ]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( G1_21Cfg.rewards ):
        base_height_target = 0.75
        class scales:
            # tracking rewards
            # alive = 5
            height = 10
            # base_height = 0
            tracking_vx = 3
            tracking_ang_vel = 3
            # tracking_vx_sqr = -0.1
            # tracking_ang_vel_sqr = -0.1
            # tracking_demo_goal_vel = 1.0
            # tracking_mul = 6
            # tracking_lin_vel = 6
            # stand_still = 3
            # tracking_goal_vel = 4


            # tracking_demo_yaw = 1
            # tracking_demo_roll_pitch = 1
            # tracking_demo_dof_pos = 3
            # tracking_demo_dof_vel = 1.0
            # tracking_demo_key_body = 2
            # tracking_demo_height = 1  # useful if want better height tracking
            # tracking_demo_lin_vel = 1
            # tracking_demo_ang_vel = 0.5

            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.4
            orientation = -2
            # dof_acc = -3e-7
            collision = -10.
            action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -1e-5
            energy = -1e-3
            # hip_pos = -0.5
            # dof_error = -0.1
            # feet_edge = -1
            dof_pos_limits = -10.0
            feet_stumble = -2
            feet_drag = -0.1
            feet_air_time = 15
            feet_height = 2
            feet_force = 0.03
            # termination = 0
            # hip_roll = -3
            # hip_yaw = -3
            # zerocommand_action_rate = -1

            # periodic action movement
            hip_periodic_sqr = -0.002
            knee_periodic_sqr = -0.002
            # hip_periodic_norm = -1e-4
            # knee_periodic_norm = -1e-4
            # arm_periodic_sqr = -0.002
            arm_action = -0.3
            # arm_action_sqr = -10
            # vx = 3

        only_positive_rewards = False
        clip_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2
        class ranges:
            lin_vel_x = [0.4, 1.0] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]#[0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-1.6, 1.6]

class G1CrawlCfgPPO( G1_21CfgPPO ):
    class runner( G1_21CfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
        record_video_interval = 200
    
    class policy( G1_21CfgPPO.policy ):
        continue_from_last_std = True
        text_feat_input_dim = G1CrawlCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = G1CrawlCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( G1_21CfgPPO.algorithm ):
        entropy_coef = 0.005

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1e-3
        hidden_dims = [128, 64]
        priv_states_dim = G1CrawlCfg.env.n_priv
        priv_start = G1CrawlCfg.env.n_feature + G1CrawlCfg.env.n_proprio + G1CrawlCfg.env.n_demo + G1CrawlCfg.env.n_scan
        
        prop_start = G1CrawlCfg.env.n_feature
        prop_dim = G1CrawlCfg.env.n_proprio
