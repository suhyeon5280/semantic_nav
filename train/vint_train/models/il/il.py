import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder, MultiLayerDecoder_mask, MultiLayerDecoder_mask2, MultiLayerDecoder_mask3

from typing import List, Dict, Optional, Tuple, Callable

class IL_dist(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_dist, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        """
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * 2),
            nn.Sigmoid()               
        )
        """        
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )        

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            #print(obsgoal_img.size(), obs_img.size())
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size())
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return dist_pred, action_pred
        
class IL_gps(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_gps, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            """
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features = self.goal_encoder._fc.in_features
            """
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        """
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()
        """
        self.compress_goal_enc = nn.Identity()
        
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        """
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * 2),
            nn.Sigmoid()               
        )
        """        
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        #self.dist_predictor = nn.Sequential(
        #    nn.Linear(32, 1),
        #)        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )        
        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            #print(obsgoal_img.size(), obs_img.size())
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)

        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size
        """
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size())
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        #dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return action_pred        

class IL_gps_map_mask(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_gps_map_mask, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=9) # context
            self.num_obs_features_map = self.goal_encoder._fc.in_features            
            """
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features = self.goal_encoder._fc.in_features
            """
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
            
        if self.num_obs_features_map != self.obs_encoding_size:
            self.compress_obs_enc_map = nn.Linear(self.num_obs_features_map, self.obs_encoding_size)
        else:
            self.compress_obs_enc_map = nn.Identity()            
            
        """
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()
        """
        self.compress_goal_enc = nn.Identity()
        
        self.decoder = MultiLayerDecoder_mask(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        """
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * 2),
            nn.Sigmoid()               
        )
        """        
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        #self.dist_predictor = nn.Sequential(
        #    nn.Linear(32, 1),
        #)        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )        
        self.goal_mask_0 = torch.zeros((1, self.context_size + 3), dtype=torch.bool)
        self.goal_mask_0[:, -2] = True # Mask out the goal 
        self.goal_mask_1 = torch.zeros((1, self.context_size + 3), dtype=torch.bool)
        self.goal_mask_1[:, -1] = True # Mask out the goal   
        self.goal_mask_2 = torch.zeros((1, self.context_size + 3), dtype=torch.bool)
        self.goal_mask_2[:, -1] = True # Mask out the goal         
        self.goal_mask_2[:, -2] = True # Mask out the goal                
        self.all_masks = torch.cat([self.goal_mask_0, self.goal_mask_1, self.goal_mask_2], dim=0)

        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor, map_images: torch.tensor, goal_mask: torch.tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            #print(obsgoal_img.size(), obs_img.size())
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)

        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size
        """
        device = obs_img.get_device()
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        map_encoding = self.goal_encoder.extract_features(map_images).unsqueeze(1)
        map_encoding = self.obs_encoder._avg_pooling(map_encoding)
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
            
        if self.goal_encoder._global_params.include_top:
            map_encoding = map_encoding.flatten(start_dim=1)
            map_encoding = self.goal_encoder._dropout(map_encoding)
                        
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        #print("obs_encoding", obs_encoding.size(), map_encoding.size())
        obs_encoding = self.compress_obs_enc(obs_encoding)
        #print(obs_encoding.size())
        map_encoding = self.compress_obs_enc_map(map_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        #obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        #obs_encoding = torch.transpose(obs_encoding, 0, 1)
                
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size(), map_encoding.size())
        tokens = torch.cat((obs_encoding, goal_encoding, map_encoding.unsqueeze(1)), dim=1)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None        
        """
        print("tokens", tokens.size())
        print("no_goal_mask", no_goal_mask.size())
        print("no_goal_mask", no_goal_mask[0])
        print("no_goal_mask", no_goal_mask[-1])        
        print("src_key_padding_mask", src_key_padding_mask.size())
        print("src_key_padding_mask", src_key_padding_mask[0])
        print("src_key_padding_mask", src_key_padding_mask[-1])        
        """
        final_repr = self.decoder(tokens, src_key_padding_mask)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        #dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return action_pred, no_goal_mask           
        
class IL_gps_map_mask3(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_gps_map_mask3, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=9) # context
            self.num_obs_features_map = self.goal_encoder._fc.in_features            
            
            if self.late_fusion:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features_img = self.goal_encoder_img._fc.in_features
            
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
            
        if self.num_obs_features_map != self.obs_encoding_size:
            self.compress_obs_enc_map = nn.Linear(self.num_obs_features_map, self.obs_encoding_size)
        else:
            self.compress_obs_enc_map = nn.Identity()            
                    
        if self.num_goal_features_img != self.goal_encoding_size:
            self.compress_goal_enc_img = nn.Linear(self.num_goal_features_img, self.goal_encoding_size)
        else:
            self.compress_goal_enc_img = nn.Identity()
        
        self.compress_goal_enc = nn.Identity()
        
        self.decoder = MultiLayerDecoder_mask2(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        """
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * 2),
            nn.Sigmoid()               
        )
        """        
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )           
               
        self.goal_mask_0 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)
        self.goal_mask_0[:, -3] = True # Mask out the goal 
        self.goal_mask_0[:, -1] = True # Mask out the goal         
        self.goal_mask_1 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)
        self.goal_mask_1[:, -2] = True # Mask out the goal   
        self.goal_mask_1[:, -1] = True # Mask out the goal          
        self.goal_mask_2 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)
        self.goal_mask_2[:, -1] = True # Mask out the goal                           
        self.goal_mask_3 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)        
        self.goal_mask_3[:, -3] = True # Mask out the goal 
        self.goal_mask_4 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)        
        self.goal_mask_4[:, -2] = True # Mask out the goal 
        self.goal_mask_5 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)        
        self.goal_mask_6 = torch.zeros((1, self.context_size + 4), dtype=torch.bool)   
        self.goal_mask_6[:, -3] = True # Mask out the goal   
        self.goal_mask_6[:, -2] = True # Mask out the goal                                         
        self.all_masks = torch.cat([self.goal_mask_0, self.goal_mask_2, self.goal_mask_3, self.goal_mask_5, self.goal_mask_1, self.goal_mask_4, self.goal_mask_6], dim=0)
        self.no_mask = torch.zeros((1, self.context_size + 4), dtype=torch.bool) 
        
        avep_mask_0 = (1.0 - self.goal_mask_0.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_0.float())))
        avep_mask_1 = (1.0 - self.goal_mask_1.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_1.float())))
        avep_mask_2 = (1.0 - self.goal_mask_2.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_2.float())))
        avep_mask_3 = (1.0 - self.goal_mask_3.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_3.float())))
        avep_mask_4 = (1.0 - self.goal_mask_4.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_4.float())))
        avep_mask_5 = (1.0 - self.goal_mask_5.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_5.float())))
        avep_mask_6 = (1.0 - self.goal_mask_6.float())*((self.context_size + 4)/(torch.sum(1.0 - self.goal_mask_6.float())))
        """
        print("avep_mask_0", avep_mask_0)
        print("avep_mask_1", avep_mask_1)
        print("avep_mask_2", avep_mask_2)
        print("avep_mask_3", avep_mask_3)
        print("avep_mask_4", avep_mask_4)
        print("avep_mask_5", avep_mask_5)
        print("avep_mask_6", avep_mask_6)        
        """
        self.avg_pool_mask = torch.cat([avep_mask_0, avep_mask_2, avep_mask_3, avep_mask_5, avep_mask_1, avep_mask_4, avep_mask_6], dim=0)
        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor, map_images: torch.tensor, goal_img: torch.tensor, goal_mask: torch.tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding_img = self.goal_encoder_img.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding_img = self.goal_encoder_img.extract_features(obsgoal_img)
        goal_encoding_img = self.goal_encoder_img._avg_pooling(goal_encoding_img)
        if self.goal_encoder._global_params.include_top:
            goal_encoding_img = goal_encoding_img.flatten(start_dim=1)
            goal_encoding_img = self.goal_encoder_img._dropout(goal_encoding_img)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding_img = self.compress_goal_enc_img(goal_encoding_img)

        if len(goal_encoding_img.shape) == 2:
            goal_encoding_img = goal_encoding_img.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding_img.shape[2] == self.goal_encoding_size
        
        device = obs_img.get_device()
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        map_encoding = self.goal_encoder.extract_features(map_images).unsqueeze(1)
        map_encoding = self.obs_encoder._avg_pooling(map_encoding)
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
            
        if self.goal_encoder._global_params.include_top:
            map_encoding = map_encoding.flatten(start_dim=1)
            map_encoding = self.goal_encoder._dropout(map_encoding)
                        
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        #print("obs_encoding", obs_encoding.size(), map_encoding.size())
        obs_encoding = self.compress_obs_enc(obs_encoding)
        #print(obs_encoding.size())
        map_encoding = self.compress_obs_enc_map(map_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        #obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        #obs_encoding = torch.transpose(obs_encoding, 0, 1)
                
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size(), map_encoding.unsqueeze(1).size(), goal_encoding_img.size())
        tokens = torch.cat((obs_encoding, goal_encoding, map_encoding.unsqueeze(1), goal_encoding_img), dim=1)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None  
            
        #print("src_key_padding_mask", src_key_padding_mask)      
        """
        print("tokens", tokens.size())
        print("no_goal_mask", no_goal_mask.size())
        print("no_goal_mask", no_goal_mask[0])
        print("no_goal_mask", no_goal_mask[-1])        
        print("src_key_padding_mask", src_key_padding_mask.size())
        print("src_key_padding_mask", src_key_padding_mask[0])
        print("src_key_padding_mask", src_key_padding_mask[-1])        
        """
        final_repr = self.decoder(tokens, src_key_padding_mask, self.avg_pool_mask.to(device), no_goal_mask)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return action_pred, dist_pred, no_goal_mask     
        
class IL_gps_map_mask3_lan(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_gps_map_mask3_lan, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=9) # context
            self.num_obs_features_map = self.goal_encoder._fc.in_features            
            
            if self.late_fusion:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features_img = self.goal_encoder_img._fc.in_features
            
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
            
        if self.num_obs_features_map != self.obs_encoding_size:
            self.compress_obs_enc_map = nn.Linear(self.num_obs_features_map, self.obs_encoding_size)
        else:
            self.compress_obs_enc_map = nn.Identity()            
                    
        if self.num_goal_features_img != self.goal_encoding_size:
            self.compress_goal_enc_img = nn.Linear(self.num_goal_features_img, self.goal_encoding_size)
        else:
            self.compress_goal_enc_img = nn.Identity()
        
        #self.compress_goal_enc = nn.Identity()        
        self.num_goal_features_lan = 4096
        if self.num_goal_features_lan != self.goal_encoding_size:
            self.compress_goal_enc_lan = nn.Linear(self.num_goal_features_lan, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc_lan = nn.Identity()
        
        self.decoder = MultiLayerDecoder_mask2(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2+1+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        
        self.film_model = build_film_model(8, 10, 128, 512)
        # LGX (Language-Grounded cross-attention). Opt-in: set model.use_lgx = True externally;
        # when off, forward skips self.lgx so behavior == base. (added to both lan variants; only
        # IL_gps_map_mask3_lan2 is instantiated.)
        self.use_lgx = False
        self.lgx = LGXModule(text_dim=512, vis_dim=self.num_obs_features,
                             hidden=256, out_dim=self.goal_encoding_size, heads=4)
               
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )           
               
        self.goal_mask_0 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_0[:, -4] = True # Mask out the goal 
        self.goal_mask_0[:, -2] = True # Mask out the goal
        self.goal_mask_0[:, -1] = True # Mask out the goal                   
        self.goal_mask_1 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_1[:, -3] = True # Mask out the goal   
        self.goal_mask_1[:, -2] = True # Mask out the goal  
        self.goal_mask_1[:, -1] = True # Mask out the goal                  
        self.goal_mask_2 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_2[:, -2] = True # Mask out the goal  
        self.goal_mask_2[:, -1] = True # Mask out the goal                                   
        self.goal_mask_3 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_3[:, -4] = True # Mask out the goal 
        self.goal_mask_3[:, -1] = True # Mask out the goal          
        self.goal_mask_4 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_4[:, -3] = True # Mask out the goal 
        self.goal_mask_4[:, -1] = True # Mask out the goal          
        self.goal_mask_5 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)     
        self.goal_mask_5[:, -1] = True # Mask out the goal       
        self.goal_mask_6 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_6[:, -4] = True # Mask out the goal   
        self.goal_mask_6[:, -3] = True # Mask out the goal  
        self.goal_mask_6[:, -1] = True # Mask out the goal          
        self.goal_mask_7 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_7[:, -4] = True # Mask out the goal   
        self.goal_mask_7[:, -3] = True # Mask out the goal 
        self.goal_mask_7[:, -2] = True # Mask out the goal         
        self.goal_mask_8 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)    
        self.goal_mask_8[:, -3] = True # Mask out the goal  
        self.goal_mask_8[:, -2] = True # Mask out the goal                                                                
        self.all_masks = torch.cat([self.goal_mask_0, self.goal_mask_2, self.goal_mask_3, self.goal_mask_5, self.goal_mask_1, self.goal_mask_4, self.goal_mask_6, self.goal_mask_7, self.goal_mask_8], dim=0)
        self.no_mask = torch.zeros((1, self.context_size + 5), dtype=torch.bool) 
        
        avep_mask_0 = (1.0 - self.goal_mask_0.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_0.float())))
        avep_mask_1 = (1.0 - self.goal_mask_1.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_1.float())))
        avep_mask_2 = (1.0 - self.goal_mask_2.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_2.float())))
        avep_mask_3 = (1.0 - self.goal_mask_3.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_3.float())))
        avep_mask_4 = (1.0 - self.goal_mask_4.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_4.float())))
        avep_mask_5 = (1.0 - self.goal_mask_5.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_5.float())))
        avep_mask_6 = (1.0 - self.goal_mask_6.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_6.float())))
        avep_mask_7 = (1.0 - self.goal_mask_7.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_7.float())))
        avep_mask_8 = (1.0 - self.goal_mask_8.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_8.float())))        
        """
        print("avep_mask_0", avep_mask_0)
        print("avep_mask_1", avep_mask_1)
        print("avep_mask_2", avep_mask_2)
        print("avep_mask_3", avep_mask_3)
        print("avep_mask_4", avep_mask_4)
        print("avep_mask_5", avep_mask_5)
        print("avep_mask_6", avep_mask_6)        
        """
        self.avg_pool_mask = torch.cat([avep_mask_0, avep_mask_2, avep_mask_3, avep_mask_5, avep_mask_1, avep_mask_4, avep_mask_6, avep_mask_7, avep_mask_8], dim=0)
        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor, map_images: torch.tensor, goal_img: torch.tensor, goal_mask: torch.tensor, feat_text: torch.tensor, current_img: torch.tensor, text_tokens: torch.tensor = None, text_valid: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get the goal encoding
        # text feature
        inst_encoding = feat_text
        obsgoal_encoding_lan = self.film_model(current_img, inst_encoding)        
        obsgoal_encoding_lan_cat = obsgoal_encoding_lan.flatten(start_dim=1)
        obsgoal_encoding_lan = self.compress_goal_enc_lan(obsgoal_encoding_lan_cat)
        # LGX residual: text tokens attend to the current-image visual grid, added to the FiLM
        # language encoding (scene-grounded language). Skipped unless use_lgx and text_tokens given.
        if getattr(self, "use_lgx", False) and text_tokens is not None:
            _vg = self.obs_encoder.extract_features(current_img).flatten(2).transpose(1, 2)
            obsgoal_encoding_lan = obsgoal_encoding_lan + self.lgx(text_tokens, _vg, text_valid)
        #print(obsgoal_encoding_lan.size(), obsgoal_encoding_lan_cat.size())

        if len(obsgoal_encoding_lan.shape) == 2:
            obsgoal_encoding_lan = obsgoal_encoding_lan.unsqueeze(1)
        assert obsgoal_encoding_lan.shape[2] == self.goal_encoding_size
        goal_encoding_lan = obsgoal_encoding_lan   
                
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding_img = self.goal_encoder_img.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding_img = self.goal_encoder_img.extract_features(obsgoal_img)
        goal_encoding_img = self.goal_encoder_img._avg_pooling(goal_encoding_img)
        if self.goal_encoder._global_params.include_top:
            goal_encoding_img = goal_encoding_img.flatten(start_dim=1)
            goal_encoding_img = self.goal_encoder_img._dropout(goal_encoding_img)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding_img = self.compress_goal_enc_img(goal_encoding_img)

        if len(goal_encoding_img.shape) == 2:
            goal_encoding_img = goal_encoding_img.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding_img.shape[2] == self.goal_encoding_size
        
        device = obs_img.get_device()
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        map_encoding = self.goal_encoder.extract_features(map_images).unsqueeze(1)
        map_encoding = self.obs_encoder._avg_pooling(map_encoding)
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
            
        if self.goal_encoder._global_params.include_top:
            map_encoding = map_encoding.flatten(start_dim=1)
            map_encoding = self.goal_encoder._dropout(map_encoding)
                        
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        #print("obs_encoding", obs_encoding.size(), map_encoding.size())
        obs_encoding = self.compress_obs_enc(obs_encoding)
        #print(obs_encoding.size())
        map_encoding = self.compress_obs_enc_map(map_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        #obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        #obs_encoding = torch.transpose(obs_encoding, 0, 1)
                
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size(), map_encoding.unsqueeze(1).size(), goal_encoding_img.size())
        tokens = torch.cat((obs_encoding, goal_encoding, map_encoding.unsqueeze(1), goal_encoding_img, goal_encoding_lan), dim=1)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None  
            
        #print("src_key_padding_mask", src_key_padding_mask)      
        """
        print("tokens", tokens.size())
        print("no_goal_mask", no_goal_mask.size())
        print("no_goal_mask", no_goal_mask[0])
        print("no_goal_mask", no_goal_mask[-1])        
        print("src_key_padding_mask", src_key_padding_mask.size())
        print("src_key_padding_mask", src_key_padding_mask[0])
        print("src_key_padding_mask", src_key_padding_mask[-1])        
        """
        final_repr = self.decoder(tokens, src_key_padding_mask, self.avg_pool_mask.to(device), no_goal_mask)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return action_pred, dist_pred, no_goal_mask        

class IL_gps_map_mask3_lan2(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(IL_gps_map_mask3_lan2, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=9) # context
            self.num_obs_features_map = self.goal_encoder._fc.in_features            
            
            if self.late_fusion:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder_img = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features_img = self.goal_encoder_img._fc.in_features
            
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
            
        if self.num_obs_features_map != self.obs_encoding_size:
            self.compress_obs_enc_map = nn.Linear(self.num_obs_features_map, self.obs_encoding_size)
        else:
            self.compress_obs_enc_map = nn.Identity()            
                    
        if self.num_goal_features_img != self.goal_encoding_size:
            self.compress_goal_enc_img = nn.Linear(self.num_goal_features_img, self.goal_encoding_size)
        else:
            self.compress_goal_enc_img = nn.Identity()
        
        #self.compress_goal_enc = nn.Identity()        
        self.num_goal_features_lan = 4096
        if self.num_goal_features_lan != self.goal_encoding_size:
            self.compress_goal_enc_lan = nn.Linear(self.num_goal_features_lan, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc_lan = nn.Identity()
        
        self.decoder = MultiLayerDecoder_mask3(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2+1+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
            #nn.Sigmoid()               
        )
        
        self.film_model = build_film_model(8, 10, 128, 512)
        # LGX (Language-Grounded cross-attention). Opt-in: set model.use_lgx = True externally;
        # when off, forward skips self.lgx so behavior == base. (added to both lan variants; only
        # IL_gps_map_mask3_lan2 is instantiated.)
        self.use_lgx = False
        self.lgx = LGXModule(text_dim=512, vis_dim=self.num_obs_features,
                             hidden=256, out_dim=self.goal_encoding_size, heads=4)
               
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )        
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),         
        )           
               
        self.goal_mask_0 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_0[:, -4] = True # Mask out the goal 
        self.goal_mask_0[:, -2] = True # Mask out the goal
        self.goal_mask_0[:, -1] = True # Mask out the goal                   
        self.goal_mask_1 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_1[:, -3] = True # Mask out the goal   
        self.goal_mask_1[:, -2] = True # Mask out the goal  
        self.goal_mask_1[:, -1] = True # Mask out the goal                  
        self.goal_mask_2 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)
        self.goal_mask_2[:, -2] = True # Mask out the goal  
        self.goal_mask_2[:, -1] = True # Mask out the goal                                   
        self.goal_mask_3 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_3[:, -4] = True # Mask out the goal 
        self.goal_mask_3[:, -1] = True # Mask out the goal          
        self.goal_mask_4 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)        
        self.goal_mask_4[:, -3] = True # Mask out the goal 
        self.goal_mask_4[:, -1] = True # Mask out the goal          
        self.goal_mask_5 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)     
        self.goal_mask_5[:, -1] = True # Mask out the goal       
        self.goal_mask_6 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_6[:, -4] = True # Mask out the goal   
        self.goal_mask_6[:, -3] = True # Mask out the goal  
        self.goal_mask_6[:, -1] = True # Mask out the goal          
        self.goal_mask_7 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)   
        self.goal_mask_7[:, -4] = True # Mask out the goal   
        self.goal_mask_7[:, -3] = True # Mask out the goal 
        self.goal_mask_7[:, -2] = True # Mask out the goal         
        self.goal_mask_8 = torch.zeros((1, self.context_size + 5), dtype=torch.bool)    
        self.goal_mask_8[:, -3] = True # Mask out the goal  
        self.goal_mask_8[:, -2] = True # Mask out the goal                                                                
        self.all_masks = torch.cat([self.goal_mask_0, self.goal_mask_2, self.goal_mask_3, self.goal_mask_5, self.goal_mask_1, self.goal_mask_4, self.goal_mask_6, self.goal_mask_7, self.goal_mask_8], dim=0)
        self.no_mask = torch.zeros((1, self.context_size + 5), dtype=torch.bool) 
        
        avep_mask_0 = (1.0 - self.goal_mask_0.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_0.float())))
        avep_mask_1 = (1.0 - self.goal_mask_1.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_1.float())))
        avep_mask_2 = (1.0 - self.goal_mask_2.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_2.float())))
        avep_mask_3 = (1.0 - self.goal_mask_3.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_3.float())))
        avep_mask_4 = (1.0 - self.goal_mask_4.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_4.float())))
        avep_mask_5 = (1.0 - self.goal_mask_5.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_5.float())))
        avep_mask_6 = (1.0 - self.goal_mask_6.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_6.float())))
        avep_mask_7 = (1.0 - self.goal_mask_7.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_7.float())))
        avep_mask_8 = (1.0 - self.goal_mask_8.float())*((self.context_size + 5)/(torch.sum(1.0 - self.goal_mask_8.float())))        
        """
        print("avep_mask_0", avep_mask_0)
        print("avep_mask_1", avep_mask_1)
        print("avep_mask_2", avep_mask_2)
        print("avep_mask_3", avep_mask_3)
        print("avep_mask_4", avep_mask_4)
        print("avep_mask_5", avep_mask_5)
        print("avep_mask_6", avep_mask_6)        
        """
        self.avg_pool_mask = torch.cat([avep_mask_0, avep_mask_2, avep_mask_3, avep_mask_5, avep_mask_1, avep_mask_4, avep_mask_6, avep_mask_7, avep_mask_8], dim=0)
        
    def forward(
        self, obs_img: torch.tensor, goal_pose: torch.tensor, map_images: torch.tensor, goal_img: torch.tensor, goal_mask: torch.tensor, feat_text: torch.tensor, current_img: torch.tensor, text_tokens: torch.tensor = None, text_valid: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get the goal encoding
        # text feature
        inst_encoding = feat_text
        obsgoal_encoding_lan = self.film_model(current_img, inst_encoding)        
        obsgoal_encoding_lan_cat = obsgoal_encoding_lan.flatten(start_dim=1)
        obsgoal_encoding_lan = self.compress_goal_enc_lan(obsgoal_encoding_lan_cat)
        # LGX residual: text tokens attend to the current-image visual grid, added to the FiLM
        # language encoding (scene-grounded language). Skipped unless use_lgx and text_tokens given.
        if getattr(self, "use_lgx", False) and text_tokens is not None:
            _vg = self.obs_encoder.extract_features(current_img).flatten(2).transpose(1, 2)
            obsgoal_encoding_lan = obsgoal_encoding_lan + self.lgx(text_tokens, _vg, text_valid)
        #print(obsgoal_encoding_lan.size(), obsgoal_encoding_lan_cat.size())

        if len(obsgoal_encoding_lan.shape) == 2:
            obsgoal_encoding_lan = obsgoal_encoding_lan.unsqueeze(1)
        assert obsgoal_encoding_lan.shape[2] == self.goal_encoding_size
        goal_encoding_lan = obsgoal_encoding_lan   
                
        # get the fused observation and goal encoding
        if self.late_fusion:
            goal_encoding_img = self.goal_encoder_img.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding_img = self.goal_encoder_img.extract_features(obsgoal_img)
        goal_encoding_img = self.goal_encoder_img._avg_pooling(goal_encoding_img)
        if self.goal_encoder._global_params.include_top:
            goal_encoding_img = goal_encoding_img.flatten(start_dim=1)
            goal_encoding_img = self.goal_encoder_img._dropout(goal_encoding_img)
        # currently, the size of goal_encoding is [batch_size, num_goal_features]
        goal_encoding_img = self.compress_goal_enc_img(goal_encoding_img)

        if len(goal_encoding_img.shape) == 2:
            goal_encoding_img = goal_encoding_img.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding_img.shape[2] == self.goal_encoding_size
        
        device = obs_img.get_device()
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)
        map_encoding = self.goal_encoder.extract_features(map_images).unsqueeze(1)
        map_encoding = self.obs_encoder._avg_pooling(map_encoding)
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
            
        if self.goal_encoder._global_params.include_top:
            map_encoding = map_encoding.flatten(start_dim=1)
            map_encoding = self.goal_encoder._dropout(map_encoding)
                        
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        #print("obs_encoding", obs_encoding.size(), map_encoding.size())
        obs_encoding = self.compress_obs_enc(obs_encoding)
        #print(obs_encoding.size())
        map_encoding = self.compress_obs_enc_map(map_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        #obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        #obs_encoding = torch.transpose(obs_encoding, 0, 1)
                
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        #print("encoding", obs_encoding.size(), goal_encoding.size(), map_encoding.unsqueeze(1).size(), goal_encoding_img.size())
        tokens = torch.cat((obs_encoding, goal_encoding, map_encoding.unsqueeze(1), goal_encoding_img, goal_encoding_lan), dim=1)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None  
            
        #print("src_key_padding_mask", src_key_padding_mask)      
        """
        print("tokens", tokens.size())
        print("no_goal_mask", no_goal_mask.size())
        print("no_goal_mask", no_goal_mask[0])
        print("no_goal_mask", no_goal_mask[-1])        
        print("src_key_padding_mask", src_key_padding_mask.size())
        print("src_key_padding_mask", src_key_padding_mask[0])
        print("src_key_padding_mask", src_key_padding_mask[-1])        
        """
        final_repr = self.decoder(tokens, src_key_padding_mask, self.avg_pool_mask.to(device), no_goal_mask)
        # currently, the size is [batch_size, 32]

        action_pred = self.action_predictor(final_repr)
        dist_pred = self.dist_predictor(final_repr)
        
        # augment outputs to match labels size-wise        
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        #if self.learn_angle:
        if True:        
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        
        #linear_vel = self.max_linvel*action_pred[:, 0:self.len_trajectory_pred]  #max +0.5 m/s min 0.0 m/s
        #angular_vel = self.max_angvel*2.0*(action_pred[:, self.len_trajectory_pred:2*self.len_trajectory_pred] - 0.5)  #max +1.0 rad/s min -1.0 rad/s        
        #return linear_vel, angular_vel, dist_pred            
        return action_pred, dist_pred, no_goal_mask  

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def create_conv_layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super(InitialFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(
            create_conv_layer(3, 128, 5, 2, 2),
            create_conv_layer(128, 128, 3, 2, 1),
            create_conv_layer(128, 128, 3, 2, 1),
        )
        
    def forward(self, x):
        return self.layers(x)

class IntermediateFeatureExtractor(nn.Module):
    def __init__(self):
        super(IntermediateFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(       
            create_conv_layer(128, 256, 3, 2, 1),
            create_conv_layer(256, 512, 3, 2, 1),
            create_conv_layer(512, 1024, 3, 2, 1),
            create_conv_layer(1024, 1024, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.layers(x)

        
def clip_token_features(clip_model, tokenized_text):
    """Per-token CLIP text features (B, L, 512) BEFORE [EOS] pooling, for LGX cross-attention.
    Returns (features, valid_mask) where valid_mask marks non-pad token positions."""
    x = clip_model.token_embedding(tokenized_text)
    x = x + clip_model.positional_embedding
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x)
    return x.float(), (tokenized_text != 0)


class LGXModule(nn.Module):
    """Language-Grounded cross-attention: text tokens (query) attend to the current-image visual
    grid (key/value), so words like 'left'/'wall' bind to scene regions. Produces a scene-grounded
    language vector, added as a RESIDUAL to the FiLM language token (conservative: FiLM preserved).
    """
    def __init__(self, text_dim=512, vis_dim=1280, hidden=256, out_dim=1024, heads=4):
        super().__init__()
        self.tp = nn.Linear(text_dim, hidden)
        self.vp = nn.Linear(vis_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, text_tokens, vis_grid, text_valid=None):
        q = self.tp(text_tokens.float())          # (B, L, hidden)
        kv = self.vp(vis_grid)                    # (B, S, hidden)
        a, _ = self.attn(q, kv, kv)               # each text token attends to the visual grid
        a = self.norm(q + a)
        if text_valid is not None:
            m = text_valid.float().unsqueeze(-1)
            pooled = (a * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = a.mean(1)
        return self.out(pooled)                   # (B, out_dim)


class FiLMTransform(nn.Module):
    def __init__(self):
        super(FiLMTransform, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x
        
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film_transform = FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film_transform(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x

class FinalClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(FinalClassifier, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        feature_map = x
        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc_layers(x)
        
        return x, feature_map
           
class FiLMNetwork(nn.Module):
    def __init__(self, num_res_blocks, num_classes, num_channels, question_dim):
        super(FiLMNetwork, self).__init__()
        question_feature_dim = question_dim

        self.film_param_generator = nn.Linear(question_feature_dim, 2 * num_res_blocks * num_channels)
        self.initial_feature_extractor = InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList()
        self.intermediate_feature_extractor = IntermediateFeatureExtractor()
        
        for _ in range(num_res_blocks):
            self.residual_blocks.append(ResidualBlock(num_channels + 2, num_channels))
            
        self.final_classifier = FinalClassifier(num_channels, num_classes)
    
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        
    def forward(self, x, question):
        batch_size = x.size(0)
        device = x.device
        
        x = self.initial_feature_extractor(x)
        film_params = self.film_param_generator(question).view(
            batch_size, self.num_res_blocks, 2, self.num_channels)
        
        d = x.size(2)
        coords = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).to(device)
        coord_x = coords.expand(batch_size, 1, d, d)
        coord_y = coords.view(d, 1).expand(batch_size, 1, d, d)
        
        for i, res_block in enumerate(self.residual_blocks):
            beta = film_params[:, i, 0, :]
            gamma = film_params[:, i, 1, :]
            
            x = torch.cat([x, coord_x, coord_y], 1)
            x = res_block(x, beta, gamma)
        
        features = self.intermediate_feature_extractor(x)
        
        return features

def build_film_model(num_res_blocks, num_classes, num_channels, question_dim):
    return FiLMNetwork(num_res_blocks, num_classes, num_channels, question_dim)           
