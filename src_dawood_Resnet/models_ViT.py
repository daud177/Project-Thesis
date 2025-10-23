import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from transformers import ViTModel

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='WinKawaks/vit-small-patch16-224', device=None):
        super(ViTFeatureExtractor, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = ViTModel.from_pretrained(model_name).to(self.device)

    def forward(self, pixel_values):
        pixel_values = pixel_values[:, 0, :, :, :]  # Adjusting the shape
        pixel_values = pixel_values.to(self.device)  # Ensure pixel_values are on the correct device
        outputs = self.vit(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0]
        return features

class PolicyWithViT(nn.Module):
    def __init__(self, config, feature_extractor=None, device=None):
        super(PolicyWithViT, self).__init__()
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device) if feature_extractor is not None else ViTFeatureExtractor(device=self.device).to(self.device)
        
        output_dim = self.feature_extractor.vit.config.hidden_size
        proprio_dim = config["proprio_dim"]
        lstm_dim = output_dim + proprio_dim
        self.lstm = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, batch_first=True).to(self.device)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"]).to(self.device)
        
        self.std = nn.Parameter(torch.ones(config["action_dim"]) * 0.1).to(self.device)  # Std deviation for action distribution
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def forward(self, pixel_values, proprio_obs):
        pixel_values = pixel_values.to(self.device)
        proprio_obs = proprio_obs.to(self.device)
        vis_features = self.feature_extractor(pixel_values)
        proprio_obs = proprio_obs.mean(dim=1).to(self.device)
        proprio_obs = proprio_obs.unsqueeze(1)
        vis_features = vis_features.to(self.device)
        combined_input = torch.cat([vis_features.unsqueeze(1), proprio_obs], dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        action_probs = torch.tanh(self.linear_out(lstm_out))
        return action_probs.squeeze(0)

    def update_params(self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj, rank):
        camera_obs_traj = camera_obs_traj.to(self.device)
        proprio_obs_traj = proprio_obs_traj.to(self.device)
        action_traj = action_traj.to(self.device)
        feedback_traj = feedback_traj.to(self.device)

        mu = self.forward(camera_obs_traj, proprio_obs_traj)
        distribution = Normal(mu, self.std)

        log_prob = distribution.log_prob(action_traj)
        feedback_traj_expanded = feedback_traj.expand(-1, -1, log_prob.size(2))
        loss = -log_prob * feedback_traj_expanded

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


"""
class Policy(nn.Module):
    def __init__(self, config, rank, world_size):
        super(Policy, self).__init__()
        # 设备设置
        self.device = torch.device(f'cuda:{rank % world_size}') if torch.cuda.is_available() else torch.device('cpu')

        # 视觉特征提取器
        self.feature_extractor = ViTFeatureExtractor().to(self.device)

        # 根据ViT输出和proprio_dim计算LSTM输入维度
        output_dim = self.feature_extractor.vit.config.hidden_size
        lstm_dim = output_dim + config["proprio_dim"]

        self.lstm = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, batch_first=True)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32).to(self.device)
        self.dropout = nn.Dropout(p=0.4)
        return

    def forward_step(self, pixel_values, proprio_obs, lstm_state=None):
        # 使用ViT特征提取器处理视觉输入
        vis_features = self.feature_extractor(pixel_values)
        # 将视觉特征和本体感觉输入合并
        combined_input = torch.cat([vis_features, proprio_obs], dim=-1).unsqueeze(0)
        # 如果没有提供初始LSTM状态，自动创建为None
        if lstm_state is None:
             lstm_state = None  # LSTM将自动初始化为零状态
         # 通过LSTM
        lstm_out, lstm_state = self.lstm(combined_input, lstm_state)
        # 应用一个线性层和激活函数来产生动作输出
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state

    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj):
        losses = []
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(action_traj[idx])
            loss = -log_prob * feedback_traj[idx]
            losses.append(loss)
        total_loss = torch.cat(losses).mean()
        return total_loss

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj,rank
    ):
        camera_obs = camera_obs_traj.to(self.device)
        proprio_obs = proprio_obs_traj.to(self.device)
        action = action_traj.to(self.device)
        feedback = feedback_traj.to(self.device)
        self.optimizer.zero_grad()
        loss = self.forward(camera_obs, proprio_obs, action, feedback)
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Device: {rank}, Gradient of {name}: {param.grad}")
        self.optimizer.step()
        training_metrics = {"loss": loss}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state
"""
    
vit_feature_extractor = ViTFeatureExtractor()

config = {
    "proprio_dim": 8,
    "action_dim": 7,
    "learning_rate": 3e-4,
    "weight_decay": 3e-6,
}

policy_model = PolicyWithViT(config)

def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action