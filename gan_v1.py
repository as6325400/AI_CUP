import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import math

class TimeSeriesGenerator(nn.Module):
    """
    GAN Generator with Transformer and LSTM architecture
    """
    def __init__(self, noise_dim=1024, condition_dim=512, sequence_length=27, feature_dim=34):
        super(TimeSeriesGenerator, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.d_model = 2048  # transformer dimension
        
        # conditional embedding
        self.condition_embedding = nn.Embedding(100, condition_dim)
        
        # input projection
        self.input_projection = nn.Sequential(
            nn.Linear(noise_dim + condition_dim * 4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.d_model)
        )
        
        # deep Transformer stack with dimensions
        encoder_layers = []
        for i in range(16):  # 16 transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=32,  # 32 attention heads
                dim_feedforward=8192,  # feedforward
                dropout=0.1,
                batch_first=True
            )
            encoder_layers.append(encoder_layer)
        self.transformer_stack = nn.ModuleList(encoder_layers)
        
        # Multiple parallel LSTM branches with hidden dimensions
        self.lstm_branch1 = nn.LSTM(self.d_model, 2048, num_layers=6, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_branch2 = nn.LSTM(4096, 1024, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_branch3 = nn.LSTM(2048, 512, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Fusion layers
        self.fusion_layer = nn.Linear(4096 + 2048 + 1024, 2048)
        
        # Multiple attention mechanisms
        self.self_attention1 = nn.MultiheadAttention(embed_dim=2048, num_heads=32, batch_first=True)
        self.self_attention2 = nn.MultiheadAttention(embed_dim=2048, num_heads=16, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        
        # Deep output network
        self.output_network = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(2048) for _ in range(5)])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self, noise, conditions):
        batch_size = noise.size(0)
        
        # conditional embedding
        cond_embeds = []
        for i in range(4):
            cond_embeds.append(self.condition_embedding(conditions[:, i]))
        cond_embed = torch.cat(cond_embeds, dim=1)
        
        # Input processing
        x = torch.cat([noise, cond_embed], dim=1)
        x = self.input_projection(x)
        
        # Expand for sequence generation
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x = self.layer_norms[0](x)
        
        # deep Transformer processing
        for transformer_layer in self.transformer_stack:
            x = transformer_layer(x)
        
        x = self.layer_norms[1](x)
        
        # Parallel LSTM branches
        lstm_out1, _ = self.lstm_branch1(x)
        lstm_out1 = self.dropout(lstm_out1)
        
        lstm_out2, _ = self.lstm_branch2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)
        
        lstm_out3, _ = self.lstm_branch3(lstm_out2)
        
        # Fusion of LSTM outputs
        fused = torch.cat([lstm_out1, lstm_out2, lstm_out3], dim=-1)
        x = self.relu(self.fusion_layer(fused))
        x = self.layer_norms[2](x)
        
        # Multiple attention mechanisms
        x, _ = self.self_attention1(x, x, x)
        x = self.layer_norms[3](x)
        
        x, _ = self.self_attention2(x, x, x)
        x = self.layer_norms[4](x)
        
        x, _ = self.cross_attention(x, x, x)
        
        # Final output
        x = self.output_network(x)
        return self.tanh(x)

class TimeSeriesDiscriminator(nn.Module):
    """
    GAN Discriminator with architecture
    """
    def __init__(self, sequence_length=27, feature_dim=34, condition_dim=512):
        super(TimeSeriesDiscriminator, self).__init__()
        
        # conditional embedding
        self.condition_embedding = nn.Embedding(100, condition_dim)
        
        # Deep convolutional stack
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(feature_dim, 256, kernel_size=5, padding=2),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.Conv1d(1024, 2048, kernel_size=3, padding=1),
            nn.Conv1d(2048, 4096, kernel_size=3, padding=1)
        ])
        
        # deep Transformer stack
        encoder_layers = []
        for i in range(12):  # 12 transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=4096, 
                nhead=64,  # 64 attention heads
                dim_feedforward=16384,  # feedforward
                dropout=0.1,
                batch_first=True
            )
            encoder_layers.append(encoder_layer)
        self.transformer_stack = nn.ModuleList(encoder_layers)
        
        # Multiple LSTM branches with enormous hidden dimensions
        self.lstm_branch1 = nn.LSTM(4096, 4096, num_layers=5, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_branch2 = nn.LSTM(8192, 2048, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_branch3 = nn.LSTM(4096, 1024, num_layers=3, batch_first=True, dropout=0.2)
        
        # Multiple attention mechanisms
        self.attention_modules = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=4096, num_heads=64, batch_first=True),
            nn.MultiheadAttention(embed_dim=4096, num_heads=32, batch_first=True),
            nn.MultiheadAttention(embed_dim=4096, num_heads=16, batch_first=True)
        ])
        
        # fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(8192 + 4096 + 1024, 8192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )
        
        # Deep classification network
        self.classifier = nn.Sequential(
            nn.Linear(2048 + condition_dim * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(4096) for _ in range(6)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, conditions):
        batch_size = x.size(0)
        
        # Deep convolutional processing
        x = x.transpose(1, 2)  # (batch, features, sequence)
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        x = self.layer_norms[0](x)
        
        # deep Transformer processing
        for transformer_layer in self.transformer_stack:
            x = transformer_layer(x)
        
        x = self.layer_norms[1](x)
        
        # Parallel LSTM processing
        lstm_out1, _ = self.lstm_branch1(x)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out1 = self.layer_norms[2](lstm_out1)
        
        lstm_out2, _ = self.lstm_branch2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)
        lstm_out2 = self.layer_norms[3](lstm_out2)
        
        lstm_out3, _ = self.lstm_branch3(lstm_out2)
        
        # Multiple attention mechanisms
        attn_out = x
        for attention_module in self.attention_modules:
            attn_out, _ = attention_module(attn_out, attn_out, attn_out)
            attn_out = self.layer_norms[4](attn_out)
        
        # Fusion
        fused = torch.cat([lstm_out1, lstm_out2, lstm_out3], dim=-1)
        pooled = torch.mean(fused, dim=1)
        x = self.fusion_network(pooled)
        
        # Conditional information
        cond_embeds = []
        for i in range(4):
            cond_embeds.append(self.condition_embedding(conditions[:, i]))
        cond_embed = torch.cat(cond_embeds, dim=1)
        
        # Final classification
        x = torch.cat([x, cond_embed], dim=1)
        x = self.classifier(x)
        
        return self.sigmoid(x)

class ConditionalGAN:
    """
    Conditional GAN for advanced time series generation
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = TimeSeriesGenerator().to(device)
        self.discriminator = TimeSeriesDiscriminator().to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        
        # Loss functions
        self.criterion_bce = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
        # Enable mixed precision for stability
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_step(self, real_data, real_conditions, noise_dim=1024):
        batch_size = real_data.size(0)
        
        # Generate multiple noise vectors for ensemble training
        noise1 = torch.randn(batch_size, noise_dim).to(self.device)
        noise2 = torch.randn(batch_size, noise_dim).to(self.device)
        noise3 = torch.randn(batch_size, noise_dim).to(self.device)
        
        # Labels with label smoothing
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
        
        # =================== Train Discriminator ===================
        self.d_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Real data
            d_real = self.discriminator(real_data, real_conditions)
            d_real_loss = self.criterion_bce(d_real, real_labels)
            
            # Multiple fake data generations
            fake_data1 = self.generator(noise1, real_conditions)
            fake_data2 = self.generator(noise2, real_conditions)
            fake_data3 = self.generator(noise3, real_conditions)
            
            d_fake1 = self.discriminator(fake_data1.detach(), real_conditions)
            d_fake2 = self.discriminator(fake_data2.detach(), real_conditions)
            d_fake3 = self.discriminator(fake_data3.detach(), real_conditions)
            
            d_fake_loss = (self.criterion_bce(d_fake1, fake_labels) + 
                          self.criterion_bce(d_fake2, fake_labels) + 
                          self.criterion_bce(d_fake3, fake_labels)) / 3
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_data, fake_data1, real_conditions)
            
            d_loss = d_real_loss + d_fake_loss + 10 * gradient_penalty
        
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        
        # =================== Train Generator ===================
        self.g_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Generate fake data
            fake_data1 = self.generator(noise1, real_conditions)
            fake_data2 = self.generator(noise2, real_conditions)
            fake_data3 = self.generator(noise3, real_conditions)
            
            # Adversarial loss
            d_fake1 = self.discriminator(fake_data1, real_conditions)
            d_fake2 = self.discriminator(fake_data2, real_conditions)
            d_fake3 = self.discriminator(fake_data3, real_conditions)
            
            g_loss_adv = (self.criterion_bce(d_fake1, real_labels) + 
                         self.criterion_bce(d_fake2, real_labels) + 
                         self.criterion_bce(d_fake3, real_labels)) / 3
            
            # Feature matching loss
            g_loss_fm = self.feature_matching_loss(real_data, fake_data1, real_conditions)
            
            # Reconstruction losses
            g_loss_mse = self.criterion_mse(fake_data1, real_data) * 0.1
            g_loss_l1 = self.criterion_l1(fake_data2, real_data) * 0.05
            
            # Diversity loss to prevent mode collapse
            diversity_loss = -self.criterion_mse(fake_data1, fake_data2) * 0.01
            
            g_loss = g_loss_adv + g_loss_fm + g_loss_mse + g_loss_l1 + diversity_loss
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        return g_loss.item(), d_loss.item()
    
    def compute_gradient_penalty(self, real_data, fake_data, conditions):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated, conditions)
        gradients = torch.autograd.grad(
            outputs=d_interpolated, inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True, retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def feature_matching_loss(self, real_data, fake_data, conditions):
        """Feature matching loss using discriminator intermediate features"""
        # Simplified version - compute L2 distance between real and fake features
        real_features = []
        fake_features = []
        
        def hook_fn_real(module, input, output):
            real_features.append(output)
        
        def hook_fn_fake(module, input, output):
            fake_features.append(output)
        
        # Register hooks on some discriminator layers
        hooks_real = []
        hooks_fake = []
        
        target_layers = [self.discriminator.conv_layers[2], 
                        self.discriminator.transformer_stack[5],
                        self.discriminator.lstm_branch1]
        
        for layer in target_layers:
            hooks_real.append(layer.register_forward_hook(hook_fn_real))
        
        # Forward pass for real data
        _ = self.discriminator(real_data, conditions)
        
        # Remove real hooks and add fake hooks
        for hook in hooks_real:
            hook.remove()
        
        for layer in target_layers:
            hooks_fake.append(layer.register_forward_hook(hook_fn_fake))
        
        # Forward pass for fake data
        _ = self.discriminator(fake_data, conditions)
        
        # Remove fake hooks
        for hook in hooks_fake:
            hook.remove()
        
        # Compute feature matching loss
        fm_loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            fm_loss += self.criterion_mse(fake_feat, real_feat)
        
        return fm_loss / len(real_features)

    def train(self, dataloader, epochs=200):
        """Train the GAN model"""
        print("Starting GAN training...")
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for batch_idx, (real_data, conditions) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                conditions = conditions.to(self.device)
                
                # Multiple training steps for better convergence
                for _ in range(2):
                    g_loss, d_loss = self.train_step(real_data, conditions)
                    g_loss_epoch += g_loss
                    d_loss_epoch += d_loss
            
            # Record losses
            self.g_losses.append(g_loss_epoch / (len(dataloader) * 2))
            self.d_losses.append(d_loss_epoch / (len(dataloader) * 2))
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], G_Loss: {g_loss_epoch/(len(dataloader)*2):.4f}, D_Loss: {d_loss_epoch/(len(dataloader)*2):.4f}')
    
    def generate_synthetic_data(self, conditions, num_samples=100):
        """Generate synthetic data"""
        self.generator.eval()
        synthetic_data = []
        
        with torch.no_grad():
            for i in range(0, num_samples, 16):  # Smaller batches due to memory
                batch_size = min(16, num_samples - i)
                noise = torch.randn(batch_size, 1024).to(self.device)
                batch_conditions = conditions[:batch_size].to(self.device)
                
                fake_data = self.generator(noise, batch_conditions)
                synthetic_data.append(fake_data.cpu().numpy())
        
        return np.concatenate(synthetic_data, axis=0)

def prepare_gan_data(x_train, y_train):
    """Prepare data for GAN training"""
    sequence_length = 27
    feature_dim = x_train.shape[1]
    
    num_sequences = len(x_train) // sequence_length
    x_sequences = x_train[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, feature_dim)
    
    le_gender = LabelEncoder()
    le_hand = LabelEncoder()
    le_years = LabelEncoder()
    le_level = LabelEncoder()
    
    conditions = np.zeros((num_sequences, 4), dtype=int)
    y_seq = y_train.iloc[:num_sequences * sequence_length:sequence_length]
    
    conditions[:, 0] = le_gender.fit_transform(y_seq['gender'])
    conditions[:, 1] = le_hand.fit_transform(y_seq['hold racket handed'])
    conditions[:, 2] = le_years.fit_transform(y_seq['play years'])
    conditions[:, 3] = le_level.fit_transform(y_seq['level'])
    
    return x_sequences, conditions, (le_gender, le_hand, le_years, le_level)

def augment_data_with_gan(x_train, y_train, augmentation_ratio=0.3):
    """Main function to augment training data using advanced GAN"""
    print("Preparing data for advanced GAN training...")
    x_sequences, conditions, label_encoders = prepare_gan_data(x_train, y_train)
    
    x_tensor = torch.FloatTensor(x_sequences)
    conditions_tensor = torch.LongTensor(conditions)
    
    # Use batch size for stable training
    dataset = TensorDataset(x_tensor, conditions_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
    
    try:
        # Initialize advanced GAN
        gan = ConditionalGAN()
        
        # Train the model
        gan.train(dataloader, epochs=100)
        
        # Generate synthetic data
        num_synthetic = int(len(x_sequences) * augmentation_ratio)
        print(f"Generating {num_synthetic} synthetic sequences...")
        
        synthetic_conditions = conditions_tensor[:num_synthetic]
        synthetic_sequences = gan.generate_synthetic_data(synthetic_conditions, num_synthetic)
        
        # Reshape and create labels
        synthetic_data = synthetic_sequences.reshape(-1, x_train.shape[1])
        
        synthetic_labels = []
        for i in range(num_synthetic):
            cond = conditions[i % len(conditions)]
            label_dict = {
                'gender': label_encoders[0].inverse_transform([cond[0]])[0],
                'hold racket handed': label_encoders[1].inverse_transform([cond[1]])[0],
                'play years': label_encoders[2].inverse_transform([cond[2]])[0],
                'level': label_encoders[3].inverse_transform([cond[3]])[0]
            }
            for _ in range(27):
                synthetic_labels.append(label_dict)
        
        synthetic_labels_df = pd.DataFrame(synthetic_labels)
        
        x_augmented = np.vstack([x_train, synthetic_data])
        y_augmented = pd.concat([y_train, synthetic_labels_df], ignore_index=True)
        
        print(f"Data augmentation completed. Original: {len(x_train)}, Augmented: {len(x_augmented)}")
        
        return x_augmented, y_augmented, gan
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory: {e}")
        print("The advanced Transformer+LSTM architecture requires substantial GPU memory.")
        print("Consider using a high-end GPU instance for training.")
        exit(1)
        
    except Exception as e:
        print(f"Advanced GAN training failed: {e}")
        print("Complex architecture encountered training difficulties.")
        return x_train, y_train, None

def main_with_gan():
    """Main function with advanced GAN architecture"""
    print("=== Advanced GAN Data Augmentation Experiment ===")
    
    # Load data
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    
    scaler = MinMaxScaler()
    le = LabelEncoder()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    print(f"Original training data shape: {x_train_scaled.shape}")
    
    # Attempt advanced GAN augmentation
    try:
        x_train_augmented, y_train_augmented, gan_model = augment_data_with_gan(
            x_train_scaled, y_train, augmentation_ratio=0.3
        )
        
        if x_train_augmented is not None:
            x_train_final = scaler.fit_transform(x_train_augmented)
            y_train_final = y_train_augmented
            print("Advanced GAN augmentation successful!")
        else:
            x_train_final = x_train_scaled
            y_train_final = y_train
            
    except Exception as e:
        print(f"Advanced GAN training failed: {e}")
        x_train_final = x_train_scaled
        y_train_final = y_train
    
    # Evaluation (same as before)
    group_size = 27
    
    def model_binary(X_train, y_train, X_test, y_test, target_name):
        clf = RandomForestClassifier(random_state=42, n_estimators=300)
        clf.fit(X_train, y_train)
        
        predicted = clf.predict_proba(X_test)
        predicted = [predicted[i][0] for i in range(len(predicted))]
        
        num_groups = len(predicted) // group_size
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        y_pred = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f"{target_name} Binary AUC: {auc_score:.4f}")
        return auc_score

    def model_multiary(X_train, y_train, X_test, y_test, target_name):
        clf = RandomForestClassifier(random_state=42, n_estimators=300)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print(f'{target_name} Multiary AUC: {auc_score:.4f}')
        return auc_score
    
    # Evaluate all targets
    results = {}
    
    y_train_le_gender = le.fit_transform(y_train_final['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    results['gender'] = model_binary(x_train_final, y_train_le_gender, x_test_scaled, y_test_le_gender, "Gender")
    
    y_train_le_hold = le.fit_transform(y_train_final['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    results['hold'] = model_binary(x_train_final, y_train_le_hold, x_test_scaled, y_test_le_hold, "Hold Racket")
    
    y_train_le_years = le.fit_transform(y_train_final['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    results['years'] = model_multiary(x_train_final, y_train_le_years, x_test_scaled, y_test_le_years, "Play Years")
    
    y_train_le_level = le.fit_transform(y_train_final['level'])
    y_test_le_level = le.transform(y_test['level'])
    results['level'] = model_multiary(x_train_final, y_train_le_level, x_test_scaled, y_test_le_level, "Level")
    
    print("\n=== Final Results ===")
    total_score = sum(results.values())
    print(f"Total AUC Score: {total_score:.4f}")
    print("Individual scores:", results)
    
    # Generate test predictions
    print("\n=== Generating Test Predictions ===")
    test_pred(x_train_final, y_train_final, scaler, le)

if __name__ == '__main__':
    main_with_gan()
