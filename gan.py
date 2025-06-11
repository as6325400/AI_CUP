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
    GAN Generator for time series data augmentation with Transformer architecture
    """
    def __init__(self, noise_dim=512, condition_dim=128, sequence_length=27, feature_dim=34):
        super(TimeSeriesGenerator, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.d_model = 768  # Transformer dimension
        
        # Conditional embedding
        self.condition_embedding = nn.Embedding(20, condition_dim)
        
        # Generator network with Transformer layers
        self.fc1 = nn.Linear(noise_dim + condition_dim * 4, self.d_model)
        
        # Multi-layer Transformer 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=16,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(self.d_model, 512, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(1024, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm3 = nn.LSTM(512, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Output layers
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, feature_dim)
        
        # Attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, noise, conditions):
        batch_size = noise.size(0)
        
        # Embed conditions
        cond_embeds = []
        for i in range(4):
            cond_embeds.append(self.condition_embedding(conditions[:, i]))
        cond_embed = torch.cat(cond_embeds, dim=1)
        
        # Concatenate noise and condition
        x = torch.cat([noise, cond_embed], dim=1)
        x = self.relu(self.fc1(x))
        x = self.layer_norm(x)
        
        # Expand for sequence generation
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Transformer processing
        x = self.transformer(x)
        
        # LSTM stack
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        
        # Self-attention
        x, _ = self.self_attention(x, x, x)
        
        # Output layers
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return self.tanh(x)

class TimeSeriesDiscriminator(nn.Module):
    """
    GAN Discriminator for time series data
    """
    def __init__(self, sequence_length=27, feature_dim=34, condition_dim=128):
        super(TimeSeriesDiscriminator, self).__init__()
        
        # Conditional embedding
        self.condition_embedding = nn.Embedding(20, condition_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=16,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(512, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # Classification layers
        self.fc1 = nn.Linear(128 + condition_dim * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x, conditions):
        batch_size = x.size(0)
        
        # Convolutional processing
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # LSTM stack
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        
        # Cross-attention
        x, _ = self.cross_attention(x, x, x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Conditional information
        cond_embeds = []
        for i in range(4):
            cond_embeds.append(self.condition_embedding(conditions[:, i]))
        cond_embed = torch.cat(cond_embeds, dim=1)
        
        # Concatenate features and conditions
        x = torch.cat([x, cond_embed], dim=1)
        
        # Classification layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return self.sigmoid(x)

class ConditionalGAN:
    """
    Conditional GAN for table tennis sensor data augmentation
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = TimeSeriesGenerator().to(device)
        self.discriminator = TimeSeriesDiscriminator().to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Loss functions
        self.criterion_bce = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def train_step(self, real_data, real_conditions, noise_dim=512):
        batch_size = real_data.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9  # Label smoothing
        fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
        
        # Generate noise
        noise = torch.randn(batch_size, noise_dim).to(self.device)
        
        # =================== Train Discriminator ===================
        self.d_optimizer.zero_grad()
        
        # Real data
        d_real = self.discriminator(real_data, real_conditions)
        d_real_loss = self.criterion_bce(d_real, real_labels)
        
        # Fake data
        fake_data = self.generator(noise, real_conditions)
        d_fake = self.discriminator(fake_data.detach(), real_conditions)
        d_fake_loss = self.criterion_bce(d_fake, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # =================== Train Generator ===================
        self.g_optimizer.zero_grad()
        
        # Generate fake data and try to fool discriminator
        fake_data = self.generator(noise, real_conditions)
        d_fake = self.discriminator(fake_data, real_conditions)
        g_loss_adv = self.criterion_bce(d_fake, real_labels)
        
        # Add reconstruction loss for better training
        g_loss_recon = self.criterion_mse(fake_data, real_data) * 0.1
        g_loss = g_loss_adv + g_loss_recon
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()

    def train(self, dataloader, epochs=100):
        """
        Train the GAN model
        """
        print("Starting GAN training...")
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for batch_idx, (real_data, conditions) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                conditions = conditions.to(self.device)
                
                g_loss, d_loss = self.train_step(real_data, conditions)
                g_loss_epoch += g_loss
                d_loss_epoch += d_loss
            
            # Record losses
            self.g_losses.append(g_loss_epoch / len(dataloader))
            self.d_losses.append(d_loss_epoch / len(dataloader))
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], G_Loss: {g_loss_epoch/len(dataloader):.4f}, D_Loss: {d_loss_epoch/len(dataloader):.4f}')
    
    def generate_synthetic_data(self, conditions, num_samples=100):
        """
        Generate synthetic data for given conditions
        """
        self.generator.eval()
        synthetic_data = []
        
        with torch.no_grad():
            for i in range(0, num_samples, 32):
                batch_size = min(32, num_samples - i)
                noise = torch.randn(batch_size, 512).to(self.device)
                batch_conditions = conditions[:batch_size].to(self.device)
                
                fake_data = self.generator(noise, batch_conditions)
                synthetic_data.append(fake_data.cpu().numpy())
        
        return np.concatenate(synthetic_data, axis=0)

def prepare_gan_data(x_train, y_train):
    """
    Prepare data for GAN training
    """
    sequence_length = 27
    feature_dim = x_train.shape[1]
    
    # Group data by sequences
    num_sequences = len(x_train) // sequence_length
    x_sequences = x_train[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, feature_dim)
    
    # Prepare conditions
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

def augment_data_with_gan(x_train, y_train, augmentation_ratio=0.5):
    """
    Main function to augment training data using GAN
    """
    print("Preparing data for GAN training...")
    x_sequences, conditions, label_encoders = prepare_gan_data(x_train, y_train)
    
    # Convert to tensors
    x_tensor = torch.FloatTensor(x_sequences)
    conditions_tensor = torch.LongTensor(conditions)
    
    # Create dataset and dataloader - use larger batch size for better training
    dataset = TensorDataset(x_tensor, conditions_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    
    # Initialize and train GAN
    gan = ConditionalGAN()
    
    try:
        gan.train(dataloader, epochs=150)  # More epochs for complex architecture
        
        # Generate synthetic data
        num_synthetic = int(len(x_sequences) * augmentation_ratio)
        print(f"Generating {num_synthetic} synthetic sequences...")
        
        synthetic_conditions = conditions_tensor[:num_synthetic]
        synthetic_sequences = gan.generate_synthetic_data(synthetic_conditions, num_synthetic)
        
        # Reshape back to original format
        synthetic_data = synthetic_sequences.reshape(-1, x_train.shape[1])
        
        # Create corresponding labels
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
        
        # Combine original and synthetic data
        x_augmented = np.vstack([x_train, synthetic_data])
        y_augmented = pd.concat([y_train, synthetic_labels_df], ignore_index=True)
        
        print(f"Data augmentation completed. Original size: {len(x_train)}, Augmented size: {len(x_augmented)}")
        
        return x_augmented, y_augmented, gan
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory: {e}")
        print("The Transformer+LSTM architecture requires substantial GPU memory.")
        print("Consider using a GPU with more VRAM or reducing batch size.")
        return x_train, y_train, None
        
    except Exception as e:
        print(f"GAN training failed: {e}")
        print("Complex architecture encountered training difficulties.")
        print("Falling back to original data without augmentation...")
        return x_train, y_train, None

def test_pred(X_train_scaled, y_train, scaler, le):
    """Generate predictions for test data and create submission file"""
    # Load test info
    test_info = pd.read_csv('./test_info.csv')
    test_unique_id = test_info['unique_id'].unique()
    test_data_path = './tabular_data_test'
    test_data_list = list(Path(test_data_path).glob('**/*.csv'))
    
    test_data = pd.DataFrame()
    
    for file in test_data_list:
        unique_id = int(Path(file).stem)
        row = test_info[test_info['unique_id'] == unique_id]
        if row.empty:
            continue
        data = pd.read_csv(file)
        if len(data) == 0:
            test_unique_id = test_unique_id[test_unique_id != unique_id]
            print(f'{unique_id} empty')
            continue
        test_data = pd.concat([test_data, data], ignore_index=True)

    test_data_scaled = scaler.transform(test_data)
    
    group_size = 27
    
    def predict_binary(X_train, y_train, X_test):
        """Return predicted probability for binary classification"""
        clf = RandomForestClassifier(random_state=42, n_estimators=200)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        predicted = [predicted[i][1] for i in range(len(predicted))]  # Probability of positive class
        
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            y_pred.append(max(group_pred))  # Take max probability in group
        
        return y_pred
    
    def predict_multiary(X_train, y_train, X_test):
        """Return predicted probabilities for multiclass classification"""
        clf = RandomForestClassifier(random_state=42, n_estimators=200)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            # Average probabilities across the group
            avg_probs = np.mean(group_pred, axis=0)
            y_pred.append(avg_probs)
        
        return y_pred

    # Generate predictions
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_train_le_years = le.fit_transform(y_train['play years'])
    y_train_le_level = le.fit_transform(y_train['level'])
    
    gender_pred = predict_binary(X_train_scaled, y_train_le_gender, test_data_scaled)
    hold_pred = predict_binary(X_train_scaled, y_train_le_hold, test_data_scaled)
    years_pred = predict_multiary(X_train_scaled, y_train_le_years, test_data_scaled)
    level_pred = predict_multiary(X_train_scaled, y_train_le_level, test_data_scaled)
    
    # Create submission file
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['unique_id', 'gender', 'hold racket handed', 
                        'play years_0', 'play years_1', 'play years_2', 
                        'level_2', 'level_3', 'level_4', 'level_5'])
        
        for i in range(len(test_unique_id)):
            if i < len(gender_pred) and i < len(hold_pred) and i < len(years_pred) and i < len(level_pred):
                row = [test_unique_id[i], 
                       gender_pred[i], 
                       hold_pred[i],
                       years_pred[i][0] if len(years_pred[i]) > 0 else 0,
                       years_pred[i][1] if len(years_pred[i]) > 1 else 0,
                       years_pred[i][2] if len(years_pred[i]) > 2 else 0,
                       level_pred[i][0] if len(level_pred[i]) > 0 else 0,
                       level_pred[i][1] if len(level_pred[i]) > 1 else 0,
                       level_pred[i][2] if len(level_pred[i]) > 2 else 0,
                       level_pred[i][3] if len(level_pred[i]) > 3 else 0]
                writer.writerow(row)
    
    print("Submission file created: submission.csv")

def main_with_gan():
    """
    Main function attempting to use GAN for data augmentation
    """
    print("=== Attempting GAN-based Data Augmentation ===")
    
    # Load and prepare data (same as baseline)
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
    
    # Standardize features
    scaler = MinMaxScaler()
    le = LabelEncoder()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    print(f"Original training data shape: {x_train_scaled.shape}")
    
    # Attempt GAN augmentation
    try:
        x_train_augmented, y_train_augmented, gan_model = augment_data_with_gan(x_train_scaled, y_train, augmentation_ratio=0.3)
        if x_train_augmented is not None:
            x_train_final = scaler.fit_transform(x_train_augmented)
            y_train_final = y_train_augmented
            print("GAN augmentation successful!")
        else:
            x_train_final = x_train_scaled
            y_train_final = y_train
    except Exception as e:
        print(f"GAN augmentation failed: {e}")
        print("Using original data...")
        x_train_final = x_train_scaled
        y_train_final = y_train
    
    # Evaluation functions
    group_size = 27
    
    def model_binary(X_train, y_train, X_test, y_test, target_name):
        clf = RandomForestClassifier(random_state=42, n_estimators=200)
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
        clf = RandomForestClassifier(random_state=42, n_estimators=200)
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
