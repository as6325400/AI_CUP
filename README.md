# 🎾 Conditional GAN for Time-Series Data Augmentation

本項目實作了一個基於 Transformer + LSTM 架構的 Conditional GAN，用於強化式生成時間序列資料，特別適用於乒乓球感測器數據的增強與分類任務。模型支援條件生成，可根據四種條件特徵生成對應序列。

---

## 📐 Generator 架構

```
[Noise + 4xCondition Embeddings]
           ↓
    [Linear → ReLU → LayerNorm]
           ↓
   [Expand to (sequence_length, d_model)]
           ↓
     [Transformer Encoder × 8]
           ↓
 [LSTM1 (Bi) → LSTM2 (Bi) → LSTM3]
           ↓
     [Self-Attention Layer]
           ↓
 [FC → ReLU → Dropout → FC → ReLU → FC]
           ↓
      [Tanh → Generated Sequence]
```

---

## 🛡 Discriminator 架構

```
       [Input Sequence]
               ↓
     [Conv1D × 3 + ReLU]
               ↓
     [Transformer Encoder × 4]
               ↓
     [LSTM1 (Bi) → LSTM2]
               ↓
     [Self-Attention Layer]
               ↓
     [Global Average Pooling]
               ↓
    [Concat with 4xCondition Embeds]
               ↓
     [FC → ReLU → Dropout → FC → ReLU → FC]
               ↓
           [Sigmoid → Real/Fake]
```

---

## 🚀 使用方式

### 安裝依賴
使用該 Dockerfile 的環境

### 執行主程式
```bash
python gan.py
```

此主程式包含：

- 資料預處理與條件編碼
- GAN 訓練與資料增強
- 使用 RandomForest 進行特徵分類
- 自動產生 submission.csv

---

## 📈 支援的目標欄位

| 欄位名稱            | 類型     |
|---------------------|----------|
| `gender`            | Binary   |
| `hold racket handed`| Binary   |
| `play years`        | Multiary (3 classes) |
| `level`             | Multiary (4 classes) |

---

## 📂 資料結構說明

```
project/
│
├── train_info.csv
├── test_info.csv
├── tabular_data_train/
│   └── 0001.csv, 0002.csv, ...
├── tabular_data_test/
│   └── 1968.csv, 1969.csv, ...
├── gan.py
└── submission.csv
```

---


## 📤 測試與提交

程式會自動根據測試資料生成 `submission.csv`，格式如下：

```csv
unique_id,gender,hold racket handed,play years_0,...,level_5
1001,0.65,0.72,0.12,...,0.33
...
```

---

## 📌 備註

- 本程式碼已包括 LabelEncoder 的逆轉換與 group-based max pooling 用於符合序列任務需求。
- 採用 Label Smoothing 與 MSE 重建損失輔助訓練 GAN，以穩定生成效果。
