import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision import transforms
from torch_geometric.nn import GCNConv
from PIL import Image
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Entity Alignment for k Parties
def align_entities(dataframes, sample_ids_list):
    common_samples = set(dataframes[0].index)
    for df in dataframes[1:]:
        common_samples = common_samples.intersection(df.index)
    for sample_ids in sample_ids_list:
        common_samples = common_samples.intersection(sample_ids)
    aligned_indices = []
    for df in dataframes:
        aligned_indices.append([df.index.get_loc(s) for s in common_samples])
    aligned_sample_ids = [[ids[i] for i in range(len(ids)) if ids[i] in common_samples] for ids in sample_ids_list]
    return list(common_samples), aligned_indices, aligned_sample_ids


# MRI Feature Extraction with Enhanced Preprocessing
def extract_mri_features(image_dir, sample_ids):
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-10)),  # Z-score normalization
    ])
    features, valid_sample_ids = [], []
    for sample_id in sample_ids:
        sample_dir = os.path.join(image_dir, sample_id)
        if not os.path.exists(sample_dir):
            continue
        image_files = [os.path.join(root, f) for root, _, files in os.walk(sample_dir)
                       for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue
        sample_features = []
        for image_file in image_files:
            img = Image.open(image_file).convert('RGB')
            img_t = preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0).to(device)
            with torch.no_grad():
                output = model(batch_t)['out']
                feature_vector = torch.mean(output, dim=[2, 3]).squeeze(0).cpu().numpy()
                if not np.any(np.isnan(feature_vector)) and not np.any(np.isinf(feature_vector)):
                    sample_features.append(feature_vector)
        if sample_features:
            features.append(np.mean(sample_features, axis=0))
            valid_sample_ids.append(sample_id)
    return np.vstack(features) if features else np.array([]), valid_sample_ids


# Gene Feature Extraction
def extract_gene_features(gene_expression_df, target_variable=None, k_best=2000):
    gene_expression_df.fillna(gene_expression_df.mean(axis=0), inplace=True)
    if target_variable is not None:
        valid_samples = gene_expression_df.index.intersection(target_variable.index)
        gene_expression_df = gene_expression_df.loc[valid_samples]
        target_variable = target_variable.loc[valid_samples]
    row_norm = np.sqrt((gene_expression_df ** 2).sum(axis=1))
    gene_expression_norm = gene_expression_df.div(row_norm, axis=0).fillna(0)
    X = gene_expression_norm.values
    final_index = gene_expression_norm.index
    if target_variable is not None:
        selector = SelectKBest(f_classif, k=k_best)
        X = selector.fit_transform(X, target_variable.values)
    return X, final_index, target_variable


# Probabilistic Graph Construction with Cosine Similarity Threshold
def build_probabilistic_graph(features, k=60, cosine_threshold=0.5):
    num_samples = features.shape[0]
    distances = np.zeros((num_samples, num_samples))
    similarities = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            sim = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-10)
            similarities[i, j] = similarities[j, i] = sim
            distances[i, j] = distances[j, i] = 1 - sim
    edge_index, edge_weight = [], []
    for i in range(num_samples):
        sorted_indices = np.argsort(distances[i])
        valid_neighbors = [j for j in sorted_indices if similarities[i, j] >= cosine_threshold and i != j]
        if len(valid_neighbors) > k:
            valid_neighbors = valid_neighbors[:k]
        if not valid_neighbors:
            continue
        d_k_plus_1 = distances[i, sorted_indices[min(k + 1, len(sorted_indices) - 1)]]
        denominator = k / (d_k_plus_1 - sum(distances[i, valid_neighbors[:min(k, len(valid_neighbors))]]))
        for j in valid_neighbors[:k]:
            p_ij = max(0, d_k_plus_1 - distances[i, j]) * denominator
            if p_ij > 0:
                edge_index.append([i, j])
                edge_weight.append(p_ij)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device) if edge_index else torch.empty(
        (2, 0), dtype=torch.long).to(device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device) if edge_weight else torch.empty((0,),
                                                                                                          dtype=torch.float).to(
        device)
    return edge_index, edge_weight


# Local GCN Encoder with Mutual Information
class ViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ViewEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bilinear = nn.Parameter(torch.randn(hidden_dim, input_dim))

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x  # Linear activation for second layer

    def mutual_information_loss(self, h, x, edge_index, adj):
        pos_score = torch.sigmoid(torch.matmul(h, torch.matmul(self.bilinear, x.t())))
        neg_x = x[torch.randperm(x.size(0)).to(device)]
        neg_score = torch.sigmoid(torch.matmul(h, torch.matmul(self.bilinear, neg_x.t())))
        mi_feature = torch.mean(-F.softplus(-pos_score) + F.softplus(neg_score))
        recon_adj = torch.sigmoid(torch.matmul(h, h.t()))
        mi_topology = F.binary_cross_entropy_with_logits(recon_adj, adj, reduction='mean')
        return mi_feature + mi_topology


# Fusion Layer
class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_views):
        super(FusionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(num_views))
        nn.init.uniform_(self.attention_weights)

    def forward(self, view_reps):
        weights = torch.softmax(self.attention_weights, dim=0)
        fused_rep = sum(weights[i] * rep for i, rep in enumerate(view_reps))
        return fused_rep


# GCN-based Encoder-Decoder for Refinement
class FusionRefiner(nn.Module):
    def __init__(self, hidden_dim, num_views):
        super(FusionRefiner, self).__init__()
        self.encoder_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.decoder_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for _ in range(num_views)])

    def forward(self, fused_rep, global_edge_index):
        z = self.encoder_conv1(fused_rep, global_edge_index)
        z = torch.relu(z)
        z = self.encoder_conv2(z, global_edge_index)
        recon_adjs = [torch.sigmoid(torch.matmul(z, w) @ z.t()) for w in self.decoder_weights]
        return z, recon_adjs


# Generalized VFLING Model for k Parties
class VFLINGModel(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_classes, num_views):
        super(VFLINGModel, self).__init__()
        self.view_encoders = nn.ModuleList([ViewEncoder(dim, hidden_dim) for dim in input_dims])
        self.fusion_layer = FusionLayer(hidden_dim, num_views)
        self.refiner = FusionRefiner(hidden_dim, num_views)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_list, edge_index_list, edge_weight_list, adj_list):
        view_reps = [encoder(x, edge_idx, edge_w) for encoder, x, edge_idx, edge_w in
                     zip(self.view_encoders, x_list, edge_index_list, edge_weight_list)]
        fused_rep = self.fusion_layer(view_reps)
        z, recon_adjs = self.refiner(fused_rep, edge_index_list[0])  # Use first graph for global structure
        logits = self.classifier(z)
        return logits, recon_adjs, view_reps


# Single-Modal Model
class SingleModalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SingleModalModel, self).__init__()
        self.encoder = ViewEncoder(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_list, edge_index_list, edge_weight_list, adj_list):
        x = x_list[0]
        edge_index = edge_index_list[0]
        edge_weight = edge_weight_list[0]
        adj = adj_list[0]
        rep = self.encoder(x, edge_index, edge_weight)
        logits = self.classifier(rep)
        return logits, [torch.sigmoid(torch.matmul(rep, rep.t()))], [rep]


# Training Function with Early Stopping
def train_model(model, x_list, edge_index_list, edge_weight_list, adj_list, y_tensor, epochs=1000, lr=0.001,
                class_weights=None, patience=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, recon_adjs, view_reps = model(x_list, edge_index_list, edge_weight_list, adj_list)
        class_loss = criterion(logits, y_tensor)
        mi_loss = sum(encoder.mutual_information_loss(rep, x, edge_idx, adj)
                      for encoder, rep, x, edge_idx, adj in
                      zip([model.encoder] if isinstance(model, SingleModalModel) else model.view_encoders,
                          view_reps, x_list, edge_index_list, adj_list))
        recon_loss = sum(F.binary_cross_entropy(recon_adj, adj) for recon_adj, adj in zip(recon_adjs, adj_list))
        total_loss = class_loss + 0.1 * mi_loss + 0.1 * recon_loss
        total_loss.backward()
        optimizer.step()
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}")


# Main Function
def main():
    gene_path = '../data/gene/breast_expr_c.csv'
    mri_dir = '../data/BRCA_120/'
    brca_info_path = '../data/gene/brca.csv'
    dna_methylation_path = '../data/dna_methylation.csv'  # Example path for additional modality

    brca_df = pd.read_csv(brca_info_path, index_col=0)
    brca_df = brca_df[brca_df['BRCA_Subtype_PAM50'] != "Normal"]
    gene_df = pd.read_csv(gene_path, index_col=0).transpose()
    dna_methylation_df = pd.read_csv(dna_methylation_path, index_col=0).transpose() if os.path.exists(
        dna_methylation_path) else None

    # Collect dataframes and sample IDs
    dataframes = [gene_df, brca_df]
    sample_ids_list = [gene_df.index.tolist()]
    if dna_methylation_df is not None:
        dataframes.append(dna_methylation_df)
        sample_ids_list.append(dna_methylation_df.index.tolist())

    mri_sample_ids = gene_df.index.tolist()  # Initial list for MRI
    mri_features, mri_valid_sample_ids = extract_mri_features(mri_dir, mri_sample_ids)
    sample_ids_list.append(mri_valid_sample_ids)

    # Entity Alignment
    common_samples, aligned_indices, aligned_sample_ids = align_entities(dataframes, sample_ids_list)
    if not common_samples:
        print("No common samples found after alignment.")
        return

    # Update dataframes
    brca_df = brca_df.loc[common_samples]
    gene_df = gene_df.loc[common_samples]
    if dna_methylation_df is not None:
        dna_methylation_df = dna_methylation_df.loc[common_samples]

    # Extract features
    X_gene, _, y = extract_gene_features(gene_df, brca_df['BRCA_Subtype_PAM50'], k_best=2000)
    mri_idx_map = {sid: i for i, sid in enumerate(mri_valid_sample_ids)}
    X_mri = mri_features[[mri_idx_map[sid] for sid in common_samples if sid in mri_idx_map]]
    X_dna = extract_gene_features(dna_methylation_df, k_best=2000)[0] if dna_methylation_df is not None else None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    print(f"Total number of classes: {num_classes}, Classes: {encoder.classes_}")

    # Prepare feature lists
    feature_list = [X_gene, X_mri]
    if X_dna is not None:
        feature_list.append(X_dna)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    metrics_multi = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    metrics_mri = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    metrics_gene = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_gene, y_encoded), 1):
        print(f"\n====== Fold {fold} ======")

        # Prepare training and validation data
        X_train_list = [X[train_idx] for X in feature_list]
        X_val_list = [X[val_idx] for X in feature_list]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Scale features
        scalers = [StandardScaler().fit(X_train) for X_train in X_train_list]
        X_train_scaled = [scaler.transform(X_train) for scaler, X_train in zip(scalers, X_train_list)]
        X_val_scaled = [scaler.transform(X_val) for scaler, X_val in zip(scalers, X_val_list)]

        # Build graphs
        edge_index_list = []
        edge_weight_list = []
        adj_list = []
        for X_train_scaled_i in X_train_scaled:
            edge_index, edge_weight = build_probabilistic_graph(X_train_scaled_i, k=60, cosine_threshold=0.5)
            adj = torch.zeros((len(train_idx), len(train_idx))).to(device)
            for i, j in edge_index.t().tolist():
                adj[i, j] = 1
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            adj_list.append(adj)

        x_train_tensors = [torch.tensor(X_train, dtype=torch.float).to(device) for X_train in X_train_scaled]
        x_val_tensors = [torch.tensor(X_val, dtype=torch.float).to(device) for X_val in X_val_scaled]
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

        # Multi-Modal Model
        print("Training Multi-Modal Model...")
        input_dims = [X.shape[1] for X in feature_list]
        model_multi = VFLINGModel(input_dims, hidden_dim=128, num_classes=num_classes, num_views=len(feature_list)).to(
            device)
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float).to(device)
        train_model(model_multi, x_train_tensors, edge_index_list, edge_weight_list, adj_list, y_train_tensor,
                    epochs=1000, class_weights=class_weights, patience=100)

        # MRI-Only Model
        print("Training MRI-Only Model...")
        model_mri = SingleModalModel(X_mri.shape[1], hidden_dim=128, num_classes=num_classes).to(device)
        train_model(model_mri, [x_train_tensors[1]], [edge_index_list[1]], [edge_weight_list[1]], [adj_list[1]],
                    y_train_tensor, epochs=1000, class_weights=class_weights, patience=100)

        # mRNA-Only Model
        print("Training mRNA-Only Model...")
        model_gene = SingleModalModel(X_gene.shape[1], hidden_dim=128, num_classes=num_classes).to(device)
        train_model(model_gene, [x_train_tensors[0]], [edge_index_list[0]], [edge_weight_list[0]], [adj_list[0]],
                    y_train_tensor, epochs=1000, class_weights=class_weights, patience=100)

        # Evaluation
        models = {
            'Multi-Modal': (model_multi, metrics_multi, x_val_tensors, edge_index_list, edge_weight_list, adj_list),
            'MRI-Only': (
            model_mri, metrics_mri, [x_val_tensors[1]], [edge_index_list[1]], [edge_weight_list[1]], [adj_list[1]]),
            'mRNA-Only': (
            model_gene, metrics_gene, [x_val_tensors[0]], [edge_index_list[0]], [edge_weight_list[0]], [adj_list[0]])}
        for name, (model, metrics, x_val, edge_idx, edge_w, adj) in models.items():
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(x_val, edge_idx, edge_w, adj)
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            acc = accuracy_score(y_val, val_preds)
            prec = precision_score(y_val, val_preds, average='macro', zero_division=0)
            rec = recall_score(y_val, val_preds, average='macro', zero_division=0)
            f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
            metrics['acc'].append(acc)
            metrics['prec'].append(prec)
            metrics['rec'].append(rec)
            metrics['f1'].append(f1)
            print(f"{name} Fold {fold} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # Summary
    print("\n====== 5-Fold CV Results ======")
    for name, metrics in [('Multi-Modal', metrics_multi), ('MRI-Only', metrics_mri), ('mRNA-Only', metrics_gene)]:
        print(f"\n{name}:")
        for metric, values in metrics.items():
            print(f"Average {metric.capitalize()}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main()