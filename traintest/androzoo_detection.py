import os
import sys
import argparse
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from feature_cnn_models import SmaliOpcodeDetailCaptureCNN, SOOpcodeDetailCaptureCNN
except ImportError:
    print("Error: feature_cnn_models.py not found.")
    sys.exit(1)

ApiCall_OT_ThresholdAnalyzer = None
try:
    from test.ot_thre import ApiCall_OT_ThresholdAnalyzer as _ApiCall_OT_ThresholdAnalyzer
    ApiCall_OT_ThresholdAnalyzer = _ApiCall_OT_ThresholdAnalyzer
except ImportError as e:
    print(f"Warning: failed to import ApiCall analyzer ({e}). ApiCall detection will be skipped.")


class ProjectSample:
    def __init__(self, group_id, project_path, is_original):
        self.group_id = group_id
        self.path = project_path
        self.is_original = is_original
        self.name = os.path.basename(project_path)
        
        self.so_npy = self._find_so_npy()
        self.smaliopcode_npy = self._find_smaliopcode_npy()
        self.gexf = self._find_gexf()
        self.icon = self._find_icon()
        
        self.so_vector = None
        self.smaliopcode_vector = None
        self.icon_content_vector = None
        self.icon_style_vector = None

    def _find_so_npy(self):
        p1 = os.path.join(self.path, 'transition_probabilities.npy')
        if os.path.exists(p1): return p1
        p2 = os.path.join(self.path, 'so_transition_probabilities.npy')
        if os.path.exists(p2): return p2
        return None

    def _find_smaliopcode_npy(self):
        p1 = os.path.join(self.path, 'dalvik.npy')
        if os.path.exists(p1): return p1
        p2 = os.path.join(self.path, 'smaliopcode', 'dalvik.npy')
        if os.path.exists(p2): return p2
        return None

    def _find_gexf(self):
        p = os.path.join(self.path, 'community_processed_graph.gexf')
        return p if os.path.exists(p) else None

    def _find_icon(self):
        img_dir = os.path.join(self.path, 'images')
        if not os.path.isdir(img_dir): return None
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(img_dir, f)
        return None

    def has_all_features(self):
        return True

def scan_androzoo(root_dir, limit=None):
    """
    Scans the AndroZoo directory structure.
    Returns a dict: { group_id: {'original': sample, 'repacks': [sample, ...]} }
    """
    groups = {}
    
    if not os.path.exists(root_dir):
        return groups
        
    group_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    group_ids.sort(key=int)
    
    if limit:
        group_ids = group_ids[:limit]
    
    print(f"Scanning {len(group_ids)} groups...")
    
    for gid in tqdm(group_ids, desc="Scanning Groups"):
        group_path = os.path.join(root_dir, gid)
        orig_dir = os.path.join(group_path, 'original_apk')
        repack_dir = os.path.join(group_path, 'repack_apk')
        
        original_sample = None
        if os.path.isdir(orig_dir):
            subdirs = [os.path.join(orig_dir, d) for d in os.listdir(orig_dir) if os.path.isdir(os.path.join(orig_dir, d))]
            if subdirs:
                original_sample = ProjectSample(gid, subdirs[0], True)
        
        if not original_sample:
            continue
            
        repack_samples = []
        if os.path.isdir(repack_dir):
            subdirs = [os.path.join(repack_dir, d) for d in os.listdir(repack_dir) if os.path.isdir(os.path.join(repack_dir, d))]
            for sd in subdirs:
                repack_samples.append(ProjectSample(gid, sd, False))
        
        if repack_samples:
            groups[gid] = {
                'original': original_sample,
                'repacks': repack_samples
            }
            
    return groups


class MatrixDataset(Dataset):
    def __init__(self, samples, feature_type='so'):
        self.samples = samples
        self.feature_type = feature_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.so_npy if self.feature_type == 'so' else sample.smaliopcode_npy
        
        if not path:
            if self.feature_type == 'so':
                return torch.zeros((1, 94, 94), dtype=torch.float32), idx
            else:
                return torch.zeros((1, 256, 256), dtype=torch.float32), idx

    def load_matrix(self, path):
        try:
            mat = np.load(path, allow_pickle=True)
            t = torch.tensor(mat, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return t
        except:
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.so_npy if self.feature_type == 'so' else sample.smaliopcode_npy
        
        default_size = 94 if self.feature_type == 'so' else 256
        zeros = torch.zeros((1, default_size, default_size), dtype=torch.float32)
        
        if not path:
            return zeros, idx
            
        try:
            mat = np.load(path, allow_pickle=True)
            t = torch.tensor(mat, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            
            if self.feature_type == 'smaliopcode':
                if t.shape[1] != 256 or t.shape[2] != 256:
                    pass
            
            return t, idx
        except Exception:
            return zeros, idx

class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.icon
        
        if not path:
            return torch.zeros((3, 224, 224)), idx, False
            
        try:
            image = Image.open(path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_t = self.transform(image)
            return img_t, idx, True
        except:
            return torch.zeros((3, 224, 224)), idx, False


def extract_features(samples, model, feature_type, batch_size, device):
    """
    Extract features for a list of samples using the given model.
    Updates the samples in-place with vectors.
    """
    if not samples:
        return

    model.eval()
    model.to(device)
    
    if feature_type == 'icon':
        dataset = ImageDataset(samples)
    else:
        dataset = MatrixDataset(samples, feature_type)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Extracting {feature_type} features...")
    
    with torch.no_grad():
        for data in tqdm(loader):
            if feature_type == 'icon':
                imgs, idxs, valids = data
                imgs = imgs.to(device)
                conv_feat, fc2 = model(imgs)
                if conv_feat is None:
                    f_content = None
                    grams = None
                else:
                    f_content = torch.nn.functional.normalize(fc2, p=2, dim=1)
                    b, ch, h, w = conv_feat.size()
                    f_style = conv_feat.view(b, ch, h * w)
                    grams = torch.bmm(f_style, f_style.transpose(1, 2)) / (ch * h * w)
                    grams = grams.view(b, -1)
                    grams = torch.nn.functional.normalize(grams, p=2, dim=1)
            else:
                mats, idxs = data
                mats = mats.to(device)
                outputs = model(mats)
                
            if feature_type == 'icon':
                if f_content is None or grams is None:
                    content_np = None
                    style_np = None
                else:
                    content_np = f_content.cpu().numpy().astype(np.float16, copy=False)
                    style_np = grams.cpu().numpy().astype(np.float16, copy=False)
            else:
                outputs = outputs.cpu().numpy()
            
            for i, idx in enumerate(idxs):
                sample = samples[idx]
                
                is_valid = False
                if feature_type == 'so':
                    vec = outputs[i]
                    if sample.so_npy: is_valid = True
                    sample.so_vector = vec if is_valid else None
                elif feature_type == 'smaliopcode':
                    vec = outputs[i]
                    if sample.smaliopcode_npy: is_valid = True
                    sample.smaliopcode_vector = vec if is_valid else None
                elif feature_type == 'icon':
                    if bool(valids[i]) and content_np is not None and style_np is not None:
                        sample.icon_content_vector = content_np[i]
                        sample.icon_style_vector = style_np[i]
                    else:
                        sample.icon_content_vector = None
                        sample.icon_style_vector = None

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        weights = VGG19_Weights.DEFAULT
        base_model = vgg19(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG19ContentStyleExtractor(nn.Module):
    def __init__(self, conv_layer_name: str = "28"):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        base_model = vgg19(weights=weights)
        self.features = nn.Sequential(*list(base_model.features.children()))
        self.avgpool = base_model.avgpool
        self.fc = nn.Sequential(*list(base_model.classifier.children())[:-1])
        self.conv_layer_name = conv_layer_name

    def forward(self, x):
        conv_feat = None
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name == self.conv_layer_name:
                conv_feat = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fc2 = self.fc(x)
        return conv_feat, fc2


def main():
    parser = argparse.ArgumentParser(description="AndroZoo Fast Detection")
    parser.add_argument('--root', type=str, required=True, help='AndroZoo root directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--smaliopcode_model', type=str, required=True, help='Path to SmaliOpcode CNN model')
    parser.add_argument('--so_model', type=str, required=True, help='Path to SO CNN model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of groups')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--apicall_threshold', type=float, default=0.85, help='ApiCall threshold for acceleration')
    parser.add_argument('--apicall_prefilter_margin', type=float, default=0.05, help='ApiCall prefilter margin (set 0 to disable)')
    parser.add_argument('--icon_alpha', type=float, default=0.6, help='Icon overall weight: alpha*content + (1-alpha)*style')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    groups = scan_androzoo(args.root, args.limit)
    if not groups:
        print("No valid groups found.")
        return

    all_samples = []
    for gid in groups:
        all_samples.append(groups[gid]['original'])
        all_samples.extend(groups[gid]['repacks'])
    
    print(f"Total samples found: {len(all_samples)}")
    
    
    print("Loading SO Model...")
    so_net = SOOpcodeDetailCaptureCNN()
    try:
        so_net.load_state_dict(torch.load(args.so_model, map_location=device))
    except Exception as e:
        print(f"Failed to load SO model: {e}")
        sys.exit(1)
    extract_features(all_samples, so_net, 'so', args.batch_size, device)
    del so_net
    
    print("Loading SmaliOpcode Model...")
    smaliopcode_net = SmaliOpcodeDetailCaptureCNN()
    try:
        smaliopcode_net.load_state_dict(torch.load(args.smaliopcode_model, map_location=device))
    except Exception as e:
        print(f"Failed to load SmaliOpcode model: {e}")
        sys.exit(1)
    extract_features(all_samples, smaliopcode_net, 'smaliopcode', args.batch_size, device)
    del smaliopcode_net
    
    print("Loading Icon Model (VGG19 content/style)...")
    icon_net = VGG19ContentStyleExtractor()
    extract_features(all_samples, icon_net, 'icon', args.batch_size, device)
    del icon_net
    
    print("Constructing Pairs...")
    pairs = []
    
    group_ids = list(groups.keys())
    
    for gid in group_ids:
        orig = groups[gid]['original']
        repacks = groups[gid]['repacks']
        
        for rp in repacks:
            pairs.append({
                'p1': orig,
                'p2': rp,
                'label': 1,
                'type': 'positive'
            })
            
        for _ in repacks:
            other_gid = random.choice(group_ids)
            while other_gid == gid and len(group_ids) > 1:
                other_gid = random.choice(group_ids)
            
            if other_gid != gid:
                other_sample = groups[other_gid]['original']
                pairs.append({
                    'p1': orig,
                    'p2': other_sample,
                    'label': 0,
                    'type': 'negative'
                })

    print(f"Total pairs: {len(pairs)}")
    
    results = []
    
    apicall_analyzer = None
    if ApiCall_OT_ThresholdAnalyzer is None:
        print("ApiCall disabled: ApiCall_OT_ThresholdAnalyzer not available.")
    else:
        try:
            apicall_analyzer = ApiCall_OT_ThresholdAnalyzer(args.root, device=device.type)
        
            print("Preloading ApiCall Graph Features (this may take time)...")
            valid_apicall_samples = [s for s in all_samples if s.gexf]
            valid_apicall_paths = [s.path for s in valid_apicall_samples]
        
            apicall_analyzer.preload_apk_features(valid_apicall_paths, max_nodes=2000) 
            print(f"Loaded {len(apicall_analyzer.features_cache)} graphs.")
        
        except Exception as e:
            print(f"ApiCall Init Warning: {e}")
            apicall_analyzer = None

    for p in tqdm(pairs, desc="Detecting"):
        s1 = p['p1']
        s2 = p['p2']
        
        row = {
            'p1_path': s1.path,
            'p2_path': s2.path,
            'label': p['label'],
            'group_id': s1.group_id
        }
        
        if s1.so_vector is not None and s2.so_vector is not None:
            sim = F.cosine_similarity(
                torch.tensor(s1.so_vector).unsqueeze(0), 
                torch.tensor(s2.so_vector).unsqueeze(0)
            ).item()
            row['so_sim'] = sim
        else:
            row['so_sim'] = -1.0
            
        if s1.smaliopcode_vector is not None and s2.smaliopcode_vector is not None:
            sim = F.cosine_similarity(
                torch.tensor(s1.smaliopcode_vector).unsqueeze(0), 
                torch.tensor(s2.smaliopcode_vector).unsqueeze(0)
            ).item()
            row['smaliopcode_sim'] = sim
        else:
            row['smaliopcode_sim'] = -1.0
            
        if s1.icon_content_vector is not None and s2.icon_content_vector is not None and s1.icon_style_vector is not None and s2.icon_style_vector is not None:
            content_sim = float(np.sum(s1.icon_content_vector * s2.icon_content_vector, dtype=np.float32))
            style_sim = float(np.sum(s1.icon_style_vector * s2.icon_style_vector, dtype=np.float32))
            overall_sim = float(args.icon_alpha * content_sim + (1.0 - args.icon_alpha) * style_sim)
            row['icon_content_similarity'] = content_sim
            row['icon_style_similarity'] = style_sim
            row['icon_sim'] = overall_sim
        else:
            row['icon_content_similarity'] = -1.0
            row['icon_style_similarity'] = -1.0
            row['icon_sim'] = -1.0
            
        if apicall_analyzer and s1.gexf and s2.gexf:
            try:
                f1 = apicall_analyzer.features_cache.get(s1.path)
                f2 = apicall_analyzer.features_cache.get(s2.path)
                
                sim = None
                if args.apicall_prefilter_margin > 0 and f1 is not None and f2 is not None:
                    approx_sim = apicall_analyzer.quick_prefilter_similarity(f1, f2)
                    if approx_sim >= (args.apicall_threshold + args.apicall_prefilter_margin) or \
                       approx_sim <= (args.apicall_threshold - args.apicall_prefilter_margin):
                        sim = approx_sim
                
                if sim is None:
                    sim = apicall_analyzer.calculate_ot_similarity(s1.path, s2.path, method='sinkhorn', sinkhorn_reg=0.1)
                
                row['apicall_sim'] = sim
            except Exception as e:
                row['apicall_sim'] = -1.0
        else:
            row['apicall_sim'] = -1.0

        results.append(row)
        
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, 'androzoo_detection_results.csv')
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()
