#!/usr/bin/env python3
"""
Intelligent Layer Matching using Machine Learning

This module uses ML to match soil layers between boreholes based on:
- Soil symbol (AASHTO classification: A-1 to A-7)
- Color (foreground and background hex colors)
- Description (text similarity)
- Depth patterns

The model learns to predict similarity scores between layers,
enabling more accurate geological cross-section generation.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class LayerFeatureExtractor:
    """Extract features from soil layer data for ML matching"""
    
    def __init__(self):
        self.symbol_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # AASHTO soil classification categories
        self.aashto_categories = {
            'A-1': {'type': 'gravel', 'quality': 'excellent', 'code': 1},
            'A-2': {'type': 'sand_gravel', 'quality': 'excellent', 'code': 2},
            'A-3': {'type': 'fine_sand', 'quality': 'excellent', 'code': 3},
            'A-4': {'type': 'silt', 'quality': 'fair', 'code': 4},
            'A-5': {'type': 'silt', 'quality': 'poor', 'code': 5},
            'A-6': {'type': 'clay', 'quality': 'poor', 'code': 6},
            'A-7': {'type': 'clay', 'quality': 'poor', 'code': 7},
        }
    
    def hex_to_rgb_normalized(self, hex_color):
        """Convert hex color to normalized RGB values (0-1)"""
        if not hex_color:
            return [0.5, 0.5, 0.5]
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return [0.5, 0.5, 0.5]
        try:
            return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        except:
            return [0.5, 0.5, 0.5]
    
    def get_aashto_features(self, symbol):
        """Extract AASHTO-based features from soil symbol"""
        if not symbol:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Parse symbol (e.g., "A-2", "A-7-5")
        parts = symbol.split('-')
        base_symbol = parts[0] + '-' + parts[1] if len(parts) >= 2 else symbol
        
        if base_symbol in self.aashto_categories:
            cat = self.aashto_categories[base_symbol]
            return [
                cat['code'],
                1 if cat['type'] == 'gravel' else 0,
                1 if cat['type'] == 'sand_gravel' else 0,
                1 if cat['type'] == 'fine_sand' else 0,
                1 if cat['type'] == 'silt' else 0,
                1 if cat['type'] == 'clay' else 0,
                1 if cat['quality'] == 'excellent' else 0,
                1 if cat['quality'] == 'fair' else 0,
                1 if cat['quality'] == 'poor' else 0,
            ]
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    def extract_layer_features(self, layer):
        """Extract features from a single layer"""
        features = []
        
        # 1. Symbol-based features (9 features)
        symbol = layer.get('layer_symbol', 'Unknown')
        features.extend(self.get_aashto_features(symbol))
        
        # 2. Color features (6 features: RGB for foreground and background)
        fg_color = layer.get('layer_forecolor', '#808080')
        bg_color = layer.get('layer_backcolor', '#FFFFFF')
        features.extend(self.hex_to_rgb_normalized(fg_color))
        features.extend(self.hex_to_rgb_normalized(bg_color))
        
        # 3. Depth features (3 features)
        layer_from = layer.get('layer_from', 0)
        layer_to = layer.get('layer_to', 0)
        thickness = layer_to - layer_from
        features.extend([layer_from / 200.0, layer_to / 200.0, thickness / 50.0])
        
        return np.array(features)
    
    def extract_pair_features(self, layer1, layer2):
        """Extract features for a pair of layers (for similarity prediction)"""
        feat1 = self.extract_layer_features(layer1)
        feat2 = self.extract_layer_features(layer2)
        
        # Combine features: individual + difference + product
        pair_features = np.concatenate([
            feat1,
            feat2,
            np.abs(feat1 - feat2),  # Absolute difference
            feat1 * feat2,  # Element-wise product (interaction)
        ])
        
        return pair_features
    
    def compute_text_similarity(self, desc1, desc2):
        """Compute text similarity between layer descriptions"""
        if not desc1 or not desc2:
            return 0.0
        
        try:
            tfidf_matrix = self.tfidf.fit_transform([desc1, desc2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0


class IntelligentLayerMatcher:
    """ML-based layer matching between boreholes"""
    
    def __init__(self):
        self.feature_extractor = LayerFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False
        self.training_data = []
    
    def generate_training_data(self, data):
        """
        Generate training data from existing cross-section polygons.
        
        Uses the polygonsBySection data which contains expert-matched
        layer connections to create labeled training pairs.
        """
        print("Generating training data from existing cross-sections...")
        
        boreholes = data.get('boreholesData', [])
        sections = data.get('polygonsBySection', [])
        
        # Create borehole lookup
        bh_lookup = {bh.get('th_title', ''): bh for bh in boreholes}
        
        positive_pairs = []
        negative_pairs = []
        
        # For each section, layers that are connected in polygons are positive examples
        for section in sections:
            section_boreholes = section.get('boreholes', [])
            polygons = section.get('polygons', [])
            
            # Get all layer symbols that appear in polygons (these are connected)
            connected_symbols = set()
            for poly in polygons:
                symbol = poly.get('symbol', '')
                if symbol:
                    connected_symbols.add(symbol)
            
            # For boreholes in this section, create positive pairs
            # (layers with same symbol that are connected in the cross-section)
            bh_names = [bh.get('name', '') for bh in section_boreholes]
            
            for i, name1 in enumerate(bh_names):
                for j, name2 in enumerate(bh_names):
                    if i >= j:
                        continue
                    
                    bh1 = bh_lookup.get(name1)
                    bh2 = bh_lookup.get(name2)
                    
                    if not bh1 or not bh2:
                        continue
                    
                    layers1 = bh1.get('th_layers', [])
                    layers2 = bh2.get('th_layers', [])
                    
                    # Positive pairs: same symbol and both in connected set
                    for l1 in layers1:
                        for l2 in layers2:
                            sym1 = l1.get('layer_symbol', '')
                            sym2 = l2.get('layer_symbol', '')
                            
                            if sym1 == sym2 and sym1 in connected_symbols:
                                positive_pairs.append((l1, l2, 1))
                            elif sym1 != sym2:
                                negative_pairs.append((l1, l2, 0))
        
        # Balance the dataset
        n_positive = len(positive_pairs)
        if len(negative_pairs) > n_positive * 2:
            np.random.shuffle(negative_pairs)
            negative_pairs = negative_pairs[:n_positive * 2]
        
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        print(f"  Generated {len(positive_pairs)} positive pairs")
        print(f"  Generated {len(negative_pairs)} negative pairs")
        print(f"  Total training pairs: {len(all_pairs)}")
        
        self.training_data = all_pairs
        return all_pairs
    
    def train(self, data=None, training_pairs=None):
        """Train the layer matching model"""
        if training_pairs is None and data is not None:
            training_pairs = self.generate_training_data(data)
        
        if not training_pairs:
            print("No training data available!")
            return False
        
        print("\nTraining layer matching model...")
        
        # Extract features
        X = []
        y = []
        
        for l1, l2, label in training_pairs:
            features = self.feature_extractor.extract_pair_features(l1, l2)
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Test accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Different', 'Match']))
        
        return True
    
    def predict_match(self, layer1, layer2):
        """
        Predict if two layers should be matched.
        
        Returns:
            float: Probability of match (0-1)
        """
        if not self.is_trained:
            # Fallback to simple symbol matching
            return 1.0 if layer1.get('layer_symbol') == layer2.get('layer_symbol') else 0.0
        
        features = self.feature_extractor.extract_pair_features(layer1, layer2)
        proba = self.model.predict_proba([features])[0]
        
        # Return probability of match (class 1)
        return proba[1] if len(proba) > 1 else proba[0]
    
    def find_best_matches(self, layers1, layers2, threshold=0.5):
        """
        Find the best matching layers between two boreholes.
        
        Args:
            layers1: List of layers from borehole 1
            layers2: List of layers from borehole 2
            threshold: Minimum probability to consider a match
        
        Returns:
            List of (layer1_idx, layer2_idx, probability) tuples
        """
        matches = []
        used_l2 = set()
        
        # Score all pairs
        scores = []
        for i, l1 in enumerate(layers1):
            for j, l2 in enumerate(layers2):
                prob = self.predict_match(l1, l2)
                scores.append((i, j, prob))
        
        # Sort by probability descending
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching (each layer matched at most once)
        used_l1 = set()
        for i, j, prob in scores:
            if prob < threshold:
                break
            if i in used_l1 or j in used_l2:
                continue
            matches.append((i, j, prob))
            used_l1.add(i)
            used_l2.add(j)
        
        # Sort by layer1 index for consistent ordering
        matches.sort(key=lambda x: x[0])
        
        return matches
    
    def save_model(self, filepath):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'is_trained': self.is_trained
            }, f)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_extractor = data['feature_extractor']
            self.is_trained = data['is_trained']
        print(f"✓ Model loaded from: {filepath}")


def analyze_layer_distributions(data):
    """Analyze and visualize layer distributions in the dataset"""
    print("\n" + "=" * 60)
    print("LAYER DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    boreholes = data.get('boreholesData', [])
    
    symbol_counts = {}
    color_counts = {}
    depth_stats = []
    
    for bh in boreholes:
        layers = bh.get('th_layers', [])
        for layer in layers:
            # Count symbols
            symbol = layer.get('layer_symbol', 'Unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # Count colors
            color = layer.get('layer_forecolor', '#808080')
            color_counts[color] = color_counts.get(color, 0) + 1
            
            # Track depths
            thickness = layer.get('layer_to', 0) - layer.get('layer_from', 0)
            depth_stats.append(thickness)
    
    print("\nSoil Symbol Distribution:")
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {count} occurrences")
    
    print(f"\nUnique colors: {len(color_counts)}")
    print(f"Layer thickness stats:")
    print(f"  Min: {min(depth_stats):.1f} ft")
    print(f"  Max: {max(depth_stats):.1f} ft")
    print(f"  Mean: {np.mean(depth_stats):.1f} ft")
    print(f"  Median: {np.median(depth_stats):.1f} ft")
    
    return {
        'symbol_counts': symbol_counts,
        'color_counts': color_counts,
        'depth_stats': depth_stats
    }


def test_intelligent_matching(data):
    """Test the intelligent layer matching on the dataset"""
    print("\n" + "=" * 60)
    print("INTELLIGENT LAYER MATCHING TEST")
    print("=" * 60)
    
    # Initialize matcher
    matcher = IntelligentLayerMatcher()
    
    # Train on data
    matcher.train(data)
    
    # Test on a specific borehole pair
    boreholes = data.get('boreholesData', [])
    bh_lookup = {bh.get('th_title', ''): bh for bh in boreholes}
    
    # Test pairs
    test_pairs = [
        ('B-55', 'B-56'),
        ('B-57', 'B-58'),
        ('B-54', 'B-54B'),
    ]
    
    print("\n" + "-" * 40)
    print("Testing layer matching on borehole pairs:")
    print("-" * 40)
    
    for bh1_name, bh2_name in test_pairs:
        bh1 = bh_lookup.get(bh1_name)
        bh2 = bh_lookup.get(bh2_name)
        
        if not bh1 or not bh2:
            print(f"\n{bh1_name} ↔ {bh2_name}: Boreholes not found")
            continue
        
        layers1 = bh1.get('th_layers', [])
        layers2 = bh2.get('th_layers', [])
        
        print(f"\n{bh1_name} ({len(layers1)} layers) ↔ {bh2_name} ({len(layers2)} layers):")
        
        matches = matcher.find_best_matches(layers1, layers2, threshold=0.3)
        
        for i, j, prob in matches:
            sym1 = layers1[i].get('layer_symbol', '?')
            sym2 = layers2[j].get('layer_symbol', '?')
            d1 = f"{layers1[i].get('layer_from', 0):.0f}-{layers1[i].get('layer_to', 0):.0f}ft"
            d2 = f"{layers2[j].get('layer_from', 0):.0f}-{layers2[j].get('layer_to', 0):.0f}ft"
            
            match_type = "✓ SAME" if sym1 == sym2 else "~ DIFF"
            print(f"  {match_type} | {sym1} ({d1}) ↔ {sym2} ({d2}) | Confidence: {prob:.1%}")
    
    # Save model
    model_path = Path(__file__).parent / "layer_matching_model.pkl"
    matcher.save_model(model_path)
    
    return matcher


def main():
    """Main function to demonstrate intelligent layer matching"""
    base_path = Path(__file__).parent
    
    # Load both datasets
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    complex_file = base_path / "Complex RSLog section (1) 1 (1).json"
    
    print("=" * 60)
    print("INTELLIGENT LAYER MATCHING - ML Training & Testing")
    print("=" * 60)
    
    # Load easier dataset first
    print(f"\nLoading: {easier_file.name}")
    with open(easier_file, 'r') as f:
        easier_data = json.load(f)
    
    # Analyze distributions
    analyze_layer_distributions(easier_data)
    
    # Train and test on easier dataset
    matcher = test_intelligent_matching(easier_data)
    
    # Now test on complex dataset
    print("\n\n" + "=" * 60)
    print("TESTING ON COMPLEX DATASET")
    print("=" * 60)
    
    print(f"\nLoading: {complex_file.name}")
    with open(complex_file, 'r') as f:
        complex_data = json.load(f)
    
    analyze_layer_distributions(complex_data)
    
    # Test the trained model on complex data
    boreholes = complex_data.get('boreholesData', [])
    if len(boreholes) >= 2:
        bh1 = boreholes[0]
        bh2 = boreholes[1]
        
        layers1 = bh1.get('th_layers', [])
        layers2 = bh2.get('th_layers', [])
        
        bh1_name = bh1.get('th_title', 'BH1')
        bh2_name = bh2.get('th_title', 'BH2')
        
        print(f"\nCross-dataset test: {bh1_name} ↔ {bh2_name}")
        matches = matcher.find_best_matches(layers1, layers2, threshold=0.3)
        
        for i, j, prob in matches[:10]:  # Show first 10
            sym1 = layers1[i].get('layer_symbol', '?')
            sym2 = layers2[j].get('layer_symbol', '?')
            print(f"  {sym1} ↔ {sym2} | Confidence: {prob:.1%}")
    
    print("\n" + "=" * 60)
    print("✓ Intelligent Layer Matching Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

