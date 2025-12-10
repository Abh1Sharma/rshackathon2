#!/usr/bin/env python3
"""
Intelligent Layer Matching using Machine Learning + OpenAI Embeddings

This module uses ML to match soil layers between boreholes based on:
- Soil symbol (AASHTO classification: A-1 to A-7)
- Color (foreground and background hex colors)
- Description (text similarity via OpenAI embeddings)
- Depth patterns
- Geological context (depth relative to water table)

The model learns to predict similarity scores between layers,
enabling more accurate geological cross-section generation.

Features:
- OpenAI text-embedding-3-small for semantic description matching
- Geological context: above/below water table, depth ratios
- Enhanced AASHTO classification with soil properties
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# OpenAI API for text embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")

# OpenAI API Key - set via environment variable or .env file
# To use: export OPENAI_API_KEY="your-api-key-here"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class OpenAIEmbedder:
    """Generate text embeddings using OpenAI API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        self.embedding_cache = {}  # Cache to avoid redundant API calls
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            print("✓ OpenAI embeddings enabled")
        else:
            print("⚠ OpenAI embeddings disabled (using fallback TF-IDF)")
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """Get embedding for a text string"""
        if not text or not self.client:
            return np.zeros(self.embedding_dim)
        
        # Check cache
        cache_key = f"{model}:{text[:100]}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"  Warning: OpenAI embedding failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def get_embeddings_batch(self, texts, model="text-embedding-3-small"):
        """Get embeddings for multiple texts efficiently"""
        if not self.client:
            return [np.zeros(self.embedding_dim) for _ in texts]
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                cache_key = f"{model}:{text[:100]}"
                if cache_key not in self.embedding_cache:
                    valid_texts.append(text)
                    valid_indices.append(i)
        
        # Get embeddings for uncached texts
        if valid_texts:
            try:
                response = self.client.embeddings.create(
                    input=valid_texts,
                    model=model
                )
                for j, emb_data in enumerate(response.data):
                    text = valid_texts[j]
                    cache_key = f"{model}:{text[:100]}"
                    self.embedding_cache[cache_key] = np.array(emb_data.embedding)
            except Exception as e:
                print(f"  Warning: OpenAI batch embedding failed: {e}")
        
        # Collect results
        results = []
        for text in texts:
            if text and text.strip():
                cache_key = f"{model}:{text[:100]}"
                results.append(self.embedding_cache.get(cache_key, np.zeros(self.embedding_dim)))
            else:
                results.append(np.zeros(self.embedding_dim))
        
        return results


class LayerFeatureExtractor:
    """Extract features from soil layer data for ML matching"""
    
    def __init__(self, use_openai=True):
        self.symbol_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.use_openai = use_openai
        
        # Initialize OpenAI embedder
        if use_openai and OPENAI_AVAILABLE:
            self.embedder = OpenAIEmbedder()
        else:
            self.embedder = None
        
        # AASHTO soil classification categories with enhanced properties
        self.aashto_categories = {
            'A-1': {'type': 'gravel', 'quality': 'excellent', 'code': 1, 
                    'permeability': 'high', 'compressibility': 'low', 'strength': 'high'},
            'A-2': {'type': 'sand_gravel', 'quality': 'excellent', 'code': 2,
                    'permeability': 'high', 'compressibility': 'low', 'strength': 'high'},
            'A-3': {'type': 'fine_sand', 'quality': 'excellent', 'code': 3,
                    'permeability': 'medium', 'compressibility': 'low', 'strength': 'medium'},
            'A-4': {'type': 'silt', 'quality': 'fair', 'code': 4,
                    'permeability': 'low', 'compressibility': 'medium', 'strength': 'medium'},
            'A-5': {'type': 'silt', 'quality': 'poor', 'code': 5,
                    'permeability': 'low', 'compressibility': 'high', 'strength': 'low'},
            'A-6': {'type': 'clay', 'quality': 'poor', 'code': 6,
                    'permeability': 'very_low', 'compressibility': 'high', 'strength': 'medium'},
            'A-7': {'type': 'clay', 'quality': 'poor', 'code': 7,
                    'permeability': 'very_low', 'compressibility': 'very_high', 'strength': 'low'},
        }
        
        # Permeability encoding
        self.permeability_map = {'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.8}
        self.compressibility_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
        self.strength_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
    
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
        """Extract AASHTO-based features from soil symbol (12 features)"""
        if not symbol:
            return [0] * 12
        
        # Parse symbol (e.g., "A-2", "A-7-5")
        parts = symbol.split('-')
        base_symbol = parts[0] + '-' + parts[1] if len(parts) >= 2 else symbol
        
        if base_symbol in self.aashto_categories:
            cat = self.aashto_categories[base_symbol]
            return [
                cat['code'] / 7.0,  # Normalized code
                1 if cat['type'] == 'gravel' else 0,
                1 if cat['type'] == 'sand_gravel' else 0,
                1 if cat['type'] == 'fine_sand' else 0,
                1 if cat['type'] == 'silt' else 0,
                1 if cat['type'] == 'clay' else 0,
                1 if cat['quality'] == 'excellent' else 0,
                1 if cat['quality'] == 'fair' else 0,
                1 if cat['quality'] == 'poor' else 0,
                # Enhanced geological properties
                self.permeability_map.get(cat.get('permeability', 'medium'), 0.5),
                self.compressibility_map.get(cat.get('compressibility', 'medium'), 0.5),
                self.strength_map.get(cat.get('strength', 'medium'), 0.5),
            ]
        return [0] * 12
    
    def get_geological_context(self, layer, water_depth=None, total_depth=None):
        """
        Extract geological context features (5 features)
        
        Args:
            layer: Layer data dict
            water_depth: Depth to water table (if known)
            total_depth: Total borehole depth
        """
        layer_from = layer.get('layer_from', 0)
        layer_to = layer.get('layer_to', 0)
        layer_mid = (layer_from + layer_to) / 2
        thickness = layer_to - layer_from
        
        features = []
        
        # 1. Relative depth (normalized by typical max depth)
        features.append(layer_mid / 200.0)
        
        # 2. Position relative to water table (if known)
        if water_depth is not None and water_depth > 0:
            # Positive = above water table, negative = below
            relative_to_water = (water_depth - layer_mid) / water_depth
            features.append(relative_to_water)
            # Binary: is layer above water table?
            features.append(1.0 if layer_mid < water_depth else 0.0)
        else:
            features.append(0.0)
            features.append(0.5)  # Unknown
        
        # 3. Layer thickness ratio (thin vs thick layers)
        features.append(min(thickness / 50.0, 1.0))
        
        # 4. Depth ratio within borehole
        if total_depth and total_depth > 0:
            features.append(layer_mid / total_depth)
        else:
            features.append(layer_mid / 200.0)
        
        return features
    
    def extract_layer_features(self, layer, water_depth=None, total_depth=None):
        """
        Extract comprehensive features from a single layer (23 base features)
        
        Features include:
        - AASHTO classification (12 features)
        - Color encoding (6 features)
        - Geological context (5 features)
        """
        features = []
        
        # 1. Symbol-based features (12 features)
        symbol = layer.get('layer_symbol', 'Unknown')
        features.extend(self.get_aashto_features(symbol))
        
        # 2. Color features (6 features: RGB for foreground and background)
        fg_color = layer.get('layer_forecolor', '#808080')
        bg_color = layer.get('layer_backcolor', '#FFFFFF')
        features.extend(self.hex_to_rgb_normalized(fg_color))
        features.extend(self.hex_to_rgb_normalized(bg_color))
        
        # 3. Geological context (5 features)
        features.extend(self.get_geological_context(layer, water_depth, total_depth))
        
        return np.array(features)
    
    def get_description_embedding(self, description):
        """Get semantic embedding for layer description using OpenAI"""
        if self.embedder and description:
            return self.embedder.get_embedding(description)
        return np.zeros(1536)  # Default embedding size
    
    def compute_description_similarity(self, desc1, desc2):
        """Compute semantic similarity between layer descriptions using OpenAI embeddings"""
        if self.embedder and desc1 and desc2:
            emb1 = self.embedder.get_embedding(desc1)
            emb2 = self.embedder.get_embedding(desc2)
            
            # Cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                return np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Fallback to simple word overlap
        if desc1 and desc2:
            words1 = set(desc1.lower().split())
            words2 = set(desc2.lower().split())
            if words1 and words2:
                return len(words1 & words2) / len(words1 | words2)
        
        return 0.0
    
    def extract_pair_features(self, layer1, layer2, water_depth1=None, water_depth2=None,
                               total_depth1=None, total_depth2=None):
        """
        Extract features for a pair of layers (for similarity prediction)
        
        Returns feature vector including:
        - Individual layer features (23 each)
        - Difference features
        - Product features (interaction terms)
        - Semantic description similarity (1 feature)
        
        Total: 23*4 + 1 = 93 features
        """
        feat1 = self.extract_layer_features(layer1, water_depth1, total_depth1)
        feat2 = self.extract_layer_features(layer2, water_depth2, total_depth2)
        
        # Combine features: individual + difference + product
        pair_features = np.concatenate([
            feat1,
            feat2,
            np.abs(feat1 - feat2),  # Absolute difference
            feat1 * feat2,  # Element-wise product (interaction)
        ])
        
        # Add semantic description similarity (using OpenAI embeddings if available)
        desc1 = layer1.get('layer_descr', '') or layer1.get('layer_title', '')
        desc2 = layer2.get('layer_descr', '') or layer2.get('layer_title', '')
        desc_similarity = self.compute_description_similarity(desc1, desc2)
        
        pair_features = np.concatenate([pair_features, [desc_similarity]])
        
        return pair_features
    
    def compute_text_similarity(self, desc1, desc2):
        """Compute text similarity between layer descriptions (fallback TF-IDF)"""
        if not desc1 or not desc2:
            return 0.0
        
        try:
            tfidf_matrix = self.tfidf.fit_transform([desc1, desc2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0


class IntelligentLayerMatcher:
    """ML-based layer matching between boreholes with OpenAI embeddings"""
    
    def __init__(self, use_openai=True):
        self.feature_extractor = LayerFeatureExtractor(use_openai=use_openai)
        # Use GradientBoosting for better handling of feature interactions
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            min_samples_split=5
        )
        self.is_trained = False
        self.training_data = []
        self.borehole_context = {}  # Cache for borehole context (water depth, total depth)
    
    def _get_borehole_context(self, bh):
        """Extract geological context from borehole"""
        bh_id = bh.get('th_id', '') or bh.get('th_title', '')
        
        if bh_id in self.borehole_context:
            return self.borehole_context[bh_id]
        
        water_depth = bh.get('th_ground_water_depth', None)
        total_depth = bh.get('th_depth', None)
        
        # Try to parse water depth
        if water_depth:
            try:
                water_depth = float(water_depth)
            except:
                water_depth = None
        
        # Try to parse total depth
        if total_depth:
            try:
                total_depth = float(total_depth)
            except:
                total_depth = None
        
        context = {'water_depth': water_depth, 'total_depth': total_depth}
        self.borehole_context[bh_id] = context
        return context
    
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
                    
                    # Get borehole context
                    ctx1 = self._get_borehole_context(bh1)
                    ctx2 = self._get_borehole_context(bh2)
                    
                    # Positive pairs: same symbol and both in connected set
                    for l1 in layers1:
                        for l2 in layers2:
                            sym1 = l1.get('layer_symbol', '')
                            sym2 = l2.get('layer_symbol', '')
                            
                            # Include borehole context in training data
                            if sym1 == sym2 and sym1 in connected_symbols:
                                positive_pairs.append((l1, l2, 1, ctx1, ctx2))
                            elif sym1 != sym2:
                                negative_pairs.append((l1, l2, 0, ctx1, ctx2))
        
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
        """Train the layer matching model with enhanced features"""
        if training_pairs is None and data is not None:
            training_pairs = self.generate_training_data(data)
        
        if not training_pairs:
            print("No training data available!")
            return False
        
        print("\nTraining layer matching model with enhanced features...")
        print("  - OpenAI embeddings for descriptions")
        print("  - Geological context (water table, depth)")
        print("  - Enhanced AASHTO properties")
        
        # Extract features
        X = []
        y = []
        
        # Handle both old format (l1, l2, label) and new format (l1, l2, label, ctx1, ctx2)
        for item in training_pairs:
            if len(item) == 5:
                l1, l2, label, ctx1, ctx2 = item
                features = self.feature_extractor.extract_pair_features(
                    l1, l2,
                    water_depth1=ctx1.get('water_depth'),
                    water_depth2=ctx2.get('water_depth'),
                    total_depth1=ctx1.get('total_depth'),
                    total_depth2=ctx2.get('total_depth')
                )
            else:
                l1, l2, label = item[:3]
                features = self.feature_extractor.extract_pair_features(l1, l2)
            
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  Feature vector size: {X.shape[1]}")
        
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
        
        print(f"\n  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Test accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Different', 'Match']))
        
        # Show feature importance (top 10)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
        
        return True
    
    def predict_match(self, layer1, layer2, ctx1=None, ctx2=None):
        """
        Predict if two layers should be matched.
        
        Args:
            layer1, layer2: Layer dictionaries
            ctx1, ctx2: Optional borehole context dicts with 'water_depth' and 'total_depth'
        
        Returns:
            float: Probability of match (0-1)
        """
        if not self.is_trained:
            # Fallback to simple symbol matching + description similarity
            sym_match = 1.0 if layer1.get('layer_symbol') == layer2.get('layer_symbol') else 0.0
            desc_sim = self.feature_extractor.compute_description_similarity(
                layer1.get('layer_descr', ''),
                layer2.get('layer_descr', '')
            )
            return 0.7 * sym_match + 0.3 * desc_sim
        
        features = self.feature_extractor.extract_pair_features(
            layer1, layer2,
            water_depth1=ctx1.get('water_depth') if ctx1 else None,
            water_depth2=ctx2.get('water_depth') if ctx2 else None,
            total_depth1=ctx1.get('total_depth') if ctx1 else None,
            total_depth2=ctx2.get('total_depth') if ctx2 else None
        )
        proba = self.model.predict_proba([features])[0]
        
        # Return probability of match (class 1)
        return proba[1] if len(proba) > 1 else proba[0]
    
    def find_best_matches(self, layers1, layers2, bh1=None, bh2=None, threshold=0.5):
        """
        Find the best matching layers between two boreholes.
        
        Args:
            layers1: List of layers from borehole 1
            layers2: List of layers from borehole 2
            bh1, bh2: Optional borehole dicts for geological context
            threshold: Minimum probability to consider a match
        
        Returns:
            List of (layer1_idx, layer2_idx, probability) tuples
        """
        matches = []
        used_l2 = set()
        
        # Get borehole context
        ctx1 = self._get_borehole_context(bh1) if bh1 else {}
        ctx2 = self._get_borehole_context(bh2) if bh2 else {}
        
        # Score all pairs
        scores = []
        for i, l1 in enumerate(layers1):
            for j, l2 in enumerate(layers2):
                prob = self.predict_match(l1, l2, ctx1, ctx2)
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
        """Save trained model to file (excluding non-picklable OpenAI client)"""
        # Temporarily remove OpenAI client for pickling
        embedder = self.feature_extractor.embedder
        embedding_cache = embedder.embedding_cache if embedder else {}
        
        # Set embedder to None for pickling
        self.feature_extractor.embedder = None
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'is_trained': self.is_trained,
                'embedding_cache': embedding_cache,  # Save cache separately
                'borehole_context': self.borehole_context
            }, f)
        
        # Restore embedder
        self.feature_extractor.embedder = embedder
        
        print(f"✓ Model saved to: {filepath}")
        print(f"  - Embedding cache: {len(embedding_cache)} entries")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_extractor = data['feature_extractor']
            self.is_trained = data['is_trained']
            self.borehole_context = data.get('borehole_context', {})
            
            # Restore embedder with cached embeddings
            if self.feature_extractor.use_openai and OPENAI_AVAILABLE:
                self.feature_extractor.embedder = OpenAIEmbedder()
                if 'embedding_cache' in data:
                    self.feature_extractor.embedder.embedding_cache = data['embedding_cache']
        
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

