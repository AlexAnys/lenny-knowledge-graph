#!/usr/bin/env python3
"""
Lenny's Podcast Knowledge Graph Analyzer

Usage:
    python analyze.py --input transcripts/ --output graph_data.json

Requirements:
    pip install numpy scikit-learn
"""

import os
import json
import math
import argparse
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Using simple clustering.")


# Core concepts to extract
CONCEPTS = [
    'product', 'growth', 'leadership', 'strategy', 'customer',
    'innovation', 'career', 'decision', 'execution', 'founder',
    'culture', 'team', 'data', 'design', 'marketing',
    'engineering', 'startup', 'scale', 'hiring', 'feedback'
]

# Cluster configuration
CLUSTER_NAMES = {
    0: '领导力与愿景',
    1: '增长与获客',
    2: '产品管理',
    3: '决策与心智',
    4: '创新与技术',
    5: '创业与融资',
    6: '职业发展',
    7: '用户研究'
}

CLUSTER_COLORS = {
    0: '#8B5CF6',
    1: '#10B981',
    2: '#F59E0B',
    3: '#EF4444',
    4: '#3B82F6',
    5: '#EC4899',
    6: '#14B8A6',
    7: '#F97316'
}


def load_transcripts(input_dir):
    """Load all transcript files from directory."""
    transcripts = {}
    for fname in os.listdir(input_dir):
        if fname.endswith('.txt'):
            guest = fname[:-4].strip()
            filepath = os.path.join(input_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    transcripts[guest] = f.read().lower()
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    return transcripts


def extract_concepts(text, concepts=CONCEPTS):
    """Extract concept frequencies from text."""
    scores = {}
    for concept in concepts:
        count = text.count(concept)
        if count > 0:
            scores[concept] = min(count / 50, 15)  # Normalize
    return scores


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0
    
    all_keys = set(v1.keys()) | set(v2.keys())
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in all_keys)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)


def assign_cluster(concepts, concept_to_cluster):
    """Assign cluster based on dominant concept."""
    if not concepts:
        return 4  # Default to innovation
    
    top_concept = max(concepts, key=concepts.get)
    return concept_to_cluster.get(top_concept, 4)


def build_graph(transcripts, similarity_threshold=0.8, max_edges=2000, n_clusters=8):
    """Build knowledge graph from transcripts."""
    
    # Concept to cluster mapping
    concept_to_cluster = {
        'leadership': 0, 'strategy': 0,
        'growth': 1, 'marketing': 1,
        'product': 2, 'execution': 2, 'design': 2,
        'decision': 3, 'culture': 3,
        'innovation': 4, 'engineering': 4, 'data': 4,
        'founder': 5, 'startup': 5, 'scale': 5,
        'career': 6, 'hiring': 6, 'team': 6,
        'customer': 7, 'feedback': 7
    }
    
    # Build nodes
    nodes = []
    for guest, text in transcripts.items():
        concepts = extract_concepts(text)
        cluster = assign_cluster(concepts, concept_to_cluster)
        richness = sum(concepts.values()) if concepts else 1
        
        nodes.append({
            'id': guest,
            'cluster': cluster,
            'concepts': concepts,
            'richness': richness
        })
    
    print(f"Built {len(nodes)} nodes")
    
    # Build edges
    edges = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            
            sim = cosine_similarity(n1['concepts'], n2['concepts'])
            if sim > similarity_threshold:
                edges.append({
                    'source': n1['id'],
                    'target': n2['id'],
                    'weight': round(sim, 3)
                })
    
    # Sort and limit edges
    edges = sorted(edges, key=lambda e: -e['weight'])[:max_edges]
    print(f"Built {len(edges)} edges")
    
    # Cluster statistics
    cluster_counts = defaultdict(int)
    for n in nodes:
        cluster_counts[n['cluster']] += 1
    print(f"Cluster distribution: {dict(cluster_counts)}")
    
    return {'nodes': nodes, 'edges': edges}


def main():
    parser = argparse.ArgumentParser(description='Analyze podcast transcripts')
    parser.add_argument('--input', '-i', required=True, help='Input directory with .txt files')
    parser.add_argument('--output', '-o', default='graph_data.json', help='Output JSON file')
    parser.add_argument('--threshold', '-t', type=float, default=0.8, help='Similarity threshold')
    parser.add_argument('--max-edges', '-e', type=int, default=2000, help='Maximum edges')
    parser.add_argument('--clusters', '-c', type=int, default=8, help='Number of clusters')
    
    args = parser.parse_args()
    
    print(f"Loading transcripts from {args.input}...")
    transcripts = load_transcripts(args.input)
    print(f"Loaded {len(transcripts)} transcripts")
    
    print("Building graph...")
    graph = build_graph(
        transcripts,
        similarity_threshold=args.threshold,
        max_edges=args.max_edges,
        n_clusters=args.clusters
    )
    
    print(f"Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    main()
