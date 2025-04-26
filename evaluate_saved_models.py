import os
import torch
import numpy as np
import pickle
import json
import re
from pathlib import Path
from tqdm import tqdm
from models.model_baseline_loss import SignLanguageModel as BaselineLossModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {device}")

# === CONFIG ===
MODELS_DIR = "models"
TEST_DIR = "data/test"
JSON_PATH = "data/WLASL_v0.3.json"

def parse_video_id(filename):
    """
    Parse the video ID from a filename, handling augmented files.
    Example: '35305_aug_7.pkl' -> '35305'
    """
    # Remove .pkl extension
    base_name = Path(filename).stem
    # Remove _aug_N suffix if present (where N is a number)
    video_id = re.sub(r'_aug_\d+$', '', base_name)
    return video_id

def load_word_mapping(json_path):
    """
    Load word to index mapping from JSON file, matching evaluate_saved_models.py.
    """
    with open(json_path) as f:
        annotations = json.load(f)
    
    # Create word to index mapping
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    # First, count instances per word
    word_counts = {}
    for entry in annotations:
        word = entry['gloss'].lower()
        instances = [i for i in entry['instances'] if Path(f"data/videos/{i['video_id']}.mp4").exists()]
        if instances:
            word_counts[word] = len(instances)
    
    # Sort words by number of instances
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create mapping for sorted words
    for word, _ in sorted_words:
        word2idx[word] = len(word2idx)
    
    # Create reverse mapping (index to word)
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Created mapping with {len(word2idx)} words")
    return word2idx, idx2word

def create_video_to_word_mapping(json_path):
    """
    Create a mapping from video IDs to their corresponding words.
    """
    with open(json_path) as f:
        annotations = json.load(f)
    
    video_to_word = {}
    for entry in annotations:
        word = entry['gloss'].lower()
        for instance in entry['instances']:
            video_to_word[instance['video_id']] = word
    
    return video_to_word

def load_test_landmarks(test_dir):
    """
    Load all landmark files from the test directory.
    The filenames should be the video IDs (possibly with augmentation suffixes).
    """
    test_dir = Path(test_dir)
    landmark_files = list(test_dir.glob("*.pkl"))
    
    if not landmark_files:
        raise FileNotFoundError(f"No landmark files found in {test_dir}")
    
    print(f"Found {len(landmark_files)} landmark files")
    
    # Load all landmark files
    landmarks = []
    video_ids = []
    for file in tqdm(landmark_files, desc="Loading landmarks"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            landmarks.append(data)
            # Parse the video ID from the filename
            video_id = parse_video_id(file.name)
            video_ids.append(video_id)
    
    return landmarks, video_ids

def predict_landmarks(model, landmarks):
    """
    Predict labels for multiple landmark sequences.
    """
    model.eval()
    predictions = []
    
    for landmark_data in tqdm(landmarks, desc="Making predictions"):
        # Convert to tensor and normalize
        x = torch.FloatTensor(landmark_data).unsqueeze(0).to(device)  # Add batch dimension
        mean, std = torch.mean(x), torch.std(x)
        x = (x - mean) / (std + 1e-7)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(x)
            pred = outputs.argmax(1).item()
            predictions.append(pred)
    
    return predictions

if __name__ == "__main__":
    # Load word mappings
    print("[data] Loading word mappings...")
    word2idx, idx2word = load_word_mapping(JSON_PATH)
    
    # Create video ID to word mapping
    print("[data] Creating video ID to word mapping...")
    video_to_word = create_video_to_word_mapping(JSON_PATH)
    
    # Load test landmarks
    print("[data] Loading test landmarks...")
    landmarks, video_ids = load_test_landmarks(TEST_DIR)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, "baseline_loss_ls_best.pth")
    print(f"\n[model] Loading model from {model_path}")
    
    model = BaselineLossModel(num_classes=len(word2idx)).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle class size mismatch
    if state_dict['fc.weight'].shape[0] != len(word2idx):
        print(f"Warning: Model was trained with {state_dict['fc.weight'].shape[0]} classes, but current dataset has {len(word2idx)} classes")
        print("Creating new output layer with correct number of classes")
        
        # Create new output layer with correct size
        new_fc_weight = torch.nn.init.xavier_uniform_(torch.zeros(len(word2idx), state_dict['fc.weight'].shape[1]))
        new_fc_bias = torch.zeros(len(word2idx))
        
        # Replace the old output layer weights
        state_dict['fc.weight'] = new_fc_weight
        state_dict['fc.bias'] = new_fc_bias
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Make predictions
    print("\nMaking predictions...")
    predicted_classes = predict_landmarks(model, landmarks)
    
    # Print results
    print(f"\nResults:")
    print("-" * 90)
    print(f"{'Video ID':<15} {'True Word':<15} {'Predicted Class':<15} {'Predicted Word':<15} {'Correct':<10}")
    print("-" * 90)
    
    correct = 0
    for video_id, pred_class in zip(video_ids, predicted_classes):
        true_word = video_to_word.get(video_id, "Unknown")
        predicted_word = idx2word.get(pred_class, f"Unknown class {pred_class}")
        is_correct = true_word.lower() == predicted_word.lower()
        if is_correct:
            correct += 1
        
        print(f"{video_id:<15} {true_word:<15} {pred_class:<15} {predicted_word:<15} {'✓' if is_correct else '✗':<10}")
    
    # Print summary
    print("-" * 90)
    accuracy = 100 * correct / len(video_ids)
    print(f"\nSummary:")
    print(f"Total files: {len(video_ids)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%") 