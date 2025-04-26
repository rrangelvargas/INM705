import cv2  # import opencv for video processing
import mediapipe as mp  # import mediapipe for landmark detection
import numpy as np  # import numpy for numerical operations
import json  # import json for loading annotations
import pickle  # import pickle for saving and loading preprocessed data
from pathlib import Path  # import path for file handling
from collections import Counter  # import counter to track per-class sample counts
from tqdm import tqdm  # import tqdm for progress bars
import torch  # import torch for model and tensor operations
from torch.utils.data import Dataset  # import dataset tool from torch

class SignLanguageDataset(Dataset):  # define custom dataset class
    def __init__(self, json_path, videos_dir, sequence_length=30, cache_dir='data/cached_landmarks', max_words=None, augment=True):  # initialize dataset
        self.videos_dir = Path(videos_dir)
        self.sequence_length = sequence_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print("loading annotations...")
        with open(json_path) as f:
            self.annotations = json.load(f)

        word_counts = {}
        word_instances = {}
        for entry in tqdm(self.annotations, desc="processing annotations"):
            word = entry['gloss'].lower()
            # commented out video verification
            instances = [i for i in entry['instances'] if (self.videos_dir / f"{i['video_id']}.mp4").exists()]
            # instances = entry['instances']
            if instances:
                word_counts[word] = len(instances)
                word_instances[word] = instances

        # Sort words by number of instances and take top max_words if specified
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        if max_words is not None:
            sorted_words = sorted_words[:max_words]
        all_words = [word for word, _ in sorted_words]
        
        print(f"total words: {len(all_words)}")
        if max_words is not None:
            print(f"using top {max_words} most frequent words")

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word in all_words:
            self.word2idx[word] = len(self.word2idx)

        self.samples = []
        word_sample_counts = Counter()

        print("\npreparing samples...")
        for word in tqdm(all_words, desc="processing words"):
            instances = word_instances[word]
            num_augment = max(1, (100 - len(instances)) // len(instances) + 1)
            for instance in instances:
                vid = instance['video_id']
                path = self.videos_dir / f"{vid}.mp4"
                if path.exists():
                    self.samples.append((path, self.word2idx[word], vid))
                    word_sample_counts[word] += 1
                    if augment:
                        for i in range(num_augment):
                            if word_sample_counts[word] >= 100:
                                break
                            aug_vid = f"{vid}_aug_{i}"
                            self.samples.append((path, self.word2idx[word], aug_vid))
                            word_sample_counts[word] += 1

        self._preprocess_all()

    def _extract_landmarks(self, landmarks, count):
        if landmarks:
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return [[0, 0, 0]] * count

    def _preprocess_single(self, sample):
        path, _, vid = sample
        cache_file = self.cache_dir / f"{vid}.pkl"
        if cache_file.exists():
            return

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                frame = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()

        landmarks_seq = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            data = []
            data.extend(self._extract_landmarks(results.pose_landmarks, 33))
            data.extend(self._extract_landmarks(results.left_hand_landmarks, 21))
            data.extend(self._extract_landmarks(results.right_hand_landmarks, 21))
            landmarks_seq.append(np.array(data, dtype=np.float32))

        holistic.close()
        array = np.stack(landmarks_seq, axis=0)

        if "_aug_" in vid:
            angle = np.random.uniform(-30, 30) * np.pi / 180
            cos, sin = np.cos(angle), np.sin(angle)
            x, z = array[..., 0], array[..., 2]
            array[..., 0] = x * cos - z * sin
            array[..., 2] = x * sin + z * cos
            array *= np.random.uniform(0.8, 1.2)
            if np.random.random() < 0.5:
                ml = int(array.shape[0] * 0.2)
                si = np.random.randint(0, array.shape[0] - ml)
                array[si:si+ml] *= 0

        with open(cache_file, 'wb') as f:
            pickle.dump(array, f)

    def _preprocess_all(self):
        print("\npreprocessing samples...")
        for s in tqdm(self.samples, desc="extracting landmarks"):
            self._preprocess_single(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, vid = self.samples[idx]
        with open(self.cache_dir / f"{vid}.pkl", 'rb') as f:
            array = pickle.load(f)
        mean, std = np.mean(array), np.std(array)
        array = (array - mean) / (std + 1e-7)
        return torch.FloatTensor(array), torch.LongTensor([label])
