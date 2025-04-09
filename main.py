import cv2  # import opencv for video processing
import mediapipe as mp  # import mediapipe for landmark detection
import numpy as np  # import numpy for numerical operations
import json  # import json for loading annotations
import torch  # import torch for model and tensor operations
import torch.nn as nn  # import neural network modules from torch
from torch.utils.data import Dataset, DataLoader  # import dataset and dataloader tools
from pathlib import Path  # import path for file handling
import pickle  # import pickle for saving and loading preprocessed data
from collections import Counter  # import counter to track per-class sample counts
from tqdm import tqdm  # import tqdm for progress bars
import matplotlib.pyplot as plt  # import matplotlib for plotting

class SignLanguageDataset(Dataset):  # define custom dataset class
    def __init__(self, json_path, videos_dir, sequence_length=30, cache_dir='cached_landmarks'):  # initialize dataset
        self.videos_dir = Path(videos_dir)  # set path to videos
        self.sequence_length = sequence_length  # set desired frame sequence length
        self.cache_dir = Path(cache_dir)  # set path to cached landmark data
        self.cache_dir.mkdir(exist_ok=True)  # create cache directory if it doesn't exist

        print("loading annotations...")  # log annotation loading
        with open(json_path) as f:  # open annotation json file
            self.annotations = json.load(f)  # load annotations from file

        word_counts = {}  # initialize word count dictionary
        word_instances = {}  # initialize word instance dictionary
        for entry in tqdm(self.annotations, desc="processing annotations"):  # loop through annotations
            word = entry['gloss'].lower()  # get word in lowercase
            instances = [i for i in entry['instances'] if (self.videos_dir / f"{i['video_id']}.mp4").exists()]  # keep only valid videos
            if instances:  # if valid instances found
                word_counts[word] = len(instances)  # store count
                word_instances[word] = instances  # store instances

        all_words = sorted(word_counts.keys())  # get all words
        print(f"total words: {len(all_words)}")  # log total words

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # initialize word-to-index map
        for word in all_words:  # assign index for each word
            self.word2idx[word] = len(self.word2idx)

        self.samples = []  # list of all (path, label, vid)
        word_sample_counts = Counter()  # track sample count per word

        print("\npreparing samples...")  # log sample preparation
        for word in tqdm(all_words, desc="processing words"):  # loop through each word
            instances = word_instances[word]  # get instances for word
            num_augment = max(1, (100 - len(instances)) // len(instances) + 1)  # determine how many augmentations needed
            for instance in instances:  # loop through instances
                vid = instance['video_id']  # get video id
                path = self.videos_dir / f"{vid}.mp4"  # get video path
                if path.exists():  # check video file exists
                    self.samples.append((path, self.word2idx[word], vid))  # add original sample
                    word_sample_counts[word] += 1  # increment sample count
                    for i in range(num_augment):  # generate augmentations
                        if word_sample_counts[word] >= 100:  # stop if 100 reached
                            break
                        aug_vid = f"{vid}_aug_{i}"  # generate augmented video id
                        self.samples.append((path, self.word2idx[word], aug_vid))  # add augmented sample
                        word_sample_counts[word] += 1  # increment count

        self._preprocess_all()  # preprocess all samples

    def _extract_landmarks(self, landmarks, count):  # extract or pad landmarks
        if landmarks:  # if landmarks exist
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]  # return landmark coords
        return [[0, 0, 0]] * count  # return zero-filled landmarks

    def _preprocess_single(self, sample):  # preprocess a single video sample
        path, _, vid = sample  # unpack sample
        cache_file = self.cache_dir / f"{vid}.pkl"  # set cache file path
        if cache_file.exists():  # skip if already cached
            return

        mp_holistic = mp.solutions.holistic  # get holistic model
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # init detector

        cap = cv2.VideoCapture(str(path))  # open video file
        frames = []  # store video frames
        while len(frames) < self.sequence_length:  # collect enough frames
            ret, frame = cap.read()  # read frame
            if not ret:  # if read fails
                frame = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)  # use last or blank frame
            frames.append(frame)  # add frame to list
        cap.release()  # close video

        landmarks_seq = []  # store all landmarks
        for frame in frames:  # process each frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
            results = holistic.process(rgb)  # run holistic model
            data = []  # collect landmarks
            data.extend(self._extract_landmarks(results.pose_landmarks, 33))  # add pose landmarks
            data.extend(self._extract_landmarks(results.left_hand_landmarks, 21))  # add left hand
            data.extend(self._extract_landmarks(results.right_hand_landmarks, 21))  # add right hand
            landmarks_seq.append(np.array(data, dtype=np.float32))  # save frame's landmarks

        holistic.close()  # close detector
        array = np.stack(landmarks_seq, axis=0)  # stack all frames

        if "_aug_" in vid:  # apply augmentations
            angle = np.random.uniform(-30, 30) * np.pi / 180  # get rotation angle
            cos, sin = np.cos(angle), np.sin(angle)  # compute cos/sin
            x, z = array[..., 0], array[..., 2]  # get x and z
            array[..., 0] = x * cos - z * sin  # rotate x
            array[..., 2] = x * sin + z * cos  # rotate z
            array *= np.random.uniform(0.8, 1.2)  # apply scaling
            if np.random.random() < 0.5:  # random masking
                ml = int(array.shape[0] * 0.2)  # get mask length
                si = np.random.randint(0, array.shape[0] - ml)  # choose start index
                array[si:si+ml] *= 0  # mask part of sequence

        with open(cache_file, 'wb') as f:  # save to cache
            pickle.dump(array, f)

    def _preprocess_all(self):  # preprocess all samples
        print("\npreprocessing samples...")  # log progress
        for s in tqdm(self.samples, desc="extracting landmarks"):  # loop with progress
            self._preprocess_single(s)  # preprocess sample

    def __len__(self):  # return dataset size
        return len(self.samples)

    def __getitem__(self, idx):  # get item by index
        path, label, vid = self.samples[idx]  # unpack sample
        with open(self.cache_dir / f"{vid}.pkl", 'rb') as f:  # load cached array
            array = pickle.load(f)
        mean, std = np.mean(array), np.std(array)  # compute stats
        array = (array - mean) / (std + 1e-7)  # normalize
        return torch.FloatTensor(array), torch.LongTensor([label])  # return tensors

class SignLanguageModel(nn.Module):  # define model class
    def __init__(self, num_classes):  # init model
        super().__init__()  # call super
        self.lstm = nn.LSTM(  # define lstm
            input_size=225,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(512, num_classes)  # define final layer

    def forward(self, x):  # define forward pass
        if x.dim() == 4:  # flatten if needed
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)  # pass through lstm
        return self.fc(x[:, -1])  # return last step output

def train():  # train the model
    dataset = SignLanguageDataset('data/WLASL_v0.3.json', 'data/videos')  # load dataset
    print(f"\ndataset info - samples: {len(dataset)}  classes: {len(dataset.word2idx)}")  # print dataset info

    train_size = int(0.8 * len(dataset))  # set train size
    val_size = len(dataset) - train_size  # set val size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])  # split data

    loader = lambda ds: DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)  # define loader
    train_loader, val_loader = loader(train_set), loader(val_set)  # create loaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose device
    print(f"\nusing device: {device}")  # print device info

    model = SignLanguageModel(len(dataset.word2idx)).to(device)  # create model
    loss_fn = nn.CrossEntropyLoss()  # define loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # define optimizer

    train_losses, train_accuracies = [], []  # init metrics
    val_losses, val_accuracies = [], []

    for epoch in range(20):  # training loop
        model.train()  # set train mode
        total, correct = 0, 0  # init counters
        total_loss, num_batches = 0.0, 0

        progress_bar = tqdm(train_loader, desc=f"epoch {epoch+1}/20")  # training bar
        for x, y in progress_bar:  # train batch
            x, y = x.to(device), y.squeeze().to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            acc = 100 * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'accuracy': f'{acc:.2f}%'})

        train_acc = 100 * correct / total
        train_avg_loss = total_loss / num_batches
        train_losses.append(train_avg_loss)
        train_accuracies.append(train_acc)

        model.eval()
        total, correct = 0, 0
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():  # disable grads
            for x, y in val_loader:
                x, y = x.to(device), y.squeeze().to(device)
                out = model(x)
                loss = loss_fn(out, y)
                total_loss += loss.item()
                num_batches += 1
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = 100 * correct / total
        val_avg_loss = total_loss / num_batches
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_acc)

        print(f"\nepoch {epoch+1} - train acc: {train_acc:.2f}%, val acc: {val_acc:.2f}%")
    
    # save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved to 'trained_model.pth'")

    plt.figure(figsize=(12, 4))  # plot figure
    plt.subplot(1, 2, 1)  # loss subplot
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)  # accuracy subplot
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='validation')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()

    plt.tight_layout()  # fix layout
    plt.savefig('training_progress.png')  # save figure
    plt.close()  # close plot

if __name__ == '__main__':  # run script
    train()  # start training