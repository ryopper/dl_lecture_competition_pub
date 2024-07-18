import re
import random
import time
from statistics import mode
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from PIL import Image

model_path = "/workspace/models/InternVL-Chat-V1-5"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 4bit量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 学習済みモデルの読み込み
intern_model = AutoModel.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16).eval()

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, model, tokenizer, answer=True):
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

        self.answer2idx = {}
        self.idx2answer = {}

        if self.answer:

            # Training_dataに含まれるAnswerを全て取得
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            # 追加でClass_mappingに含まれるAnswerを取得
            class_mapping = pd.read_csv("/workspace/class_mapping.csv")
            self.idx2answer = {}
            for word, idx in zip(class_mapping["answer"], class_mapping["class_id"]):
                word = process_text(word)
                self.answer2idx[word] = idx

            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        self.model = model
        self.tokenizer = tokenizer

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def extract_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=303)
        input_ids = inputs.input_ids.to(self.model.device)
        with torch.no_grad():
            text_features = self.model.language_model.model.tok_embeddings(input_ids)
        return text_features

    def extract_image_features(self, pixel_values):
        with torch.no_grad():
            image_features = self.model.vision_model.embeddings(pixel_values)
        return image_features

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        input_size = 224
        max_num = 1
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(torch.bfloat16)
        pixel_values = self.extract_image_features(pixel_values)

        question = self.df["question"][idx]
        # 質問文の前処理
        question = process_text(question)
        question = self.extract_text_features(question)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return pixel_values, question, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return pixel_values, question

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()

        # vision
        self.vision_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(3, 3), padding=1)
        self.vision_bn1 = nn.BatchNorm2d(16)
        self.vision_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(3, 3), padding=1)
        self.vision_bn2 = nn.BatchNorm2d(32)
        self.vision_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(3, 3), padding=1)
        self.vision_bn3 = nn.BatchNorm2d(64)
        self.vision_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        # text
        self.text_conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 2), stride=(2, 2), padding=1)
        self.text_bn1 = nn.BatchNorm2d(4)
        self.text_conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 2), stride=(2, 2), padding=1)
        self.text_bn2 = nn.BatchNorm2d(8)
        self.text_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2), stride=(2, 2), padding=1)
        self.text_bn3 = nn.BatchNorm2d(16)
        self.text_conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(2, 2), padding=1)
        self.text_bn4 = nn.BatchNorm2d(32)
        self.text_pool= nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # combined
        self.fc1 = nn.Linear(1600, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_answer)

        # 重みの初期化
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, vision_input, text_input):
        
        # vision 
        v = F.relu(self.vision_bn1(self.vision_bn1(self.vision_conv1(vision_input))))
        v = F.relu(self.vision_bn2(self.vision_bn2(self.vision_conv2(v))))
        v = self.vision_pool(v)
        v = F.relu(self.vision_bn3(self.vision_conv3(v)))
        v = self.vision_pool(v)
        v = v.view(v.size(0), -1)

        # text
        t = F.relu(self.text_bn1(self.text_conv1(text_input)))
        t = self.text_pool(t)
        t = F.relu(self.text_bn2(self.text_conv2(t)))
        t = self.text_pool(t)
        t = F.relu(self.text_bn3(self.text_conv3(t)))
        t = self.text_pool(t)
        t = F.relu(self.text_bn4(self.text_conv4(t)))
        t = self.text_pool(t)
        t = t.view(t.size(0), -1)

        # combined
        combined = torch.cat((v, t), dim=1)
        x = F.relu(self.bn1(self.fc1(combined)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x
    
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader, desc="Batch", leave=False):
        image, question, answers, mode_answer = \
            image.to(device, dtype=torch.bfloat16), question.to(device, dtype=torch.bfloat16), answers.to(device), mode_answer.to(device)
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze().to(torch.long))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers) 
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item() 

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device, dtype=torch.bfloat16), question.to(device, dtype=torch.bfloat16), answers.to(device), mode_answer.to(device)
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze().to(torch.long))

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = VQADataset(df_path="/workspace/data/train.json", image_dir="/workspace/data/train", model=intern_model, tokenizer=tokenizer, answer=True)
    test_dataset = VQADataset(df_path="/workspace/data/valid.json", image_dir="/workspace/data/valid", model=intern_model, tokenizer=tokenizer, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)
    model = model.to(torch.bfloat16)  # モデル全体をbfloat16に変換


    num_epoch = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in tqdm(range(num_epoch), desc="Epoch"):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
                f"train time: {train_time:.2f} [s]\n"
                f"train loss: {train_loss:.4f}\n"
                f"train acc: {train_acc:.4f}\n"
                f"train simple acc: {train_simple_acc:.4f}")
        
        if True:
            model.eval()
            submission = []
            for image, question in test_loader:
                image, question = image.to(device, dtype=torch.bfloat16), question.to(device, dtype=torch.bfloat16)
                pred = model(image, question)
                pred = pred.argmax(1).cpu().item()
                submission.append(pred)

            submission = [train_dataset.idx2answer[id] for id in submission]
            submission = np.array(submission)
            torch.save(model.state_dict(), f"/workspace/submissions/model/model_{epoch}.pth")
            np.save(f"/workspace/submissions/npy/submission_{epoch}.npy", submission)

if __name__ == "__main__":
    main()