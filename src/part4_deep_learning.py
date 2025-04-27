import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(tokenizer_name='bert-base-uncased', max_len=128, batch_size=32):
    # Charger les données
    df = pd.read_parquet('dataset/train.parquet')
    df_test = pd.read_parquet('dataset/test.parquet')
    # Split train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.1, random_state=42, stratify=df['label']
    )
    test_texts = df_test['text']
    test_labels = df_test['label']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len)
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, bidirectional=False, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        conv_x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(cx, dim=2)[0] for cx in conv_x]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.fc(dropped)

# GRU model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, bidirectional=False, dropout=0.5):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)
        return self.fc(hidden)

def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device, model_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    model_dir = f'outputs/part4/models/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs('outputs/part4/plots', exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = np.mean(val_losses)
        val_acc = correct / total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"{model_name} Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.4f} val_loss: {avg_val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

    # Training curves
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig(f'outputs/part4/plots/{model_name}_training_curve.png')
    plt.close()

    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    # Save metrics
    metrics_path = os.path.join(model_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"accuracy: {test_acc}\nprecision: {prec}\nrecall: {rec}\nf1: {f1}\n")
    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Test Confusion Matrix')
    plt.savefig(os.path.join('outputs/part4/plots', f'{model_name}_test_confusion_matrix.png'))
    plt.close()
    return history

def run_lstm(train_loader, val_loader, test_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.vocab_size
    model = LSTMClassifier(vocab_size=vocab_size, embed_dim=128,
                           hidden_dim=256, output_dim=2,
                           n_layers=1, bidirectional=True, dropout=0.3)
    return train_model(model, train_loader, val_loader, test_loader,
                       epochs=5, lr=1e-3,
                       device=device, model_name='lstm')

def run_cnn(train_loader, val_loader, test_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.vocab_size
    model = TextCNN(vocab_size=vocab_size, embed_dim=128, num_classes=2,
                    kernel_sizes=[3,4,5], num_filters=100, dropout=0.3)
    return train_model(model, train_loader, val_loader, test_loader,
                       epochs=5, lr=1e-3,
                       device=device, model_name='textcnn')

def run_gru(train_loader, val_loader, test_loader, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.vocab_size
    model = GRUClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256,
                          output_dim=2, n_layers=1, bidirectional=True, dropout=0.3)
    return train_model(model, train_loader, val_loader, test_loader,
                       epochs=5, lr=1e-3,
                       device=device, model_name='gru')

def run_transformer(train_dataset, val_dataset, test_dataset, tokenizer,
                    pretrained_model_name='distilbert-base-uncased'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=2
    ).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        cm = confusion_matrix(labels, preds)
        os.makedirs('outputs/part4/plots', exist_ok=True)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{pretrained_model_name} Confusion Matrix')
        plt.savefig(f'outputs/part4/plots/{pretrained_model_name}_confusion_matrix.png')
        plt.close()
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    training_args = TrainingArguments(
        output_dir='outputs/part4/transformer',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='outputs/part4/logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    test_metrics = trainer.predict(test_dataset).metrics
    with open('outputs/part4/transformer_test_metrics.txt', 'w') as f:
        f.write(str(test_metrics))
    print(f"Transformer test metrics: {test_metrics}")
    return test_metrics

def main():
    os.makedirs('outputs/part4', exist_ok=True)
    train_loader, val_loader, test_loader, tokenizer = prepare_data()
    run_lstm(train_loader, val_loader, test_loader, tokenizer)
    run_cnn(train_loader, val_loader, test_loader, tokenizer)
    run_gru(train_loader, val_loader, test_loader, tokenizer)
    run_transformer(train_loader.dataset, val_loader.dataset, test_loader.dataset, tokenizer)

if __name__ == '__main__':
    main()
