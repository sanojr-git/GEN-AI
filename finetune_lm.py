"""
Experiment 10: Fine-Tuning a Pre-Trained Language Model for a Domain-Specific Application
Course: CS4V48 - GenAI & LLM Laboratory
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

def main():
    print("Loading IMDB movie reviews dataset...")
    dataset = load_dataset("imdb")

    small_train = dataset["train"].shuffle(seed=42).select(range(2000))
    small_test = dataset["test"].shuffle(seed=42).select(range(500))

    print("Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    train_ds = small_train.map(tokenize, batched=True)
    test_ds = small_test.map(tokenize, batched=True)

    print("Loading DistilBERT sequence classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating model...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    save_dir = "./fine_tuned_distilbert_imdb"
    model.save_pretrained(save_dir)
    print(f"Model saved to '{save_dir}'")

if __name__ == "__main__":
    main()
