"""
Fine-tune DistilBERT on the custom training dataset.
Run: python train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model.trainer import train

if __name__ == '__main__':
    acc = train(
        data_path='data/training_data.csv',
        output_dir='model/fine_tuned',
        epochs=3,
    )
    print(f"\nFinal accuracy: {acc:.2f}%")
