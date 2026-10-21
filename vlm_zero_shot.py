import base64
import io

import torch
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np
from ml.concrete.retinopathy_dataset import RetinopathyDataset
from ml.concrete.retinopathy_pipeline import RetinopathyPipeline
import torchvision.transforms as T

# Inicjalizacja klienta OpenAI wskazującego na lokalny serwer vLLM (port 8000)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM tego nie wymaga
)


def evaluate_zero_shot(test_dataset):
    y_true = []
    y_pred = []

    prompt = "Classify this retinal image into one of the diabetic retinopathy grades from 0 to 4. Return ONLY the integer digit."

    to_pil = T.ToPILImage()

    for image, label in tqdm(test_dataset, desc="Zero-shot Evaluation"):
        if isinstance(image, torch.Tensor):
            img_pil = to_pil(image)
        else:
            img_pil = image

        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_base64}"
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3.5-4B",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI. Output only a single digit from 0 to 4."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "Classify this retinopathy image. Output only the digit (0-4):"}
                        ]
                    }
                ],
                max_tokens=3,
                temperature=0.0
            )

            answer_text = response.choices[0].message.content.strip()

            import re
            numbers = re.findall(r'\d+', answer_text)
            if numbers:
                pred = int(numbers[0])
            else:
                pred = -1

            print(f"Model raw output: {answer_text} | Extracted: {pred}")

        except Exception as e:
            print(f"Błąd inferencji: {e}")
            pred = -1

        y_true.append(label)
        y_pred.append(pred)

    # Obliczenie metryk
    acc = accuracy_score(y_true, y_pred) * 100
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    cm = confusion_matrix(y_true, y_pred)

    print(f"Zero-shot Accuracy: {acc:.2f}%")
    print(f"Zero-shot QWK: {qwk:.4f}")
    print("Confusion Matrix:\n", cm)

    return acc, qwk, cm


pipeline = RetinopathyPipeline()

_, _, test_dataset = pipeline.run(
    dir_path="./augmented_resized_V2",
    max_images=100
)

evaluate_zero_shot(test_dataset)
