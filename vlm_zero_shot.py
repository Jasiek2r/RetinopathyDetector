import base64
import io
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np
from ml.concrete.retinopathy_dataset import RetinopathyDataset
from ml.concrete.retinopathy_pipeline import RetinopathyPipeline

# Inicjalizacja klienta OpenAI wskazującego na lokalny serwer vLLM (port 8000)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM tego nie wymaga
)


def evaluate_zero_shot(test_dataset):
    y_true = []
    y_pred = []

    prompt = (
        "You are an expert ophthalmologist. Analyze this retinal image for Diabetic Retinopathy. "
        "Classify the stage on a scale from 0 to 4 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative). "
        "Return ONLY a single integer number (0, 1, 2, 3, or 4) representing the diagnosis."
    )

    for image, label in tqdm(test_dataset, desc="Zero-shot Evaluation"):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_base64}"

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3.5-4B",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=5,
                temperature=0.0
            )

            answer_text = response.choices[0].message.content.strip()
            pred = int(answer_text)

        except Exception as e:
            print(f"Błąd inferencji: {e}")
            pred = -1  #

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
