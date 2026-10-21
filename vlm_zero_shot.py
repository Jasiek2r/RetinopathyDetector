import base64
import io
import re

import torch
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from ml.concrete.retinopathy_pipeline import RetinopathyPipeline
import torchvision.transforms as T

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)


def evaluate_zero_shot(test_dataset):
    y_true = []
    y_pred = []

    prompt = (
        "Classify this retinal image into diabetic retinopathy grades:\n"
        "0: No DR\n"
        "1: Mild\n"
        "2: Moderate\n"
        "3: Severe\n"
        "4: Proliferative\n"
        "Return ONLY the single digit number (0, 1, 2, 3, or 4)."
    )

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
                model="Qwen/Qwen2.5-VL-3B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text",
                             "text": prompt}
                        ]
                    }
                ],
                max_tokens=3,
                temperature=0.0
            )

            answer_text = response.choices[0].message.content.strip()

            numbers = re.findall(r'[0-4]', answer_text)
            if numbers:
                pred = int(numbers[-1])  # Ostatnia cyfra to zazwyczaj ostateczny werdykt
            else:
                pred = -1

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
    max_images=None
)

evaluate_zero_shot(test_dataset)
