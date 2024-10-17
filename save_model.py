## This script is used to download the model and save it to the model store.
from __future__ import annotations
import PIL.Image
import PIL.ImageOps
import transformers

import bentoml
##use bentoML to save the model
def download_model() -> int:
    test_path = "./samples/NORMAL2-IM-1427-0001.jpeg"
    extractor = transformers.ViTImageProcessor.from_pretrained(
        "nickmuchi/vit-finetuned-chest-xray-pneumonia"
    )
    model = transformers.AutoModelForImageClassification.from_pretrained(
        "nickmuchi/vit-finetuned-chest-xray-pneumonia"
    )

##print("Model:", model)
    im = PIL.Image.open(test_path)
    im = PIL.ImageOps.exif_transpose(im).convert("RGB")
    outputs = model(**extractor(images=im, return_tensors="pt"))

    top_k = len(model.config.id2label)  
    probs = outputs.logits.softmax(-1)[0]
    scores, ids = probs.topk(top_k)
##print("Prediction:", [{"score": score, "label": model.config.id2label[id_]} for score, id_ in zip(scores.tolist(), ids.tolist())])
    print(
        "Prediction:",
        [
            {"score": score, "label": model.config.id2label[id_]}
            for score, id_ in zip(scores.tolist(), ids.tolist())
        ],
    )
    try:
        bmodel = bentoml.transformers.get("vit-model-pneumonia")
        print("Model already saved to model store:", bmodel)
    except bentoml.exceptions.NotFound:
        print(
            "Saved model:",
            bentoml.transformers.save_model(
                "vit-model-pneumonia",
                model,
                metadata={"top_k": len(model.config.id2label)},
                custom_objects={"id2label": model.config.id2label},
            ),
        )
##print("Extractor:", extractor)
    try:
        bextractor = bentoml.transformers.get("vit-extractor-pneumonia")
        print("Extractor already saved to model store:", bextractor)
    except bentoml.exceptions.NotFound:
        print(
            "Saved model:",
            bentoml.transformers.save_model("vit-extractor-pneumonia", extractor),
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(download_model())    
##download_model()