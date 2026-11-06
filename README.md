# 🌿 Leaf Doctor: CNN Plant Leaf Disease Detection

A Convolutional Neural Network that classifies crop leaf images into disease
categories, trained on the **PlantVillage** dataset, with a farmer-facing
**Streamlit** app for camera/upload diagnosis.

## Pipeline
1. **Data**: PlantVillage (38 classes, ~54k images), stratified 70/15/15 split
2. **Pre-processing**: resize to 224x224, normalisation baked into the model
3. **Feature engineering / augmentation**: flips, rotation, zoom, contrast, brightness (simulates field photos); class weights for imbalance
4. **Model**: MobileNetV2 transfer learning + fine-tuning (or a custom CNN with `--arch custom`)
5. **Evaluation**: per-class precision/recall/F1, confusion matrix (`reports/`)
6. **App**: Streamlit, with confidence threshold, top-3 predictions, and plain-language treatment advice

## Run it
```bash
pip install -r requirements.txt

# 1. Download PlantVillage (Kaggle: "abdallahalidev/plantvillage-dataset", use the "color" folder)
#    and place it at data/PlantVillage/<class_name>/*.jpg

# 2. Train
python train.py --data data/PlantVillage --arch mobilenet

# 3. Launch the app
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push to GitHub, then at share.streamlit.io choose the repo and `app.py`.
Commit `model/plant_model.keras` and `model/class_names.json` (keep the model under 100 MB).

## Limitations
PlantVillage images are lab-style (plain background). Accuracy on real field
photos is lower; add field images for fine-tuning before real-world use.
