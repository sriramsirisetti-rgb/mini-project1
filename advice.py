"""Simple, general guidance shown to farmers. Not a substitute for an agronomist."""

GENERIC = {
    "healthy": "The leaf looks healthy. Keep monitoring and maintain good watering and spacing.",
    "blight": "Remove and destroy affected leaves. Avoid overhead watering and improve airflow. Ask your local agri-extension officer about a suitable fungicide.",
    "rust": "Remove infected leaves, avoid wetting foliage, and consult an extension officer about fungicide options.",
    "mold": "Reduce humidity, improve ventilation, and remove infected leaves.",
    "spot": "Remove spotted leaves, avoid overhead irrigation, and rotate crops next season.",
    "mildew": "Improve airflow, avoid crowding plants, and consult an extension officer about treatment.",
    "virus": "Viral diseases cannot be cured. Remove and destroy infected plants and control insect vectors (whiteflies, aphids).",
    "mite": "Spray with water or neem-based products and consult an extension officer for miticides.",
    "scab": "Remove fallen and infected leaves and consult an extension officer about fungicide timing.",
    "rot": "Remove infected parts, improve drainage, and avoid injuring the plant.",
    "greening": "No cure. Remove infected trees and control psyllid insects.",
    "scorch": "Water consistently and protect plants from extreme heat and wind.",
}


def get_advice(disease_name: str) -> str:
    d = disease_name.lower()
    for key, text in GENERIC.items():
        if key in d:
            return text
    return "Consult your local agriculture extension officer for confirmation and treatment."
