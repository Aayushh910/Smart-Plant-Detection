def get_treatment(disease):
    info = {
        'Tomato_Blight': 'Remove infected leaves, apply copper fungicide.',
        'Potato_Early_Blight': 'Use neem oil spray and crop rotation.',
        'Potato_Late_Blight': 'Avoid overhead watering; use fungicide with mancozeb.',
        'Tomato_Healthy': 'Plant is healthy! Maintain regular watering and sunlight.'
    }
    return info.get(disease, "No treatment info available.")
