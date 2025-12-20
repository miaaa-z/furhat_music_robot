FEATURE_RANGES = {
    'tempo': {
        'slow': (0, 100),
        'moderate': (100, 130),
        'fast': (130, 200)
    },
    'energy': {
        'low': (0, 0.15),  # Quiet
        'medium': (0.15, 0.23),  # Moderate
        'high': (0.23, 1.0)  # Loud
    }
}

# Global behavior mapping based on tempo + energy
GLOBAL_BEHAVIOR_MAP = {
    # SLOW TEMPO
    ('slow', 'low'): {
        'base_gesture': 'Nod',
        'gesture_speed': 0.5,
        'led_color': '#6B7280',  # Gray-blue - contemplative
        'expression': 'Thoughtful',
        'beat_response': 'subtle',
        'description': 'Slow and quiet - contemplative mood'
    },
    ('slow', 'medium'): {
        'base_gesture': 'Nod',
        'gesture_speed': 0.7,
        'led_color': '#87CEEB',  # Sky blue - calm
        'expression': 'Smile',
        'beat_response': 'moderate',
        'description': 'Slow and medium - relaxed mood'
    },
    ('slow', 'high'): {
        'base_gesture': 'Nod',
        'gesture_speed': 0.8,
        'led_color': '#9370DB',  # Purple - dramatic
        'expression': 'Oh',
        'beat_response': 'moderate',
        'description': 'Slow but intense - dramatic mood'
    },

    # MODERATE TEMPO
    ('moderate', 'low'): {
        'base_gesture': 'Nod',
        'gesture_speed': 0.8,
        'led_color': '#B0C4DE',  # Light steel blue - gentle
        'expression': 'Smile',
        'beat_response': 'subtle',
        'description': 'Moderate tempo, low energy - gentle groove'
    },
    ('moderate', 'medium'): {
        'base_gesture': 'Nod',
        'gesture_speed': 1.0,
        'led_color': '#FFFFFF',  # White - neutral/happy
        'expression': 'Smile',
        'beat_response': 'moderate',
        'description': 'Moderate tempo and energy - balanced mood'
    },
    ('moderate', 'high'): {
        'base_gesture': 'Nod',
        'gesture_speed': 1.2,
        'led_color': '#FFD700',  # Gold - upbeat
        'expression': 'BigSmile',
        'beat_response': 'strong',
        'description': 'Moderate tempo, high energy - upbeat mood'
    },

    # FAST TEMPO
    ('fast', 'low'): {
        'base_gesture': 'Shake',
        'gesture_speed': 1.0,
        'led_color': '#00CED1',  # Dark cyan - quirky
        'expression': 'Smile',
        'beat_response': 'moderate',
        'description': 'Fast but quiet - quirky/playful'
    },
    ('fast', 'medium'): {
        'base_gesture': 'Shake',
        'gesture_speed': 1.3,
        'led_color': '#FFA500',  # Orange - energetic
        'expression': 'BigSmile',
        'beat_response': 'strong',
        'description': 'Fast and medium energy - energetic'
    },
    ('fast', 'high'): {
        'base_gesture': 'Shake',
        'gesture_speed': 1.5,
        'led_color': '#FF6B6B',  # Bright red - intense
        'expression': 'Surprise',
        'beat_response': 'strong',
        'description': 'Fast and loud - intense/exciting'
    }
}

# Beat response behaviors
BEAT_BEHAVIORS = {
    'subtle': {
        'strong_beat': {
            'gesture': 'Nod',
            'led_flash': True,
            'led_flash_duration': 0.1,
            'led_flash_brightness': 1.2,  # 20% brighter
            'blink': False,
            'description': 'Subtle nod on strong beats only'
        },
        'weak_beat': None  # No response to weak beats
    },

    'moderate': {
        'strong_beat': {
            'gesture': 'Nod',
            'led_flash': True,
            'led_flash_duration': 0.15,
            'led_flash_brightness': 1.3,  # 30% brighter
            'blink': False,
            'description': 'Clear nod with LED flash on strong beats'
        },
        'weak_beat': {
            'gesture': None,  # No gesture
            'led_flash': True,
            'led_flash_duration': 0.08,
            'led_flash_brightness': 1.1,  # 10% brighter
            'blink': False,
            'description': 'Small LED pulse on weak beats'
        }
    },

    'strong': {
        'strong_beat': {
            'gesture': 'Nod',  # or can be 'Shake' based on tempo
            'led_flash': True,
            'led_flash_duration': 0.2,
            'led_flash_brightness': 1.5,  # 50% brighter
            'blink': True,  # Blink on strong beats
            'description': 'Emphatic response with gesture, LED, and blink'
        },
        'weak_beat': {
            'gesture': None,
            'led_flash': True,
            'led_flash_duration': 0.1,
            'led_flash_brightness': 1.2,
            'blink': False,
            'description': 'Moderate LED flash on weak beats'
        }
    }
}

DEFAULT_BEHAVIOR = {
    'base_gesture': 'Nod',
    'gesture_speed': 1.0,
    'led_color': '#FFFFFF',
    'expression': 'Smile',
    'beat_response': 'moderate',
    'description': 'Default neutral behavior'
}


# Helper functions
def categorize_tempo(tempo_bpm):
    """Categorize tempo value into slow/moderate/fast"""
    if tempo_bpm < 90:
        return 'slow'
    elif tempo_bpm < 120:
        return 'moderate'
    else:
        return 'fast'


def categorize_energy(energy_value):
    """Categorize energy (RMS) value into low/medium/high"""
    if energy_value < 0.02:
        return 'low'
    elif energy_value < 0.05:
        return 'medium'
    else:
        return 'high'


def get_behavior(tempo_cat, energy_cat):
    """
    Get behavior configuration for given tempo and energy categories

    Args:
        tempo_cat (str): 'slow', 'moderate', or 'fast'
        energy_cat (str): 'low', 'medium', or 'high'

    Returns:
        dict: Behavior configuration dictionary
    """
    key = (tempo_cat, energy_cat)
    behavior = GLOBAL_BEHAVIOR_MAP.get(key, DEFAULT_BEHAVIOR)
    return behavior.copy()  # Return a copy to avoid mutations


def get_beat_behavior(beat_response_level, beat_strength):
    """
    Get beat response behavior

    Args:
        beat_response_level (str): 'subtle', 'moderate', or 'strong'
        beat_strength (str): 'strong_beat' or 'weak_beat'

    Returns:
        dict or None: Beat behavior configuration, or None if no response
    """
    if beat_response_level not in BEAT_BEHAVIORS:
        beat_response_level = 'moderate'

    beat_config = BEAT_BEHAVIORS[beat_response_level].get(beat_strength)

    if beat_config is None:
        return None

    return beat_config.copy()