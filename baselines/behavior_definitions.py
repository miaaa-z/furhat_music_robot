import random

FEATURE_RANGES = {
    'tempo': {'slow': (0, 110), 'moderate': (110, 130), 'fast': (130, 300)},
    'energy': {'low': (0, 0.10), 'medium': (0.10, 0.20), 'high': (0.20, 0.40)},
    'brightness': {'dark': (0, 1800), 'neutral': (1800, 2300), 'bright': (2300, 5000)}
}

# LED colors
LED_COLORS = {
    'white': '#FFFFFF', 'red': '#FF0000', 'orange': '#FFA500', 'yellow': '#FFFF00',
    'green': '#00FF00', 'blue': '#0000FF', 'pink': '#FF69B4', 'purple': '#800080'
}

# Behavior map: (tempo, energy, brightness) -> gestures and LED
BEHAVIOR_MAP = {
    # Slow tempo
    ('slow', 'low', 'dark'): {
        'gestures': ['Thoughtful', 'CloseEyes', 'head_sway', 'ExpressSad'],
        'led': LED_COLORS['blue']
    },
    ('slow', 'low', 'neutral'): {
        'gestures': ['Thoughtful', 'CloseEyes', 'head_sway', 'Smile', 'GazeAway'],
        'led': LED_COLORS['blue']
    },
    ('slow', 'low', 'bright'): {
        'gestures': ['Thoughtful', 'head_sway', 'Smile', 'OpenEyes'],
        'led': LED_COLORS['green']
    },

    ('slow', 'medium', 'dark'): {
        'gestures': ['head_sway', 'Smile', 'Thoughtful', 'CloseEyes'],
        'led': LED_COLORS['green']
    },
    ('slow', 'medium', 'neutral'): {
        'gestures': ['head_sway', 'Smile', 'GazeAway'],
        'led': LED_COLORS['green']
    },
    ('slow', 'medium', 'bright'): {
        'gestures': ['head_sway', 'Smile', 'BrowRaise', 'OpenEyes', 'BigSmile'],
        'led': LED_COLORS['yellow']
    },

    ('slow', 'high', 'dark'): {
        'gestures': ['head_nod_fast', 'ExpressSad', 'BrowRaise', 'Thoughtful'],
        'led': LED_COLORS['purple']
    },
    ('slow', 'high', 'neutral'): {
        'gestures': ['head_nod_fast', 'BrowRaise', 'Surprise', 'Smile'],
        'led': LED_COLORS['purple']
    },
    ('slow', 'high', 'bright'): {
        'gestures': ['head_nod_fast', 'Surprise', 'BrowRaise', 'BigSmile', 'OpenEyes'],
        'led': LED_COLORS['orange']
    },

    # Moderate tempo
    ('moderate', 'low', 'dark'): {
        'gestures': ['head_sway', 'Thoughtful', 'GazeAway', 'CloseEyes'],
        'led': LED_COLORS['green']
    },
    ('moderate', 'low', 'neutral'): {
        'gestures': ['head_sway', 'Smile', 'GazeAway', 'head_nod_fast'],
        'led': LED_COLORS['green']
    },
    ('moderate', 'low', 'bright'): {
        'gestures': ['head_sway', 'Smile', 'BrowRaise', 'OpenEyes'],
        'led': LED_COLORS['white']
    },

    ('moderate', 'medium', 'dark'): {
        'gestures': ['head_sway', 'Smile', 'Thoughtful', 'ExpressSad'],
        'led': LED_COLORS['white']
    },
    ('moderate', 'medium', 'neutral'): {
        'gestures': ['head_sway', 'Smile', 'BrowRaise', 'GazeAway'],
        'led': LED_COLORS['white']
    },
    ('moderate', 'medium', 'bright'): {
        'gestures': ['head_sway', 'Smile', 'BrowRaise', 'Surprise', 'BigSmile'],
        'led': LED_COLORS['yellow']
    },

    ('moderate', 'high', 'dark'): {
        'gestures': ['head_nod_fast', 'head_sway', 'Thoughtful', 'BrowRaise'],
        'led': LED_COLORS['yellow']
    },
    ('moderate', 'high', 'neutral'): {
        'gestures': ['head_nod_fast', 'head_sway', 'Smile', 'Surprise', 'BrowRaise'],
        'led': LED_COLORS['yellow']
    },
    ('moderate', 'high', 'bright'): {
        'gestures': ['head_nod_fast', 'Smile', 'Surprise', 'BrowRaise', 'BigSmile'],
        'led': LED_COLORS['orange']
    },

    # Fast tempo
    ('fast', 'low', 'dark'): {
        'gestures': ['head_shake_fast', 'head_sway', 'Thoughtful', 'CloseEyes'],
        'led': LED_COLORS['pink']
    },
    ('fast', 'low', 'neutral'): {
        'gestures': ['head_shake_fast', 'head_sway', 'Smile', 'GazeAway'],
        'led': LED_COLORS['pink']
    },
    ('fast', 'low', 'bright'): {
        'gestures': ['head_shake_fast', 'Smile', 'BrowRaise', 'OpenEyes'],
        'led': LED_COLORS['yellow']
    },

    ('fast', 'medium', 'dark'): {
        'gestures': ['head_shake_fast', 'Smile', 'Thoughtful', 'swaying'],
        'led': LED_COLORS['orange']
    },
    ('fast', 'medium', 'neutral'): {
        'gestures': ['head_shake_fast', 'Smile', 'BrowRaise', 'swaying', 'GazeAway'],
        'led': LED_COLORS['orange']
    },
    ('fast', 'medium', 'bright'): {
        'gestures': ['head_shake_fast', 'Smile', 'BrowRaise', 'Surprise', 'BigSmile'],
        'led': LED_COLORS['red']
    },

    ('fast', 'high', 'dark'): {
        'gestures': ['head_shake_fast', 'head_nod_fast', 'BrowRaise', 'ExpressSad'],
        'led': LED_COLORS['red']
    },
    ('fast', 'high', 'neutral'): {
        'gestures': ['head_shake_fast', 'head_nod_fast', 'Surprise', 'BrowRaise'],
        'led': LED_COLORS['red']
    },
    ('fast', 'high', 'bright'): {
        'gestures': ['head_shake_fast', 'head_nod_fast', 'Surprise', 'BrowRaise', 'BigSmile'],
        'led': LED_COLORS['red']
    }
}


def categorize_tempo(bpm):
    return 'slow' if bpm < 110 else 'moderate' if bpm < 130 else 'fast'


def categorize_energy(energy):
    return 'low' if energy < 0.10 else 'medium' if energy < 0.20 else 'high'


def categorize_brightness(hz):
    return 'dark' if hz < 1800 else 'neutral' if hz < 2300 else 'bright'


def get_behavior(tempo_cat, energy_cat, brightness_cat):
    """Get two gesture names and LED color"""

    config = BEHAVIOR_MAP.get((tempo_cat, energy_cat, brightness_cat),
                              {'gestures': ['Smile', 'head_sway'], 'led': LED_COLORS['white']})

    # Randomly select 2 gestures from the list
    gestures = random.sample(config['gestures'], min(2, len(config['gestures'])))

    return gestures, config['led']
