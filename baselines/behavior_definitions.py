"""
Feature-based behavior definitions for Furhat music robot
Based on audio feature ranges instead of genres
"""

# Feature ranges for categorization
FEATURE_RANGES = {
    'tempo': {
        'very_slow': (0, 70),  # Contemplative
        'slow': (70, 90),  # Relaxed
        'moderate': (90, 110),  # Balanced
        'fast': (110, 130),  # Energetic
        'very_fast': (130, 200)  # Intense
    },
    'energy': {
        'very_low': (0, 0.2),  # Quiet/Soft
        'low': (0.2, 0.4),  # Gentle
        'medium': (0.4, 0.6),  # Balanced
        'high': (0.6, 0.8),  # Strong
        'very_high': (0.8, 1.0)  # Powerful
    },
    'spectral_centroid': {
        'very_low': (0, 1000),  # Deep/Dark
        'low': (1000, 2000),  # Warm
        'medium': (2000, 3000),  # Neutral
        'high': (3000, 4000),  # Bright
        'very_high': (4000, 10000)  # Brilliant
    },
    'zero_crossing_rate': {
        'very_low': (0, 0.05),  # Smooth
        'low': (0.05, 0.1),  # Clean
        'medium': (0.1, 0.15),  # Textured
        'high': (0.15, 0.2),  # Rough
        'very_high': (0.2, 1.0)  # Noisy
    }
}

# Behavior mappings based on feature combinations
FEATURE_BEHAVIOR_MAP = {
    # TEMPO-BASED PRIMARY BEHAVIORS
    'tempo_behaviors': {
        'very_slow': {
            'gesture': 'Nod',
            'gesture_speed': 0.4,
            'gesture_intensity': 0.5,
            'gesture_duration': 3.0,
            'head_movement_pattern': 'slow_sway',
            'head_pitch_range': (-0.1, 0.1),
            'head_yaw_range': (-0.15, 0.15),
            'blink_frequency': 0.3  # Slow blinking
        },
        'slow': {
            'gesture': 'Nod',
            'gesture_speed': 0.6,
            'gesture_intensity': 0.7,
            'gesture_duration': 2.5,
            'head_movement_pattern': 'gentle_nod',
            'head_pitch_range': (-0.15, 0.15),
            'head_yaw_range': (-0.1, 0.1),
            'blink_frequency': 0.4
        },
        'moderate': {
            'gesture': 'Nod',
            'gesture_speed': 1.0,
            'gesture_intensity': 1.0,
            'gesture_duration': 2.0,
            'head_movement_pattern': 'rhythmic_nod',
            'head_pitch_range': (-0.2, 0.2),
            'head_yaw_range': (-0.05, 0.05),
            'blink_frequency': 0.5
        },
        'fast': {
            'gesture': 'Shake',
            'gesture_speed': 1.3,
            'gesture_intensity': 1.2,
            'gesture_duration': 1.5,
            'head_movement_pattern': 'active_bob',
            'head_pitch_range': (-0.25, 0.25),
            'head_yaw_range': (-0.2, 0.2),
            'blink_frequency': 0.6
        },
        'very_fast': {
            'gesture': 'Shake',
            'gesture_speed': 1.8,
            'gesture_intensity': 1.5,
            'gesture_duration': 1.0,
            'head_movement_pattern': 'vigorous_shake',
            'head_pitch_range': (-0.3, 0.3),
            'head_yaw_range': (-0.3, 0.3),
            'blink_frequency': 0.8
        }
    },

    # ENERGY-BASED MODIFIERS
    'energy_behaviors': {
        'very_low': {
            'expression': 'Thoughtful',
            'expression_intensity': 0.6,
            'led_color': '#4A5568',  # Dim gray
            'led_brightness': 0.3,
            'eye_openness': 0.7,  # Slightly closed eyes
            'gesture_amplitude_modifier': 0.5
        },
        'low': {
            'expression': 'Oh',
            'expression_intensity': 0.7,
            'led_color': '#718096',  # Soft gray-blue
            'led_brightness': 0.5,
            'eye_openness': 0.85,
            'gesture_amplitude_modifier': 0.7
        },
        'medium': {
            'expression': 'Smile',
            'expression_intensity': 0.8,
            'led_color': '#FFFFFF',  # Pure white
            'led_brightness': 0.7,
            'eye_openness': 1.0,
            'gesture_amplitude_modifier': 1.0
        },
        'high': {
            'expression': 'BigSmile',
            'expression_intensity': 1.0,
            'led_color': '#FFD700',  # Gold
            'led_brightness': 0.9,
            'eye_openness': 1.0,
            'gesture_amplitude_modifier': 1.3
        },
        'very_high': {
            'expression': 'Surprise',
            'expression_intensity': 1.2,
            'led_color': '#FF6B6B',  # Bright red
            'led_brightness': 1.0,
            'eye_openness': 1.1,  # Wide eyes
            'gesture_amplitude_modifier': 1.5
        }
    },

    # SPECTRAL CENTROID (BRIGHTNESS) MODIFIERS
    'brightness_behaviors': {
        'very_low': {
            'led_color_modifier': {'hue_shift': -30, 'saturation': 0.7},
            'additional_expression': 'BrowFrown',
            'attention_y_offset': -0.1  # Look slightly down
        },
        'low': {
            'led_color_modifier': {'hue_shift': -15, 'saturation': 0.85},
            'additional_expression': None,
            'attention_y_offset': -0.05
        },
        'medium': {
            'led_color_modifier': {'hue_shift': 0, 'saturation': 1.0},
            'additional_expression': None,
            'attention_y_offset': 0
        },
        'high': {
            'led_color_modifier': {'hue_shift': 15, 'saturation': 1.1},
            'additional_expression': 'BrowRaise',
            'attention_y_offset': 0.05  # Look slightly up
        },
        'very_high': {
            'led_color_modifier': {'hue_shift': 30, 'saturation': 1.2},
            'additional_expression': 'BrowRaise',
            'attention_y_offset': 0.1
        }
    },

    # ZERO CROSSING RATE (PERCUSSIVENESS) MODIFIERS
    'percussiveness_behaviors': {
        'very_low': {
            'gesture_smoothness': 1.0,  # Very smooth
            'gesture_type_override': None,
            'micro_movements': False
        },
        'low': {
            'gesture_smoothness': 0.8,
            'gesture_type_override': None,
            'micro_movements': False
        },
        'medium': {
            'gesture_smoothness': 0.6,
            'gesture_type_override': None,
            'micro_movements': True,
            'micro_movement_frequency': 0.5
        },
        'high': {
            'gesture_smoothness': 0.4,
            'gesture_type_override': 'Shake',  # Force shake for percussive music
            'micro_movements': True,
            'micro_movement_frequency': 0.8
        },
        'very_high': {
            'gesture_smoothness': 0.2,  # Very sharp movements
            'gesture_type_override': 'Shake',
            'micro_movements': True,
            'micro_movement_frequency': 1.0,
            'add_blink_accent': True  # Blink on beats
        }
    }
}

# Combined behavior profiles for specific feature combinations
COMBINED_PROFILES = {
    'contemplative': {
        # Slow tempo + Low energy + Low brightness
        'conditions': {
            'tempo': 'very_slow',
            'energy': 'low',
            'spectral_centroid': 'low'
        },
        'behaviors': {
            'primary_gesture': 'Nod',
            'gesture_pattern': 'slow_breathing',
            'led_color': '#6B7280',
            'led_pattern': 'slow_pulse',
            'expression_sequence': ['Thoughtful', 'CloseEyes', 'OpenEyes'],
            'attention_pattern': 'drift_gaze'
        }
    },

    'energetic': {
        # Fast tempo + High energy + High brightness
        'conditions': {
            'tempo': 'fast',
            'energy': 'high',
            'spectral_centroid': 'high'
        },
        'behaviors': {
            'primary_gesture': 'Shake',
            'gesture_pattern': 'rhythmic_bounce',
            'led_color': '#FFA500',
            'led_pattern': 'strobe',
            'expression_sequence': ['BigSmile', 'Wink', 'Surprise'],
            'attention_pattern': 'active_scan'
        }
    },

    'groovy': {
        # Moderate tempo + Medium energy + Varied brightness
        'conditions': {
            'tempo': 'moderate',
            'energy': 'medium',
            'zero_crossing_rate': 'medium'
        },
        'behaviors': {
            'primary_gesture': 'Nod',
            'gesture_pattern': 'syncopated_nod',
            'led_color': '#00CED1',
            'led_pattern': 'wave',
            'expression_sequence': ['Smile', 'Wink'],
            'attention_pattern': 'side_to_side'
        }
    },

    'intense': {
        # Very fast tempo + Very high energy + High percussiveness
        'conditions': {
            'tempo': 'very_fast',
            'energy': 'very_high',
            'zero_crossing_rate': 'high'
        },
        'behaviors': {
            'primary_gesture': 'Shake',
            'gesture_pattern': 'aggressive_headbang',
            'led_color': '#DC143C',
            'led_pattern': 'rapid_flash',
            'expression_sequence': ['Surprise', 'BrowFrown'],
            'attention_pattern': 'intense_focus'
        }
    },

    'mellow': {
        # Slow tempo + Medium energy + Medium brightness
        'conditions': {
            'tempo': 'slow',
            'energy': 'medium',
            'spectral_centroid': 'medium'
        },
        'behaviors': {
            'primary_gesture': 'Roll',
            'gesture_pattern': 'gentle_sway',
            'led_color': '#87CEEB',
            'led_pattern': 'breathe',
            'expression_sequence': ['Smile', 'Thoughtful'],
            'attention_pattern': 'relaxed_wander'
        }
    }
}

# Special behavior patterns for specific audio events
EVENT_BEHAVIORS = {
    'beat': {
        'strong': {
            'gesture': 'Nod',
            'intensity_spike': 1.5,
            'led_flash': True,
            'blink': False
        },
        'weak': {
            'gesture': None,
            'intensity_spike': 1.1,
            'led_flash': False,
            'blink': False
        }
    },

    'onset': {
        'percussion': {
            'gesture': 'Blink',
            'led_pulse': True,
            'micro_movement': True
        },
        'melodic': {
            'gesture': None,
            'led_pulse': False,
            'expression_change': True
        }
    },

    'silence': {
        'gesture': 'GazeAway',
        'led_dim': True,
        'expression': 'Thoughtful',
        'return_to_center': True
    },

    'crescendo': {
        'gesture_intensity_increase': 0.1,  # Per second
        'led_brightness_increase': 0.1,
        'expression_intensity_increase': 0.05,
        'head_lift': True
    },

    'diminuendo': {
        'gesture_intensity_decrease': 0.1,
        'led_brightness_decrease': 0.1,
        'expression_intensity_decrease': 0.05,
        'head_lower': True
    }
}

# Default behavior for when no specific features match
DEFAULT_BEHAVIOR = {
    'gesture': 'Nod',
    'gesture_speed': 1.0,
    'gesture_intensity': 1.0,
    'gesture_duration': 2.0,
    'led_color': '#FFFFFF',
    'led_brightness': 0.7,
    'expression': 'Smile',
    'expression_intensity': 0.8,
    'head_pitch': 0.0,
    'head_yaw': 0.0,
    'attention_x': 0.0,
    'attention_y': 0.0,
    'attention_z': 1.0,
    'description': 'Default neutral behavior'
}


# Function to get categorized value
def categorize_feature(value, feature_name):
    """Categorize a continuous feature value into discrete categories"""
    if feature_name not in FEATURE_RANGES:
        return 'medium'

    ranges = FEATURE_RANGES[feature_name]
    for category, (min_val, max_val) in ranges.items():
        if min_val <= value < max_val:
            return category

    return 'medium'  # Default fallback


# Function to combine behaviors from different features
def combine_behaviors(tempo_cat, energy_cat, brightness_cat, percussive_cat):
    """Combine behaviors from different feature categories"""
    combined = DEFAULT_BEHAVIOR.copy()

    # Apply tempo behaviors
    if tempo_cat in FEATURE_BEHAVIOR_MAP['tempo_behaviors']:
        tempo_behavior = FEATURE_BEHAVIOR_MAP['tempo_behaviors'][tempo_cat]
        combined.update(tempo_behavior)

    # Apply energy modifiers
    if energy_cat in FEATURE_BEHAVIOR_MAP['energy_behaviors']:
        energy_behavior = FEATURE_BEHAVIOR_MAP['energy_behaviors'][energy_cat]
        combined['expression'] = energy_behavior['expression']
        combined['expression_intensity'] = energy_behavior['expression_intensity']
        combined['led_color'] = energy_behavior['led_color']
        combined['led_brightness'] = energy_behavior['led_brightness']

        # Modify gesture intensity based on energy
        if 'gesture_amplitude_modifier' in energy_behavior:
            combined['gesture_intensity'] *= energy_behavior['gesture_amplitude_modifier']

    # Apply brightness modifiers
    if brightness_cat in FEATURE_BEHAVIOR_MAP['brightness_behaviors']:
        brightness_behavior = FEATURE_BEHAVIOR_MAP['brightness_behaviors'][brightness_cat]
        if brightness_behavior['attention_y_offset']:
            combined['attention_y'] = brightness_behavior['attention_y_offset']
        if brightness_behavior['additional_expression']:
            combined['additional_expression'] = brightness_behavior['additional_expression']

    # Apply percussiveness modifiers
    if percussive_cat in FEATURE_BEHAVIOR_MAP['percussiveness_behaviors']:
        percussive_behavior = FEATURE_BEHAVIOR_MAP['percussiveness_behaviors'][percussive_cat]
        if percussive_behavior['gesture_type_override']:
            combined['gesture'] = percussive_behavior['gesture_type_override']
        combined['gesture_smoothness'] = percussive_behavior['gesture_smoothness']
        combined['micro_movements'] = percussive_behavior['micro_movements']

    # Check for special combined profiles
    for profile_name, profile in COMBINED_PROFILES.items():
        conditions_met = all(
            categorize_feature(combined.get(feat, 0), feat) == cat
            for feat, cat in profile['conditions'].items()
        )
        if conditions_met:
            combined['special_profile'] = profile_name
            combined['special_behaviors'] = profile['behaviors']
            break

    return combined