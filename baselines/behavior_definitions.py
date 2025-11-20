# Based on the literature  written in rule_based_design.md

GENRE_BEHAVIOR_MAP = {
    'jazz': {
        'gesture': 'Nod',
        'speed': 0.7,
        'intensity': 0.8,
        'led_color': '#6496FF',
        'expression': 'Smile',
        'head_pitch': 0.0,
        'head_yaw': 0.0,
        'description': 'Contemplative sway with blue LED'
    },
    'rock': {
        'gesture': 'Shake',
        'speed': 1.5,
        'intensity': 1.5,
        'led_color': '#FF3232',
        'expression': 'Smile',
        'head_pitch': 0.1,
        'head_yaw': 0.0,
        'description': 'Energetic head shake with red LED'
    },
    'classical': {
        'gesture': 'Nod',
        'speed': 0.5,
        'intensity': 0.6,
        'led_color': '#FFF0C8',
        'expression': 'Oh',
        'head_pitch': -0.05,
        'head_yaw': 0.0,
        'description': 'Gentle nod with warm white LED'
    },
    'pop': {
        'gesture': 'Nod',
        'speed': 1.0,
        'intensity': 1.0,
        'led_color': '#FFD700',
        'expression': 'Smile',
        'head_pitch': 0.05,
        'head_yaw': 0.0,
        'description': 'Standard nod with bright yellow LED'
    },
    'blues': {
        'gesture': 'Nod',
        'speed': 0.6,
        'intensity': 0.7,
        'led_color': '#1E3A8A',
        'expression': 'Oh',
        'head_pitch': -0.1,
        'head_yaw': 0.0,
        'description': 'Slow sway with dark blue LED'
    },
    'metal': {
        'gesture': 'Shake',
        'speed': 1.8,
        'intensity': 2.0,
        'led_color': '#8B0000',
        'expression': 'Surprise',
        'head_pitch': 0.15,
        'head_yaw': 0.0,
        'description': 'Vigorous shake with dark red LED'
    },
    'disco': {
        'gesture': 'Nod',
        'speed': 1.3,
        'intensity': 1.3,
        'led_color': '#9333EA',
        'expression': 'Smile',
        'head_pitch': 0.1,
        'head_yaw': 0.0,
        'description': 'Active nod with purple LED'
    },
    'hiphop': {
        'gesture': 'Nod',
        'speed': 1.1,
        'intensity': 1.1,
        'led_color': '#00FF88',
        'expression': 'Smile',
        'head_pitch': 0.0,
        'head_yaw': 0.0,
        'description': 'Head bob with green LED'
    },
    'country': {
        'gesture': 'Shake',
        'speed': 0.8,
        'intensity': 0.9,
        'led_color': '#FF8C42',
        'expression': 'Smile',
        'head_pitch': 0.0,
        'head_yaw': 0.0,
        'description': 'Gentle shake with warm orange LED'
    },
    'reggae': {
        'gesture': 'Nod',
        'speed': 0.9,
        'intensity': 0.9,
        'led_color': '#9ACD32',
        'expression': 'Smile',
        'head_pitch': 0.0,
        'head_yaw': 0.0,
        'description': 'Relaxed sway with yellow-green LED'
    }
}

DEFAULT_BEHAVIOR = {
    'gesture': 'Nod',
    'speed': 1.0,
    'intensity': 1.0,
    'led_color': '#FFFFFF',
    'expression': 'Smile',
    'head_pitch': 0.0,
    'head_yaw': 0.0,
    'description': 'Default neutral behavior'
}
