# 31 Facial Regions from Paper (Table 4) using MediaPipe 468 landmarks
# Reference: sensors-21-07923-v2.pdf

ALL_REGIONS = {
    # Region 0
    'upper_medial_forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378],
    
    # Region 1
    'right_upper_lateral_forehead': [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58],
    
    # Region 2
    'left_upper_lateral_forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
    
    # Region 3
    'lower_medial_forehead': [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400],
    
    # Region 4
    'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    
    # Region 5
    'left_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    
    # Region 6
    'right_temporal_lobe': [127, 234, 93, 132, 58, 172, 136, 150, 149, 176],
    
    # Region 7
    'left_temporal_lobe': [356, 454, 323, 361, 288, 397, 365, 379, 378, 400],
    
    # Region 8
    'right_lower_lateral_forehead': [21, 54, 103, 67, 109, 10, 151, 9, 8, 168],
    
    # Region 9
    'left_lower_lateral_forehead': [251, 284, 332, 297, 338, 10, 9, 8, 168, 6],
    
    # Region 10
    'glabella': [9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94],
    
    # Region 11
    'upper_nasal_dorsum': [168, 6, 197, 195, 5, 4],
    
    # Region 12
    'right_mid_nasal_sidewall': [98, 97, 2, 326, 327, 294, 278, 344, 440, 275],
    
    # Region 13
    'left_mid_nasal_sidewall': [327, 326, 2, 97, 98, 64, 48, 115, 220, 45],
    
    # Region 14
    'right_lower_nasal_sidewall': [294, 278, 344, 440, 275, 4, 5, 195, 197],
    
    # Region 15
    'left_lower_nasal_sidewall': [64, 48, 115, 220, 45, 4, 5, 195, 197],
    
    # Region 16
    'lower_nasal_dorsum': [1, 4, 5, 195, 197, 6, 168],
    
    # Region 17
    'nasal_tip': [1, 2, 98, 97, 327, 326],
    
    # Region 18
    'left_upper_lip': [267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    
    # Region 19
    'right_upper_lip': [37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    
    # Region 20
    'philtrum': [37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    
    # Region 21
    'lower_nasal_sidewall': [64, 48, 115, 220, 45, 294, 278, 344, 440, 275],
    
    # Region 22
    'right_nasolabial_fold': [36, 142, 126, 217, 198, 209, 49, 129, 203, 205, 50, 101],
    
    # Region 23
    'left_nasolabial_fold': [266, 371, 355, 437, 420, 429, 279, 358, 423, 425, 280, 330],
    
    # Region 24
    'chin': [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338],
    
    # Region 25
    'right_marionette_fold': [57, 186, 92, 165, 167, 164, 393, 391, 322, 410],
    
    # Region 26
    'left_marionette_fold': [287, 410, 322, 391, 393, 164, 167, 165, 92, 186],
    
    # Region 27
    'right_malar': [116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 203, 206, 216],
    
    # Region 28
    'left_malar': [345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 423, 426, 436],
    
    # Region 29
    'right_lower_cheek': [205, 50, 101, 116, 117, 118, 119, 120, 121, 214, 212, 216, 206, 203],
    
    # Region 30
    'left_lower_cheek': [425, 280, 330, 345, 346, 347, 348, 349, 350, 434, 432, 436, 426, 423],
}
# Landmark groups for SpO2 studies
# Reference: apply 5 models and compare skin vs skin+face
SKIN_FACE_REGIONS = list(ALL_REGIONS.keys())

# Face only (non-skin regions): Eyes, nasal tip, temporal lobes (often noisy or have specific dynamics)
FACE_REGIONS = ['right_eye', 'left_eye', 'nasal_tip', 'right_temporal_lobe', 'left_temporal_lobe']

# Skin only: Exclude Face regions
SKIN_REGIONS = [k for k in ALL_REGIONS.keys() if k not in FACE_REGIONS]
