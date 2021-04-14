import pyzed.sl as sl


ID_COLORS = [(232, 176,59)
            ,(175, 208,25)
            ,(102, 205,105)
            ,(185, 0,255)
            ,(99, 107,252)]


# Slightly differs from sl.BODY_BONES in order to draw the spine
SKELETON_BONES = [ (sl.BODY_PARTS.NOSE, sl.BODY_PARTS.NECK),
                (sl.BODY_PARTS.NECK, sl.BODY_PARTS.RIGHT_SHOULDER),
                (sl.BODY_PARTS.RIGHT_SHOULDER, sl.BODY_PARTS.RIGHT_ELBOW),
                (sl.BODY_PARTS.RIGHT_ELBOW, sl.BODY_PARTS.RIGHT_WRIST),
                (sl.BODY_PARTS.NECK, sl.BODY_PARTS.LEFT_SHOULDER),
                (sl.BODY_PARTS.LEFT_SHOULDER, sl.BODY_PARTS.LEFT_ELBOW),
                (sl.BODY_PARTS.LEFT_ELBOW, sl.BODY_PARTS.LEFT_WRIST),
                (sl.BODY_PARTS.RIGHT_HIP, sl.BODY_PARTS.RIGHT_KNEE),
                (sl.BODY_PARTS.RIGHT_KNEE, sl.BODY_PARTS.RIGHT_ANKLE),
                (sl.BODY_PARTS.LEFT_HIP, sl.BODY_PARTS.LEFT_KNEE),
                (sl.BODY_PARTS.LEFT_KNEE, sl.BODY_PARTS.LEFT_ANKLE),
                (sl.BODY_PARTS.RIGHT_SHOULDER, sl.BODY_PARTS.LEFT_SHOULDER),
                (sl.BODY_PARTS.RIGHT_HIP, sl.BODY_PARTS.LEFT_HIP),
                (sl.BODY_PARTS.NOSE, sl.BODY_PARTS.RIGHT_EYE),
                (sl.BODY_PARTS.RIGHT_EYE, sl.BODY_PARTS.RIGHT_EAR),
                (sl.BODY_PARTS.NOSE, sl.BODY_PARTS.LEFT_EYE),
                (sl.BODY_PARTS.LEFT_EYE, sl.BODY_PARTS.LEFT_EAR) ]

def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))
        

def generate_color_id_u(idx):
    arr = []
    if(idx < 0):
        arr = [236,184,36,255]
    else:
        color_idx = idx % 5
        arr = [ID_COLORS[color_idx][0], ID_COLORS[color_idx][1], ID_COLORS[color_idx][2], 255]
    return arr
