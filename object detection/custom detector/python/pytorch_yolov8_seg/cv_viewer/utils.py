import cv2
import numpy as np
import pyzed.sl as sl

id_colors = [(232, 176,59),
            (175, 208,25),
            (102, 205,105),
            (185, 0 ,255),
            (99, 107,252)]


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
        arr = [id_colors[color_idx][0], id_colors[color_idx][1], id_colors[color_idx][2], 255]
    return arr

def draw_vertical_line(left_display, start_pt, end_pt, clr, thickness):
    n_steps = 7
    pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps
         , ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
    pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps
         , (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]
    
    cv2.line(left_display, (int(start_pt[0]),int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
    cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]),int(end_pt[1])), clr, thickness)

def draw_mask(target_img, mask, color, roi_shift=(0,0), line_thickness=2):
    overlay_roi_binary = (mask != 0).astype(np.uint8) * 255
    contours,_ = cv2.findContours(overlay_roi_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour[:,:,0] += int(roi_shift[0])
        contour[:,:,1] += int(roi_shift[1])
    # draw contours in `left_display` to avoid blending
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    cv2.drawContours(target_img, contours, -1, color, line_thickness, cv2.LINE_AA)