import cv2


def transfer_color(src, dest,pixel_clip = False):
    if (pixel_clip):
        src, dest = src.clip(0,255), dest.clip(0,255)
        
   
    w,h,_ = src.shape 
    dest_resize = cv2.resize(dest, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    
    dest_gray = cv2.cvtColor(dest_resize, cv2.COLOR_BGR2GRAY) 
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)  
    src_yiq[...,0] = dest_gray                         
    
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)