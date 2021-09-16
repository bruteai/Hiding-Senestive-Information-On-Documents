#!/usr/bin/env python
# coding: utf-8
#import1
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import pytesseract
import sys
from geometry import Rect
import getopt
import os
print('python', sys.version_info)
print('opencv', cv2.__version__)
print('tesseract', pytesseract.get_tesseract_version())


def load_image(input_name, verbose=False, vis=False, dilate=1, erode=1, thresh=210, otsu=False, figsize=(10, 18)):
    im = plt.imread(input_name)
    if verbose: print('orig', im.shape, np.min(im), np.max(im))
    if vis:
        _, ax = plt.subplots(3, 1, figsize=figsize, dpi=300)
        ax[0].imshow(im)
        
    im_grey = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
    if verbose: print('gray', im_grey.shape, np.min(im_grey), np.max(im_grey))
    if vis:
        ax[1].imshow(im_grey, 'gray')
    
    if otsu:
        im_bw = cv2.threshold((im_grey * 255).astype(np.uint8), thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        im_bw = cv2.threshold((im_grey * 255).astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw = cv2.erode(im_bw, np.ones((erode, erode)))
    im_bw = cv2.dilate(im_bw, np.ones((dilate, dilate)))
    if verbose: print('bw', im_bw.shape, np.min(im_bw), np.max(im_bw))
    if vis:
        ax[2].imshow(im_bw, 'gray')
        #ax[2].set_ylim((250, 0))
    return im, im_grey, im_bw

def get_boxes(image):
    boxes = pytesseract.image_to_boxes(image, lang='kor', 
                                       config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789:-.',
                                       output_type='dict')
    if len(boxes['top']) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(boxes)
    h, w = image.shape[:2]
    df.top = h - df.top
    df.bottom = h - df.bottom
    df['h'] = df.bottom - df.top
    df['w'] = df.right - df.left
    #df = df.query('h >= 5 and w >= 5')
    df.sort_values(by=['top', 'bottom', 'left', 'right'], inplace=True)
    return df

def get_rects(df):
    rects = []
    for _, row in df.iterrows():
        l, w, t, h = (row[k] for k in ['left', 'w', 'top', 'h'])
        rects.append(Rect(l, t, w, h))
    return rects

def join(r1, r2):
    l = min(r1.l_top.x, r2.l_top.x)
    t = min(r1.l_top.y, r2.l_top.y)
    r = max(r1.r_bot.x, r2.r_bot.x)
    b = max(r1.r_bot.y, r2.r_bot.y)
    return Rect(l, t, r - l, b - t)

def group_rects(rects, min_dist=8):
    joined = []
    used = []
    for i in range(15):
        for r in rects:
            if r in used:
                continue
            added = False
            for i, j in enumerate(joined):
                if (r.overlaps_with(j)) or (r.distance_to_rect(j) < min_dist):
                    new_j = join(j, r)
                    joined[i] = new_j
                    added = True
                    used.append(r)
                    break
            if not added:
                joined.append(r)
                used.append(r)
    
    new_joined = []
    new_used = []
    for i in range(5):
        for r in joined:
            if r in used:
                continue
            added = False
            for i, j in enumerate(new_joined):
                if (r.overlaps_with(j)) or (r.distance_to_rect(j) < min_dist):
                    new_j = join(j, r)
                    new_joined[i] = new_j
                    added = True
                    new_used.append(r)
                    break
            if not added:
                new_joined.append(r)
                new_used.append(r)
    
    final_joined = [j for j in new_joined if j.width > 5 and j.height > 5]
    return final_joined

def split_by_height(joined, max_height=15):
    res = []
    for j in joined:
        if j.height > max_height:
            for i in range(j.height // max_height):
                res.append(Rect(j.l_top.x, j.l_top.y + i * max_height , 
                                j.width, max_height))
            res.append(Rect(j.l_top.x, j.r_bot.y - max_height , 
                                j.width, max_height))
        else:
            res.append(j)
    return res

def is_good_text(txt, white='0123456789-.'):
    text = "".join([c if c in white else '_' for c in txt]).strip('_')
    #print(text)
    if len(text) not in range(13, 17):
        return False
    if ((text.count('-') == 1)
         and (text.index('-') in [5, 6, 7])):
        return True
    if ((text.count('.') == 1)
         and (text.index('.') in [5, 6, 7])):
        return True
    return False


def get_id_rect(joined, image, verbose=False, vis_rects=False, margin=2):
    res = []
    for j in joined:
        t, b, l, r = (j.l_top.y - margin, 
                      j.r_bot.y + margin, 
                      j.l_top.x - margin, 
                      j.r_bot.x + margin)
        im_small = image[t:b, l:r]
        if vis_rects:
            plt.imshow(im_small, 'gray')
            plt.show()
        try:
            text = pytesseract.image_to_string(im_small, lang='kor',
                                       config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789-.:')
            text = text.replace('\n', '').replace('\r', '')
            if verbose:
                print(text)
            if is_good_text(text):
                res.append((j, text))
                continue

            text = pytesseract.image_to_string(im_small, lang='eng',
                                       config='--psm 7 --oem 3') #-c tessedit_char_whitelist=0123456789-.:')
            text = text.replace('\n', '').replace('\r', '')
            if verbose:
                print(text)
            if is_good_text(text):
                res.append((j, text))
            
        except:
            print('err in get_id_rect')
    return res

def blur_rect_on_image(image, rect, save_fpath=None, margin=2):
    if image.shape[2] == 3:
        im_res = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGRA)
    else:
        im_res = image.copy()
    t, b, l, r = (rect.l_top.y - margin, 
                  rect.r_bot.y + margin, 
                  rect.l_top.x - margin, 
                  rect.r_bot.x + margin)
    im_res[t:b, l:r] = 0
    im_res[t:b, l:r, 3] = 0.2
    if save_fpath is not None:
        plt.imsave(save_fpath, im_res)
    return im_res

def blur(input_name, output_name):
    im_res = None
    found_1 = False
    for dilate in [1, 2]:
        for otsu in [False, True]:
            for thresh in range(190, 236, 10):
                try:
                    #print(otsu, dilate, thresh)
                    im, im_grey, im_bw = load_image(input_name, verbose=False, vis=False, 
                                                    otsu=otsu, dilate=dilate, thresh=thresh)  
                    if im_res is None:
                        im_res = im.copy()
                    df = get_boxes(im_bw)
                    if len(df) == 0:
                        continue
                    rects = get_rects(df)
                    joined = group_rects(rects)
                    joined = split_by_height(joined, max_height=12)
                    for res_rect, txt in get_id_rect(joined, im_bw, verbose=False, vis_rects=False):
                        if res_rect is not None:
                            im_res = blur_rect_on_image(im_res, res_rect, save_fpath=None)
                            print('success:', txt, otsu, dilate, thresh)
                            found_1 = True
                except:
                    print('err in blur')
    if not found_1:
        print('fail')
    else:
        if output_name is not None:
            plt.imsave(output_name, im_res)
    return im_res


helpline = """Usage:
blur.py -i <input image> -o <output image>
default output image "<input image>_out.png"

blur.py -h print this helpline"""

def main(argv):
    output_file = None
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print(helpline) 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpline)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    if output_file is None:
        output_file = input_file + '_out.png'
    blur(input_file, output_file)
    

if __name__ == "__main__":
    main(sys.argv[1:])         



