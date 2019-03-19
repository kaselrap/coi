import os, shutil
import numpy as np
import cv2
import timeit
import random
import json
import math
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
##Remove all images from folder
# folder = './static/images'
# for the_file in os.listdir(folder):
#     file_path = os.path.join(folder, the_file)
#     try:
#         if os.path.isfile(file_path):
#             os.unlink(file_path)
#         #elif os.path.isdir(file_path): shutil.rmtree(file_path)
#     except Exception as e:
#         print(e)

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule('/static/images/<filename>', 'uploaded_file',
                 build_only=True)

def contrast(grayscale):
    histogram = getHistogram(grayscale)
    height, width = grayscale.shape

    minH = np.amin(grayscale)
    maxH = np.amax(grayscale)

    final = np.zeros((height, width))

    for row in range(height):
        for col in range(width):
            final[row][col] = ((grayscale[row][col] - minH) / (maxH - minH) * 255)
    
    return final

def resize(image, src, power):
    height, width, level = image.shape

    global DIRECTION
    if power > 0:
        DIRECTION = 'up'
    else:
        DIRECTION = 'down'

    zoomHowMuch = abs(power)
    if zoomHowMuch == 1:
        cv2.imwrite(src, image)
        exit(0)

    newHeight = int(height / zoomHowMuch)
    newWidth = int(width / zoomHowMuch)
    zoomInImage = np.zeros((height, width, level))

    x = 0
    y = 0

    for row in range(height):
        x = 0
        for col in range(width):
            for zoom in range(zoomHowMuch):
                zoomInImage[y + zoom - 1][x + zoom - 1] = image[row][col]
                zoomInImage[y + zoom - 1][x+zoom] = image[row][col]
                zoomInImage[y+zoom][x + zoom - 1] = image[row][col]
                zoomInImage[y+zoom][x+zoom] = image[row][col]
                if (x + zoomHowMuch > width - zoomHowMuch):
                    break
                x += zoomHowMuch
        if (y + zoomHowMuch > height - zoomHowMuch):
            break
        y += zoomHowMuch


    zoomOutImage = np.zeros((height, width, level))

    x = 0
    y = 0

    for row in range(height):
        x = 0
        for col in range(width):
            for zoom in range(zoomHowMuch):
                zoomOutImage[int(newHeight/2) + row][int(newWidth/2) + col] = image[y][x]
            if (x + zoomHowMuch > width - zoomHowMuch):
                break
            x += zoomHowMuch
        if (y + zoomHowMuch > height - zoomHowMuch):
            break
        y += zoomHowMuch

    if DIRECTION == 'up':
        cv2.imwrite(src, zoomInImage)
    else:
        cv2.imwrite(src, zoomOutImage)

def brightness(image, value=1):
    return image * value

def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img

def gauss(image, power):
    height, width, ch = image.shape
    gaussian = np.random.randn(height, width, ch) * power
    noisy = image + gaussian

    return noisy

def salt_an_pepper(image):
    height, width, _ = image.shape

    for row in range(height):
        for col in range(width):
            if random.randint(0, 150) > 149:
                image[row][col] = 255
            elif random.randint(0, 150) < 1:
                image[row][col] = 0

    return image

def getImage(image):
    return cv2.imread(image)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getHistogram(array):
  height, width = array.shape
  countOfPixels = height * width
  histogram = np.zeros((height, width))

  countOfRepeat = np.zeros(256)

  for row in range(height):
    for col in range(width):
      if countOfRepeat[int(array[row][col])] < 255:
        countOfRepeat[int(array[row][col])] += 1

  return countOfRepeat

def median_filter(grayscaleImgae):
    height, width = grayscaleImgae.shape
    outfile = np.zeros(grayscaleImgae.shape)

    for row in range(1, height-1):
        for col in range(1, width-1):
            matrix = np.array([
                [grayscaleImgae[row-1][col-1], grayscaleImgae[row-1][col], grayscaleImgae[row-1][col+1]],
                [grayscaleImgae[row][col-1], grayscaleImgae[row][col], grayscaleImgae[row][col+1]],
                [grayscaleImgae[row+1][col-1], grayscaleImgae[row+1][col], grayscaleImgae[row+1][col+1]]
            ])

            sortedMatrix = np.sort(matrix, axis=None)
            outfile[row][col] = sortedMatrix[int(len(sortedMatrix)/2)]

    return outfile

def filter_previtta_roberts_sobelya(grayscaleImgae, type_algoritm) :
    height, width = grayscaleImgae.shape
    outfile = np.zeros(grayscaleImgae.shape)

    for row in range(1, height-1):
        for col in range(1, width-1):
            matrix = np.array([
                [grayscaleImgae[row-1][col-1], grayscaleImgae[row-1][col], grayscaleImgae[row-1][col+1]],
                [grayscaleImgae[row][col-1], grayscaleImgae[row][col], grayscaleImgae[row][col+1]],
                [grayscaleImgae[row+1][col-1], grayscaleImgae[row+1][col], grayscaleImgae[row+1][col+1]]
            ])

            if type_algoritm == 'previtta':
                Gx = (matrix[2][0] + matrix[2][1] + matrix[2][2]) - (matrix[0][0] + matrix[0][1] + matrix[0][2])
                Gy = (matrix[0][2] + matrix[1][2] + matrix[2][2]) - (matrix[0][0] + matrix[1][0] + matrix[2][0])
            elif type_algoritm == 'robertsa':
                Gx = (matrix[2][2] - matrix[1][1])
                Gy = (matrix[2][1] - matrix[1][2])
            elif type_algoritm == 'sobelya':
                Gx = (matrix[2][0] + (2 * matrix[2][1]) + matrix[2][2]) - (matrix[0][0] + (2 * matrix[0][1]) + matrix[0][2])
                Gy = (matrix[0][2] + (2 * matrix[1][2]) + matrix[2][2]) - (matrix[0][0] + (2 * matrix[1][0]) + matrix[2][0])
            
            sum = math.sqrt((Gx ** 2) + (Gy ** 2))

            if sum > 255:
                outfile[row][col] = 255
            elif sum < 0:
                outfile[row][col] = 0
            else:
                outfile[row][col] = sum
    return outfile

def mask_filter(grayscaleImgae, mask_type, depth_mask = '1') :
    global mask
    global add
    add = 0
    if str(mask_type) == 'high-frequency':
        if depth_mask == '5':
            mask = [
                0, -1, -1, -1, 0,
                -1, 0, -1, 0, -1,
                -1, -1, 17, -1, -1,
                -1, 0, -1, 0, -1,
                0, -1, -1, -1, 0,
            ]
        elif depth_mask == '7':
            mask = [
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, 49, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
            ]
        else:
            mask = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
        kern = 1
    elif str(mask_type) == 'low-frequency':
        if depth_mask == '5':
            mask = [
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ]
        elif depth_mask == '7':
           mask = [
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1,
            ]
        else:
            mask = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        kern = np.sum(mask)
    elif str(mask_type) == 'gaussian-blur':
        if depth_mask == '5':
            mask = [
                1, 2, 1, 1, 2,
                1, 2, 1, 1, 2,
                1, 2, 1, 1, 2,
                1, 2, 1, 1, 2,
                1, 2, 1, 1, 2,
            ]
        elif depth_mask == '7':
           mask = [
                1, 1, 1, 1, 1, 1, 1,
                1, 2, 2, 2, 2, 2, 1,
                1, 2, 1, 1, 1, 2, 1,
                1, 2, 1, 1, 1, 2, 1,
                1, 2, 1, 1, 1, 2, 1,
                1, 2, 2, 2, 2, 2, 1,
                1, 1, 1, 1, 1, 1, 1,
            ]
        else:
            mask = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        kern = np.sum(mask)
    elif str(mask_type) == 'embossing':
        mask = [
        0, -1, 0, 
        -1, 4, -1, 
        0, -1, 0]
        
        kern = 1
        add = 128
    elif str(mask_type) == 'vertical-linear':
        mask = [-3, -3, 5, -3, 0, 5, -3, -3, 5]
        kern = 1
    elif str(mask_type) == 'horizontal-linear':
        mask = [1, 2, 1, 0, 0, 0, -1, -2, -1]
        kern = 1
    elif str(mask_type) == 'diagonal':
        mask = [-3, -3, -3, 5, 0, -3, 5, 5, -3]
        kern = 1
    elif str(mask_type) == 'laplasa':
        mask = [1, 1, 1, 1, -8, 1, 1, 1, 1]
        kern = 1
    else:
        mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        kern = np.sum(mask) / 2

    height, width = grayscaleImgae.shape
    outfile = np.zeros(grayscaleImgae.shape)

    if depth_mask == '5':
        forrange = 2
    elif depth_mask == '7':
        forrange = 3
    else:
        forrange = 1
    
    for row in range(forrange, height - forrange):
        for col in range(forrange, width - forrange):
            if depth_mask == '5':
                matrix = np.array([
                    [grayscaleImgae[row-2][col-2], grayscaleImgae[row-2][col-1], grayscaleImgae[row-2][col], grayscaleImgae[row-2][col+1], grayscaleImgae[row-2][col+2]],
                    [grayscaleImgae[row-1][col-2], grayscaleImgae[row-1][col-1], grayscaleImgae[row-1][col], grayscaleImgae[row-1][col+1], grayscaleImgae[row-1][col+2]],
                    [grayscaleImgae[row][col-2], grayscaleImgae[row][col-1], grayscaleImgae[row][col], grayscaleImgae[row][col+1], grayscaleImgae[row][col+2]],
                    [grayscaleImgae[row+1][col-2], grayscaleImgae[row+1][col-1], grayscaleImgae[row+1][col], grayscaleImgae[row+1][col+1], grayscaleImgae[row+1][col+2]],
                    [grayscaleImgae[row+2][col-2], grayscaleImgae[row+2][col-1], grayscaleImgae[row+2][col], grayscaleImgae[row+2][col+1], grayscaleImgae[row+2][col+2]],
                ])

                pixelSum = np.sum([
                    matrix[0][0] * mask[0],
                    matrix[0][1] * mask[1],
                    matrix[0][2] * mask[2],
                    matrix[0][3] * mask[3],
                    matrix[0][4] * mask[4],
                    matrix[1][0] * mask[5],
                    matrix[1][1] * mask[6],
                    matrix[1][2] * mask[7],
                    matrix[1][3] * mask[8],
                    matrix[1][4] * mask[9],
                    matrix[2][0] * mask[10],
                    matrix[2][1] * mask[11],
                    matrix[2][2] * mask[12],
                    matrix[2][3] * mask[13],
                    matrix[2][4] * mask[14],
                    matrix[3][0] * mask[15],
                    matrix[3][1] * mask[16],
                    matrix[3][2] * mask[17],
                    matrix[3][3] * mask[18],
                    matrix[3][4] * mask[19],
                    matrix[4][0] * mask[20],
                    matrix[4][1] * mask[21],
                    matrix[4][2] * mask[22],
                    matrix[4][3] * mask[23],
                    matrix[4][4] * mask[24],
                ])
            elif depth_mask == '7':
                matrix = np.array([
                    [grayscaleImgae[row-3][col-3], grayscaleImgae[row-3][col-2], grayscaleImgae[row-3][col-1], grayscaleImgae[row-3][col], grayscaleImgae[row-3][col+1], grayscaleImgae[row-3][col+2], grayscaleImgae[row-3][col+3]],
                    [grayscaleImgae[row-2][col-3], grayscaleImgae[row-2][col-2], grayscaleImgae[row-2][col-1], grayscaleImgae[row-2][col], grayscaleImgae[row-2][col+1], grayscaleImgae[row-2][col+2], grayscaleImgae[row-2][col+3]],
                    [grayscaleImgae[row-1][col-3], grayscaleImgae[row-1][col-2], grayscaleImgae[row-1][col-1], grayscaleImgae[row-1][col], grayscaleImgae[row-1][col+1], grayscaleImgae[row-1][col+2], grayscaleImgae[row-1][col+3]],
                    [grayscaleImgae[row][col-3], grayscaleImgae[row][col-2], grayscaleImgae[row][col-1], grayscaleImgae[row][col], grayscaleImgae[row][col+1], grayscaleImgae[row][col+2], grayscaleImgae[row][col+3]],
                    [grayscaleImgae[row+1][col-3], grayscaleImgae[row+1][col-2], grayscaleImgae[row+1][col-1], grayscaleImgae[row+1][col], grayscaleImgae[row+1][col+1], grayscaleImgae[row+1][col+2], grayscaleImgae[row+1][col+3]],
                    [grayscaleImgae[row+2][col-3], grayscaleImgae[row+2][col-2], grayscaleImgae[row+2][col-1], grayscaleImgae[row+2][col], grayscaleImgae[row+2][col+1], grayscaleImgae[row+2][col+2], grayscaleImgae[row+2][col+3]],
                    [grayscaleImgae[row+3][col-3], grayscaleImgae[row+3][col-2], grayscaleImgae[row+3][col-1], grayscaleImgae[row+3][col], grayscaleImgae[row+3][col+1], grayscaleImgae[row+3][col+2], grayscaleImgae[row+3][col+3]],
                ])

                pixelSum = np.sum([
                    matrix[0][0] * mask[0],
                    matrix[0][1] * mask[1],
                    matrix[0][2] * mask[2],
                    matrix[0][3] * mask[3],
                    matrix[0][4] * mask[4],
                    matrix[0][5] * mask[5],
                    matrix[0][6] * mask[6],

                    matrix[1][0] * mask[7],
                    matrix[1][1] * mask[8],
                    matrix[1][2] * mask[9],
                    matrix[1][3] * mask[10],
                    matrix[1][4] * mask[11],
                    matrix[1][5] * mask[12],
                    matrix[1][6] * mask[13],

                    matrix[2][0] * mask[14],
                    matrix[2][1] * mask[15],
                    matrix[2][2] * mask[16],
                    matrix[2][3] * mask[17],
                    matrix[2][4] * mask[18],
                    matrix[2][5] * mask[19],
                    matrix[2][6] * mask[20],

                    matrix[3][0] * mask[21],
                    matrix[3][1] * mask[22],
                    matrix[3][2] * mask[23],
                    matrix[3][3] * mask[24],
                    matrix[3][4] * mask[25],
                    matrix[3][5] * mask[26],
                    matrix[3][6] * mask[27],

                    matrix[4][0] * mask[28],
                    matrix[4][1] * mask[29],
                    matrix[4][2] * mask[30],
                    matrix[4][3] * mask[31],
                    matrix[4][4] * mask[32],
                    matrix[4][5] * mask[33],
                    matrix[4][6] * mask[34],

                    matrix[5][0] * mask[35],
                    matrix[5][1] * mask[36],
                    matrix[5][2] * mask[37],
                    matrix[5][3] * mask[38],
                    matrix[5][4] * mask[39],
                    matrix[5][5] * mask[40],
                    matrix[5][6] * mask[41],

                    matrix[6][0] * mask[42],
                    matrix[6][1] * mask[43],
                    matrix[6][2] * mask[44],
                    matrix[6][3] * mask[45],
                    matrix[6][4] * mask[46],
                    matrix[6][5] * mask[47],
                    matrix[6][6] * mask[48],
                ])
            else:
                matrix = np.array([
                    [grayscaleImgae[row-1][col-1], grayscaleImgae[row-1][col], grayscaleImgae[row-1][col+1]],
                    [grayscaleImgae[row][col-1], grayscaleImgae[row][col], grayscaleImgae[row][col+1]],
                    [grayscaleImgae[row+1][col-1], grayscaleImgae[row+1][col], grayscaleImgae[row+1][col+1]]
                ])

                pixelSum = np.sum([
                    matrix[0][0] * mask[0],
                    matrix[0][1] * mask[1],
                    matrix[0][2] * mask[2],
                    matrix[1][0] * mask[3],
                    matrix[1][1] * mask[4],
                    matrix[1][2] * mask[5],
                    matrix[2][0] * mask[6],
                    matrix[2][1] * mask[7],
                    matrix[2][2] * mask[8]
                ])

            sum = (int(pixelSum) / kern) + add

            if sum > 255:
                outfile[row][col] = 255
            elif sum < 0:
                outfile[row][col] = 0
            else:
                outfile[row][col] = sum
    return outfile

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'my_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['my_image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            images = {}
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'normal_' + filename))
            arrayOfPixels = getImage(os.path.join(app.config['UPLOAD_FOLDER'],'normal_' + filename))
            grayScaleImage = rgb2gray(arrayOfPixels)
            select = request.form.get('task')
            if select == 'normal':
                return render_template(
                    'index.html', image = '/static/images/normal_' + filename)
            elif select == 'image-to-gray':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale_' + filename), grayScaleImage)  
                return render_template(
                    'index.html', image = '/static/images/grayscale_' + filename)
            elif select == 'otsu':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'otsu_' + filename), otsu(grayScaleImage))  
                return render_template(
                    'index.html', image = '/static/images/otsu_' + filename)
            elif select == 'contrast':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'contrast_' + filename), contrast(grayScaleImage))
                return render_template(
                    'index.html', image = '/static/images/contrast_' + filename)
            elif select == 'salt_and_pepper':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'salt_and_pepper_' + filename),
                    salt_an_pepper(arrayOfPixels))
                return render_template(
                    'index.html', image = '/static/images/salt_and_pepper_' + filename)
            elif select == 'gauss-noize':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'gauss'+ request.form.get('power') +'_' + filename),
                    gauss(arrayOfPixels, int(request.form.get('power'))))
                return render_template(
                    'index.html', image = '/static/images/gauss'+ request.form.get('power') +'_' + filename)
            elif select == 'brightness':
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'brightness_'+ request.form.get('power') +'_' + filename),
                brightness(arrayOfPixels, float(request.form.get('power'))))
                return render_template(
                    'index.html', image = '/static/images/brightness_'+ request.form.get('power') +'_' + filename)
            elif select == 'zoom':
                resize(
                    arrayOfPixels,
                    os.path.join(app.config['UPLOAD_FOLDER'], 'zoom-'+ request.form.get('power') +'-_' + filename),
                    int(request.form.get('power'))
                )
                return render_template(
                    'index.html', image = '/static/images/zoom-'+ request.form.get('power') +'-_' + filename)
            elif select == 'type-mask':
                if request.form.get('depth-mask') == '3':
                    depthMask = '1'
                elif request.form.get('depth-mask') == '5':
                    depthMask = '5'
                elif request.form.get('depth-mask') == '7':
                    depthMask = '7'
                else:
                    depthMask = '1'
                cv2.imwrite(
                    os.path.join(app.config['UPLOAD_FOLDER'], 'mask_-'+ request.form.get('mask') +'-depth-'+ depthMask +'-_' + filename),
                    mask_filter(grayScaleImage, request.form.get('mask'), depthMask)
                )

                return render_template(
                    'index.html',
                    image = '/static/images/mask_-'+ request.form.get('mask') +'-depth-'+ depthMask +'-_' + filename
                )
            elif select == 'previtta' or select == 'sobelya' or select == 'robertsa':
                cv2.imwrite(
                    os.path.join(app.config['UPLOAD_FOLDER'], select + '_' + filename),
                    filter_previtta_roberts_sobelya(grayScaleImage, select)
                )

                return render_template(
                    'index.html',
                    image = '/static/images/' + select + '_' + filename
                )
            elif select == 'median':
                cv2.imwrite(
                    os.path.join(app.config['UPLOAD_FOLDER'], 'median_' + filename),
                    median_filter(grayScaleImage)
                )

                return render_template(
                    'index.html',
                    image = '/static/images/median_' + filename
                )
            else:
                return render_template(
                    'index.html', image = '/static/images/normal_' + filename)
  
    return render_template(
        'index.html')

if __name__ == '__main__':
    app.run()