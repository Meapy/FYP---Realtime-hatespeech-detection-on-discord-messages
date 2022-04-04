import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def setup_image(img):
    # convert the image to gray scale
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray
    #only keep the white pixels in the image
    img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
    # save the thresh holded image
    cv2.imwrite("data/images/output.png", img)
    return img

def convert_to_text(img):
    # convert the image to text
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    text = pytesseract.image_to_string(img, config='--oem 3 --psm 12')
    return text