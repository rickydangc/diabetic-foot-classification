# Importing required libs
# Importing required libs
import cv2
import numpy as np
from skimage.measure import label, regionprops
from flask import Flask, render_template, request, Response
from model import *
import io
from PIL import Image
import base64

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    # try:
    if request.method == 'POST':
        img = preprocess_img(request.files['file'].stream)

        # image read
        image_rgb = img.copy()

        # skin thresholding
        mask = segment_skin(image_rgb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # hole filling
        mask = morphology.binary_fill_holes(mask[:, :, 1])

        # image labeling by object size
        labels = label(mask)
        max_index = biggestCC_index(labels)

        mask = border(mask, labels, max_index)
        #plt.imsave("./results/mask.jpeg", mask)

        # desired skin for each channel R, G and B
        skin = np.zeros_like(image_rgb)
        bord = image_rgb
        for i in range(0, 3):
            skin[:, :, i] = mask * image_rgb[:, :, i]
            bord[:, :, i] = image_rgb[:, :, i] * (mask == 0)

        #plt.imsave("./results/skin.jpg", skin)
        # plt.imsave("./results/bord.jpg", bord)

        skin_bgr = skin[..., ::-1]

        # fungus detection masks
        necrosis, fibrin, granulation = fungus_detection(skin_bgr)
        print(skin.shape)
        # process image
        final, total_n, total_f, total_g, total = process(skin, necrosis, fibrin, granulation)
        segmented = final

        # print(total_n,total_f,total_g)

        # write
        h, w, t = skin.shape
        n = "Percentage of necrosis: " + str(round((total_n / total) * 100, 2)) + "%"
        f = "Percentage of fibrin: " + str(round((total_f / total) * 100, 2)) + "%"
        g = "Percentage of granulation: " + str(round((total_g / total) * 100, 2)) + "%"

        # add the text to the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(segmented, g, (50, h - 100), font, 3, color, 10, cv2.LINE_AA)
        cv2.putText(segmented, f, (50, h - 200), font, 3, color, 10, cv2.LINE_AA)
        cv2.putText(segmented, n, (50, h - 300), font, 3, color, 10, cv2.LINE_AA)

        # Convert array to image
        img1 = Image.fromarray(segmented)
        # Create an in-memory buffer to hold the image file
        buffer = io.BytesIO()
        # Save image to buffer
        img1.save(buffer, format='JPEG')
        # img1.show()
        buffer.seek(0)
        # segmented image + original
        # plt.imsave("./results/image_result.jpg", segmented)
        # plt.imshow(segmented)
        # plt.figure()
        # print(buffer)
        str_equivalent_image = base64.b64encode(buffer.getvalue()).decode()
        buffer_path = "data:image/jpeg;base64," + str_equivalent_image
        pred = [n, f, g]
        # response = Response(buffer, content_type='image/jpeg')
        return render_template("result.html", predictions=pred, imagesend=buffer_path)

    # except:
    #     error = "File cannot be processed."
    #     return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
