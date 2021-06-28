import glob

import cv2
import json
import os


# from https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
class BoundingBoxWidget(object):
    def __init__(self, img):
        self.original_image = cv2.imread(img)
        self.clone = self.original_image.copy()

        self.bounding_boxes = []

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))

            x = self.image_coordinates[0][0]
            y = self.image_coordinates[0][1]
            w = self.image_coordinates[1][0] - self.image_coordinates[0][0]
            h = self.image_coordinates[1][1] - self.image_coordinates[0][1]

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36, 255, 12), 2)

            self.bounding_boxes.append([x, y, w, h])

            cv2.imshow("image", self.clone)

            # print(face_16_32_64)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_bounding_boxes(self):
        return self.bounding_boxes


def manually_annotate_character_colour_bbox(paths, out_file_path):
    """
    Method to manually annotate characters, bounding boxes for characters and background colour.
    How to annotate:
        1. Mark bounding boxes of characters, left to right
        2. Press 'n' to move onto text annotation
        0. Input character numbers as they appear left to right in the comic, separated by a space
        4. Put a comma in between the characters and the background colour,
            input background colour as abbreviated letter (see printed options)
    N.B. Order is important, annotate left to right for bounding boxes and characters.
        If there is no character, don't input a bounding box.
    :param paths:
    :param out_file_path:
    :return:
    """
    result = {}

    for i, img_path in enumerate(paths):
        bounding_box_widget = BoundingBoxWidget(img_path)
        while True:
            cv2.imshow('image', bounding_box_widget.show_image())
            key = cv2.waitKey(1)

            # after annotating bounding boxes, click 'n' to do character and color annotation
            # you can also continue without annotating bounding boxes if no characters present
            if key == ord('n'):
                print("Available characters:\n"
                      "Dilbert: 1\n"
                      "Dogbert: 2\n"
                      "Boss: 3\n"
                      "CEO: 4\n"
                      "Wolly: 5\n"
                      "Alice: 6\n"
                      "Catbert: 7\n"
                      "Asok: 8\n"
                      "Non-recurring/main Character: 9\n"
                      "Building: 0")

                # what to do about non-recurring characters?
                # building as character
                print("Availble background colours:\n"
                      "yellow: y\n"
                      "green: g:\n"
                      "purple: p\n"
                      "blue: b\n"
                      "pink: pi\n"
                      "white: w")
                # brown??

                char_and_colour = input(f"{i + 1}/{len(paths)}: character character etc., background_colour: ")
                try:
                    cc_list = char_and_colour.split(",")
                    chars = cc_list[0].split(" ")
                    colour = cc_list[1]

                    filename = os.path.basename(img_path)

                    # json will have only filename as key instead of filepath as Ben had previously
                    result[filename] = {"Characters": chars, "Colour": colour,
                                        "Bounding boxes": bounding_box_widget.return_bounding_boxes()}
                    break

                except:
                    print("you messed up, saving results til now")
                    with open(out_file_path, "w+") as out_file:
                        json.dump(result, out_file)

                    raise Exception("you messed up, saving results til now")

                cv2.destroyAllWindows()
                break

    with open(out_file_path, "w+") as out_file:
        json.dump(result, out_file)

    exit(1)


if __name__ == '__main__':
    # here define the filepath to dilbert images in your file directory
    # imgs_path = '../data/scraped_images_dilbert/*'
    # out_file_path = '../data/annotations.json'

    imgs_path = "dataset/resized/resized/*"

    out_file_path = "dataset/resized/resized_char_and_colour_darwin.json"
    paths = glob.glob(imgs_path)
    paths = sorted(paths)
    # here define how many images you want to annotate
    # ben selected = paths[0:750]
    # darwin selected = paths[750:1500]
    # bartek selected = paths[1500:2250}
    selected = paths[1004:1500]
    print(selected[0])

    manually_annotate_character_colour_bbox(selected, out_file_path)