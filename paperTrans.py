import layoutparser as lp
import cv2
import numpy as np
import pytesseract
from mdutils.mdutils import MdUtils
import os, gc
from nltk.tokenize import word_tokenize
from pdf2image import convert_from_path
from tqdm import tqdm
from pygtrans import Translate

client = Translate()

model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
    label_map={
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure"
    })


class paperTrans:

    def __init__(self, path: str, leng: str = 'zh-tw') -> None:

        # extract elements(Figure、Title、Text) from paper
        self.path = path.replace('.pdf', '')
        self.leng = leng
        Images = convert_from_path(path, fmt='png')
        all_elements = []
        for Image in tqdm(range(len(Images))):
            if Image == 0:
                elements_, elements = self.toElements(np.array(Images[0]),
                                                      first=True,
                                                      path=self.path)
                all_elements += elements_
            else:
                elements_, elements = self.toElements(np.array(Images[Image]),
                                                      first=False,
                                                      path=self.path)
                all_elements += elements_
        self.title = all_elements[0][0]

        # translate elements to chinese
        all_elements_ = all_elements
        all_elements_ = self.elementsTrans(all_elements_)

        # write to Markdown :cn_version
        mdFile = MdUtils(file_name=self.path + '/' + self.title + '_cn' +
                         '.md',
                         title=self.title)
        for i in all_elements_[1:]:
            if i[1] == 'Text':
                mdFile.new_paragraph(i[0])
            elif i[1] == 'Title':
                mdFile.new_header(level=2,
                                  title=i[0],
                                  add_table_of_contents='n')
            elif i[1] == 'Figure':
                mdFile.new_line(mdFile.new_inline_image(text='', path=i[0]))
        mdFile.create_md_file()

    # OCR the text
    def detialVirtualize(self, element, image_):
        img_cv = image_[element[3]:element[1], element[0]:element[2]]
        cv2.imwrite('image_.png', img_cv)

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(img_rgb)

    # save the Image and return the path
    def saveImage(self, image, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            num = 0
        else:
            files = os.listdir(path)
            files = [int(i.replace('.png', '')) for i in files if '.png' in i]
            try:
                num = int(max(files)) + 1
            except:
                num = 0
        cv2.imwrite(path + '/' + str(num) + '.png', image)
        return str(num) + '.png'

    # turn paper to elements
    def toElements(self, image, first=False, path='output'):
        image_ = image
        image = image[..., ::-1]
        img_width = image.shape[1]
        layout = model.detect(image)
        lp.draw_box(image, layout, box_width=3)

        # remove overlap  text

        # lx = left top x-coordinate
        # fy = righe bottom y-coordinate ...etc
        def checkOverlap(lx1, ly1, rx1, ry1, lx2, ly2, rx2, ry2):
            if lx1 <= rx2 and rx1 >= lx2 and ly2 >= ry1 and ly1 >= ry2:
                return [
                    True,
                    [
                        min(lx1, lx2),
                        max(ly1, ly2),
                        max(rx1, rx2),
                        min(ry1, ry2), 'Text'
                    ]
                ]
            else:
                return [False, 0]

        # merge texts as Text
        texts = [z for z in layout if z.type == 'Text' or z.type == 'List']
        texts = [[
            int(x[3][0]),
            int(x[3][1]),
            int(x[1][0]),
            int(x[1][1]), 'Text'
        ] for x in [z.points for z in texts]]

        while True:
            break_ = True
            for i in range(len(texts) - 1):
                try:
                    Com = texts[i]
                except:
                    pass
                for beCom in range(i + 1, len(texts)):
                    beCom = texts[beCom]
                    state = checkOverlap(Com[0], Com[1], Com[2], Com[3],
                                         beCom[0], beCom[1], beCom[2],
                                         beCom[3])
                    if state[0] == True:
                        texts.remove(Com)
                        texts.remove(beCom)
                        texts.append(state[1])
                        break_ = False
                        break
            if break_ == True: break

        # merge figures and tables as Figures
        figures = [
            z for z in layout if z.type == 'Figure' or z.type == 'Table'
        ]
        figures = [[
            int(x[3][0]),
            int(x[3][1]),
            int(x[1][0]),
            int(x[1][1]), 'Figure'
        ] for x in [z.points for z in figures]]

        while True:
            break_ = True
            for i in range(len(figures) - 1):
                try:
                    Com = figures[i]
                except:
                    pass
                for beCom in range(i + 1, len(figures)):
                    beCom = figures[beCom]
                    state = checkOverlap(Com[0], Com[1], Com[2], Com[3],
                                         beCom[0], beCom[1], beCom[2],
                                         beCom[3])
                    if state[0] == True:
                        figures.remove(Com)
                        figures.remove(beCom)
                        figures.append(state[1])
                        break_ = False
                        break
            if break_ == True: break

        # merge all elements
        elements = texts + figures
        for i in layout:
            if i.type == 'Title':
                p = i.points
                elements.append([
                    int(p[3][0]),
                    int(p[3][1]),
                    int(p[1][0]),
                    int(p[1][1]), i.type
                ])

        # dilate elements
        for i in range(len(elements)):
            bias = 8
            elements[i][0] -= bias
            elements[i][1] += bias
            elements[i][2] += bias
            elements[i][3] -= bias

        # sort elements
        elements.sort(key=lambda i: i[1])

        element_l = []
        element_r = []
        for element in elements:
            width = int(element[2]) - int(element[0])
            center_x = (element[2] + element[0]) / 2

            if element[4] == 'Table':
                element[4] = 'Figure'

            if center_x > img_width / 2 and img_width / 2 > width > img_width / 4.2:
                element_l.append(element)
            elif element[4] == 'Title' and center_x > img_width / 1.8:
                element_l.append(element)
            else:
                element_r.append(element)

        elements = element_r + element_l

        # discriminator
        if first == True:
            elements[0][4] = 'Title'
            for i in range(len(elements)):
                image_ = image
                if self.detialVirtualize(
                        elements[i],
                        image_=image_).strip().lower() == 'abstract':
                    elements[i][4] = 'Title'
                    for r in range(1, i):
                        elements[r][4] = 'Text'
                    break

        for i in range(len(elements)):
            if elements[i][4] == 'Title':
                image_ = image
                ocr = self.detialVirtualize(elements[i], image_=image_)
                if len(word_tokenize(ocr)) > len(ocr) / 3:
                    elements[i][4] = 'Figure'

        # elements to elements
        image_ = image
        n_elements = []
        for i in elements:
            if i[4] == 'Title' or i[4] == 'Text':
                n_elements.append(
                    [self.detialVirtualize(i, image_=image_).strip(), i[4]])
            elif i[4] == 'Figure':
                file_name = self.saveImage(image[i[3]:i[1], i[0]:i[2]], path)
                n_elements.append([file_name, i[4]])

        return n_elements, elements

    # translate the elements
    def elementsTrans(self, elements):
        cn = []
        for i in elements:
            if i[1] == 'Text':
                cn.append(i[0])
        texts = client.translate(cn, target=self.leng)
        texts = [text.translatedText for text in texts]
        a = 0
        for i in (range(len(elements))):
            if elements[i][1] == 'Text':
                elements[i][0] = texts[a]
                a += 1

        return elements


if __name__ == '__name__':
    paperTrans('2205.14100.pdf')