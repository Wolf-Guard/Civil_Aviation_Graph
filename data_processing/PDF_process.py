import fitz
import os

def readpdfname():
    pdf_files = []
    name = os.listdir('pdf')
    for filename in name:
        pdf_text = 'pdf/'+filename
        pdf_files.append(pdf_text)
    return pdf_files


def get_pdf_picture():
    count = 0
    for file in readpdfname():
        count = count + 1
        doc = fitz.open(file)
        imgcount = 0
        for page in doc:
            imageList = page.getImageList()
            print(imageList)
            for imginfo in imageList:
                pix = fitz.Pixmap(doc,imginfo[0])
                pix.writePNG(os.path.join('pdf_picture/t'+str(count)+'_{}.png'.format(imgcount)))
                imgcount +=1






if __name__ == '__main__':
    readpdfname()
    get_pdf_picture()

