import xml.etree.ElementTree as ET
from xml.dom.minidom import getDOMImplementation
from xml.dom.minidom import parse
import os


def xml_save(size, objects, save_path='./', filename= '5', floder_text = 'VOC2007',
             segmented_text = '0', difficult_text = '0', truncated_text = '0'):
    '''
    save the annotation of image to a .xml file with the VOC type
    :param size:  list of string  [H,W,C]
    :param objects: list of string [N,5]  [:,0:4]:bounding boxes  [:,4]:class
    :param save_path:
    :param filename:
    :param floder_text:
    :param segmented_text:
    :param difficult_text:
    :param truncated_text:
    :return:
    '''
    size_text=size
    filename_text=filename
    impl = getDOMImplementation()

    doc_new = impl.createDocument(None, "annotation", None)
    root = doc_new.documentElement

    floder=doc_new.createElement("floder")
    floder_textnode = doc_new.createTextNode(floder_text)
    floder.appendChild(floder_textnode)
    root.appendChild(floder)
    filename=doc_new.createElement("filename")
    filename_textnode = doc_new.createTextNode(filename_text+'.jpg')
    filename.appendChild(filename_textnode)
    root.appendChild(filename)
    segmented=doc_new.createElement("segmented")
    segmented_textnode = doc_new.createTextNode(segmented_text)
    segmented.appendChild(segmented_textnode)
    root.appendChild(segmented)

    size=doc_new.createElement("size")
    width=doc_new.createElement("width")
    width_textnode=doc_new.createTextNode(size_text[1])   # node contents must be a string
    width.appendChild(width_textnode)
    height=doc_new.createElement("height")
    height_textnode=doc_new.createTextNode(size_text[0])
    height.appendChild(height_textnode)
    depth=doc_new.createElement("depth")
    depth_textnode=doc_new.createTextNode(size_text[2])
    depth.appendChild(depth_textnode)
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    root.appendChild(size)

    for i in range(len(objects)):
        object=doc_new.createElement("object")
        bndbox=doc_new.createElement('bndbox')
        xmin=doc_new.createElement('xmin')
        xmin_textnode=doc_new.createTextNode(objects[i][0])
        xmin.appendChild(xmin_textnode)
        ymin=doc_new.createElement('ymin')
        ymin_textnode = doc_new.createTextNode(objects[i][1])
        ymin.appendChild(ymin_textnode)
        xmax=doc_new.createElement('xmax')
        xmax_textnode = doc_new.createTextNode(objects[i][2])
        xmax.appendChild(xmax_textnode)
        ymax=doc_new.createElement('ymax')
        ymax_textnode = doc_new.createTextNode(objects[i][3])
        ymax.appendChild(ymax_textnode)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
        name=doc_new.createElement('name')
        name_textnode = doc_new.createTextNode(objects[i][4])
        name.appendChild(name_textnode)
        difficult=doc_new.createElement('difficult')
        difficult_textnode = doc_new.createTextNode(difficult_text)
        difficult.appendChild(difficult_textnode)
        truncated=doc_new.createElement('truncated')
        truncated_textnode = doc_new.createTextNode(truncated_text)
        truncated.appendChild(truncated_textnode)
        object.appendChild(bndbox)
        object.appendChild(name)
        object.appendChild(difficult)
        object.appendChild(truncated)
        root.appendChild(object)

    save_file = os.path.join(save_path, filename_text+'.xml')
    with open(save_file, "w", encoding="utf-8") as f:
        doc_new.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")




# doc=parse("./test.xml")
# bookShelf1=doc.getElementsByTagName("bookShelf")[0]
# new_book=doc_new.createElement("book")
# new_book.setAttribute("isbn","******")
# new_book.setAttribute("date","2022-10-3")
# new_book.appendChild(doc_new.createTextNode("鲁迅全集"))
# root.appendChild(new_book)
# with open("new.xml", "w", encoding="utf-8") as f:
#     doc_new.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

if __name__ == '__main__':
    size=['2048','1024','3']
    objects=[['20','20','30','30','class'],['20','20','30','30','class']]
    xml_save(size = size, objects = objects)





