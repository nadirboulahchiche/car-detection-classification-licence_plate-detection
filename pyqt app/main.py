import detect_vehicules
import classify
import get_license

##
##path = 'car2.jpg'
##
##detect_vehicules.get_car(path)
##classify.get_class('object.jpg')
##get_licence.get_plate('object.jpg')

from PySide2.QtWidgets import QMainWindow, QApplication,QFileDialog
from PySide2.QtGui import QImage,QPixmap
from PySide2.QtCore import QTimer,QAbstractTableModel,Qt,QCoreApplication
from ui_main import Ui_MainWindow
import sys
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # load image
        self.ui.load_image.clicked.connect(self.read_image)
        

    def read_image(self):
        filename = QFileDialog.getOpenFileName()                                 # get the path of the file in list
        detect_vehicules.get_car(filename[0])
        self.image = cv2.imread('cache_image/small_object.jpg')
        ##self.image = cv2.resize(self.image , (461, 261))
        self.image = QImage(self.image.data, self.image.shape[1], self.image.shape[0],QImage.Format_RGB888).rgbSwapped()
        self.ui.image_frame.setPixmap(QPixmap.fromImage(self.image))
        classe = classify.get_class('cache_image/object.jpg')
        self.ui.car_class.setText(classe)
        licence = get_license.get_plate('cache_image/object.jpg')
        self.ui.licence_plate.setText(licence)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
