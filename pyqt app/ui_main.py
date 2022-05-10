# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_mainIAzsDK.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *
import sys


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(640, 480)
        MainWindow.setStyleSheet(u"background-color: rgb(85, 255, 255);")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.image_frame = QLabel(self.centralwidget)
        self.image_frame.setObjectName(u"image_frame")
        self.image_frame.setGeometry(QRect(110, 20, 461, 261))
        self.image_frame.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 290, 181, 41))
        font = QFont()
        font.setFamily(u"Plantagenet Cherokee")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setStyleSheet(u"background-color: rgb(85, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.car_class = QLabel(self.centralwidget)
        self.car_class.setObjectName(u"car_class")
        self.car_class.setGeometry(QRect(210, 290, 361, 41))
        self.car_class.setFont(font)
        self.car_class.setStyleSheet(u"background-color: rgb(85, 255, 255);\n"
"background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.car_class.setFrameShape(QFrame.Box)
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 340, 191, 41))
        self.label_3.setFont(font)
        self.label_3.setStyleSheet(u"background-color: rgb(85, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.licence_plate = QLabel(self.centralwidget)
        self.licence_plate.setObjectName(u"licence_plate")
        self.licence_plate.setGeometry(QRect(210, 340, 361, 41))
        self.licence_plate.setFont(font)
        self.licence_plate.setStyleSheet(u"background-color: rgb(85, 255, 255);\n"
"background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.licence_plate.setFrameShape(QFrame.Box)
        self.load_image = QPushButton(self.centralwidget)
        self.load_image.setObjectName(u"load_image")
        self.load_image.setGeometry(QRect(220, 400, 181, 51))
        self.load_image.setFont(font)
        self.load_image.setFocusPolicy(Qt.NoFocus)
        self.load_image.setStyleSheet(u"background-color: rgb(0, 94, 245);")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.image_frame.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"car class :", None))
        self.car_class.setText("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"License plate :", None))
        self.licence_plate.setText("")
        self.load_image.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
    # retranslateUi

##if __name__ == "__main__":
##    app = QApplication(sys.argv)
##    window = Ui_MainWindow()
##    sys.exit(app.exec_())

