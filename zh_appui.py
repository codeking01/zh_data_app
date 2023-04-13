# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'zh_appui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSizePolicy, QSpacerItem, QWidget)
import res_rc

class Ui_zh_data_app(object):
    def setupUi(self, zh_data_app):
        if not zh_data_app.objectName():
            zh_data_app.setObjectName(u"zh_data_app")
        zh_data_app.setEnabled(True)
        zh_data_app.resize(846, 487)
        font = QFont()
        font.setStrikeOut(True)
        font.setKerning(False)
        zh_data_app.setFont(font)
        zh_data_app.setCursor(QCursor(Qt.ArrowCursor))
        zh_data_app.setTabletTracking(False)
        zh_data_app.setContextMenuPolicy(Qt.NoContextMenu)
        icon = QIcon()
        icon.addFile(u"icons/tray.png", QSize(), QIcon.Normal, QIcon.Off)
        zh_data_app.setWindowIcon(icon)
        zh_data_app.setLayoutDirection(Qt.LeftToRight)
        zh_data_app.setStyleSheet(u"")
        self.horizontalLayout = QHBoxLayout(zh_data_app)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_4 = QGroupBox(zh_data_app)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.gridLayout_7 = QGridLayout(self.groupBox_4)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.groupBox_3 = QGroupBox(self.groupBox_4)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMinimumSize(QSize(0, 200))
        self.groupBox_3.setMaximumSize(QSize(400, 1000))
        self.groupBox_3.setStyleSheet(u"QGroupBox{\n"
"	background-color: rgb(255, 255, 255);\n"
"}")
        self.gridLayout_4 = QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.line = QFrame(self.groupBox_3)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_4.addWidget(self.line, 1, 0, 1, 1)

        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(180, 0))
        self.label.setMaximumSize(QSize(550, 65))
        font1 = QFont()
        font1.setFamilies([u"\u5e7c\u5706"])
        font1.setPointSize(20)
        font1.setBold(True)
        self.label.setFont(font1)
        self.label.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setIndent(-4)

        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(300, 16777215))
        self.label_4.setStyleSheet(u"background-color: rgb(85, 255, 255);")
        self.label_4.setTextFormat(Qt.PlainText)
        self.label_4.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout_4.addWidget(self.label_4, 2, 0, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox_3, 0, 0, 1, 1)

        self.groupBox_2 = QGroupBox(self.groupBox_4)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setStyleSheet(u"QGroupBox{\n"
"	background-color: rgb(255, 255, 255);\n"
"}")
        self.gridLayout_3 = QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalSpacer_10 = QSpacerItem(10, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_10, 1, 2, 1, 1)

        self.verticalSpacer_5 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_3.addItem(self.verticalSpacer_5, 4, 1, 1, 1)

        self.groupBox_8 = QGroupBox(self.groupBox_2)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setAlignment(Qt.AlignCenter)
        self.gridLayout_6 = QGridLayout(self.groupBox_8)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.horizontalSpacer_8 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_8, 2, 0, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_6, 2, 2, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_5, 0, 2, 1, 1)

        self.train_progressBar = QProgressBar(self.groupBox_8)
        self.train_progressBar.setObjectName(u"train_progressBar")
        self.train_progressBar.setCursor(QCursor(Qt.ArrowCursor))
        self.train_progressBar.setStyleSheet(u"")
        self.train_progressBar.setValue(0)

        self.gridLayout_6.addWidget(self.train_progressBar, 0, 4, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_7, 0, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox_8)
        self.label_2.setObjectName(u"label_2")
        font2 = QFont()
        font2.setFamilies([u"\u5e7c\u5706"])
        font2.setBold(True)
        self.label_2.setFont(font2)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout_6.addWidget(self.label_2, 0, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox_8)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font2)
        self.label_3.setFrameShape(QFrame.NoFrame)
        self.label_3.setFrameShadow(QFrame.Plain)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.gridLayout_6.addWidget(self.label_3, 2, 1, 1, 1)

        self.predict_progressBar = QProgressBar(self.groupBox_8)
        self.predict_progressBar.setObjectName(u"predict_progressBar")
        self.predict_progressBar.setMaximum(100)
        self.predict_progressBar.setValue(0)

        self.gridLayout_6.addWidget(self.predict_progressBar, 2, 4, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_8, 3, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_3.addItem(self.verticalSpacer_3, 0, 1, 1, 1)

        self.groupBox = QGroupBox(self.groupBox_2)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setCursor(QCursor(Qt.ArrowCursor))
        self.gridLayout_5 = QGridLayout(self.groupBox)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.train_stop_button = QPushButton(self.groupBox)
        self.train_stop_button.setObjectName(u"train_stop_button")
        font3 = QFont()
        font3.setStrikeOut(False)
        font3.setKerning(False)
        self.train_stop_button.setFont(font3)
        self.train_stop_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.train_stop_button.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.train_stop_button, 3, 2, 1, 1, Qt.AlignHCenter)

        self.groupBox_7 = QGroupBox(self.groupBox)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.gridLayout_2 = QGridLayout(self.groupBox_7)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.modeling_file_path = QLabel(self.groupBox_7)
        self.modeling_file_path.setObjectName(u"modeling_file_path")
        self.modeling_file_path.setMinimumSize(QSize(130, 0))
        self.modeling_file_path.setMaximumSize(QSize(150, 16777215))
        self.modeling_file_path.setFont(font3)
        self.modeling_file_path.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.modeling_file_path, 0, 1, 1, 1)

        self.load_modeling_files = QPushButton(self.groupBox_7)
        self.load_modeling_files.setObjectName(u"load_modeling_files")
        self.load_modeling_files.setMinimumSize(QSize(90, 0))
        self.load_modeling_files.setMaximumSize(QSize(100, 16777215))
        self.load_modeling_files.setCursor(QCursor(Qt.PointingHandCursor))
        self.load_modeling_files.setStyleSheet(u"")

        self.gridLayout_2.addWidget(self.load_modeling_files, 0, 0, 1, 1, Qt.AlignLeft)


        self.gridLayout_5.addWidget(self.groupBox_7, 3, 0, 1, 1)

        self.predict_stop_button = QPushButton(self.groupBox)
        self.predict_stop_button.setObjectName(u"predict_stop_button")
        self.predict_stop_button.setFont(font3)
        self.predict_stop_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.predict_stop_button.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.predict_stop_button, 5, 2, 1, 1, Qt.AlignHCenter)

        self.predict_button = QPushButton(self.groupBox)
        self.predict_button.setObjectName(u"predict_button")
        self.predict_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.predict_button.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.predict_button, 5, 1, 1, 1, Qt.AlignHCenter)

        self.groupBox_6 = QGroupBox(self.groupBox)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setMinimumSize(QSize(0, 0))
        self.groupBox_6.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout = QGridLayout(self.groupBox_6)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalSpacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.modeling_exist_file_path = QLabel(self.groupBox_6)
        self.modeling_exist_file_path.setObjectName(u"modeling_exist_file_path")
        self.modeling_exist_file_path.setMinimumSize(QSize(130, 0))
        self.modeling_exist_file_path.setMaximumSize(QSize(150, 16777215))
        self.modeling_exist_file_path.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.modeling_exist_file_path, 0, 1, 1, 2)

        self.predict_file_path = QLabel(self.groupBox_6)
        self.predict_file_path.setObjectName(u"predict_file_path")
        self.predict_file_path.setMinimumSize(QSize(130, 0))
        self.predict_file_path.setMaximumSize(QSize(150, 16777215))
        self.predict_file_path.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.predict_file_path, 2, 1, 1, 2)

        self.load_predict_files = QPushButton(self.groupBox_6)
        self.load_predict_files.setObjectName(u"load_predict_files")
        self.load_predict_files.setMinimumSize(QSize(90, 0))
        self.load_predict_files.setMaximumSize(QSize(100, 16777215))
        self.load_predict_files.setCursor(QCursor(Qt.PointingHandCursor))
        self.load_predict_files.setStyleSheet(u"")

        self.gridLayout.addWidget(self.load_predict_files, 2, 0, 1, 1, Qt.AlignLeft)

        self.load_modeling_exist_files = QPushButton(self.groupBox_6)
        self.load_modeling_exist_files.setObjectName(u"load_modeling_exist_files")
        self.load_modeling_exist_files.setEnabled(False)
        self.load_modeling_exist_files.setMinimumSize(QSize(90, 0))
        self.load_modeling_exist_files.setMaximumSize(QSize(100, 16777215))
        self.load_modeling_exist_files.setCursor(QCursor(Qt.ForbiddenCursor))
        self.load_modeling_exist_files.setStyleSheet(u"")

        self.gridLayout.addWidget(self.load_modeling_exist_files, 0, 0, 1, 1, Qt.AlignLeft)


        self.gridLayout_5.addWidget(self.groupBox_6, 5, 0, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_5.addItem(self.verticalSpacer_4, 4, 0, 1, 1)

        self.train_button = QPushButton(self.groupBox)
        self.train_button.setObjectName(u"train_button")
        self.train_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.train_button.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.train_button, 3, 1, 1, 1, Qt.AlignHCenter)


        self.gridLayout_3.addWidget(self.groupBox, 1, 1, 1, 1)

        self.verticalSpacer_6 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_3.addItem(self.verticalSpacer_6, 2, 1, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(10, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_4, 1, 0, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox_2, 0, 1, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_4)


        self.retranslateUi(zh_data_app)

        QMetaObject.connectSlotsByName(zh_data_app)
    # setupUi

    def retranslateUi(self, zh_data_app):
        zh_data_app.setWindowTitle(QCoreApplication.translate("zh_data_app", u"\u632f\u534e\u65b0\u6750\u6599\u6570\u636e\u5efa\u6a21\u5de5\u5177V1.0", None))
        self.groupBox_4.setTitle("")
        self.groupBox_3.setTitle("")
        self.label.setText(QCoreApplication.translate("zh_data_app", u"\u6b22\u8fce\u4f7f\u7528", None))
        self.label_4.setText(QCoreApplication.translate("zh_data_app", u"\u8bf4\u660e\uff1a", None))
        self.groupBox_2.setTitle("")
        self.groupBox_8.setTitle("")
        self.label_2.setText(QCoreApplication.translate("zh_data_app", u"\u8bad\u7ec3\u8fdb\u5ea6\u6761", None))
        self.label_3.setText(QCoreApplication.translate("zh_data_app", u"\u9884\u6d4b\u8fdb\u5ea6\u6761", None))
        self.groupBox.setTitle("")
        self.train_stop_button.setText(QCoreApplication.translate("zh_data_app", u"\u7ec8\u6b62", None))
        self.groupBox_7.setTitle("")
        self.modeling_file_path.setText(QCoreApplication.translate("zh_data_app", u"\u6682\u672a\u9009\u62e9", None))
        self.load_modeling_files.setText(QCoreApplication.translate("zh_data_app", u"\u9009\u62e9\u5efa\u6a21\u6587\u4ef6", None))
        self.predict_stop_button.setText(QCoreApplication.translate("zh_data_app", u"\u7ec8\u6b62", None))
        self.predict_button.setText(QCoreApplication.translate("zh_data_app", u"\u9884\u6d4b", None))
        self.groupBox_6.setTitle("")
        self.modeling_exist_file_path.setText(QCoreApplication.translate("zh_data_app", u"\u5df2\u6709\u8bad\u7ec3\u597d\u7684\u6a21\u578b", None))
        self.predict_file_path.setText(QCoreApplication.translate("zh_data_app", u"\u6682\u672a\u9009\u62e9", None))
        self.load_predict_files.setText(QCoreApplication.translate("zh_data_app", u"\u9009\u62e9\u9884\u6d4b\u6587\u4ef6", None))
        self.load_modeling_exist_files.setText(QCoreApplication.translate("zh_data_app", u"\u66ff\u6362\u5df2\u6709\u6a21\u578b", None))
        self.train_button.setText(QCoreApplication.translate("zh_data_app", u"\u8bad\u7ec3", None))
    # retranslateUi

