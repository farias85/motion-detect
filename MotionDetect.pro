#-------------------------------------------------
#
# Project created by QtCreator 2015-11-20T07:26:08
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = MotionDetect
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


LIBS += -LC:\OpenCV2.2\lib \
    -lopencv_core220 \
    -lopencv_highgui220 \
    -lopencv_video220 \
    -lopencv_imgproc220

INCLUDEPATH += C:\OpenCV2.2\include\

SOURCES += main.cpp \
   kalman.cpp

HEADERS += \
   kalman.hpp
