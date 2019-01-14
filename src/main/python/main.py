from fbs_runtime.application_context import ApplicationContext
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import *

import sys
import os
import cv2
import csv

import numpy as np
import scipy.io
import bisect

INF = sys.maxsize

class QtCapture(QGroupBox):
    def __init__(self, *args):
        super(QGroupBox, self).__init__()

        self.setTitle('Video')
        self.fps = 60
        self.cap = cv2.VideoCapture(*args)

        self.video_frame = QLabel()
        self.video_frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay = QVBoxLayout()
        #lay.setMargin(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

    def setFPS(self, fps):
        self.fps = fps

    def getFPS(self):
        return self.fps

    def nextFrameSlot(self, yellow=False):
        ret, frame = self.cap.read()
        # OpenCV yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if yellow:
            frame *= np.array([1, 1, 0]).astype(np.uint8)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pix = pix.scaledToWidth(self.video_frame.width())
        self.video_frame.setPixmap(pix)

    def getTotalFrames(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def getFrame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def setFrame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def deleteLater(self):
        self.cap.release()
        super(QGroupBox, self).deleteLater()

class MouseTrack(QMainWindow):
    def __init__(self, parent = None):
        super(MouseTrack, self).__init__(parent)
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        self.resize(1000, 750)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.timer = QtCore.QTimer()

        self.setWindowModified(False)
        self.sessDir = QFileDialog.getExistingDirectory(self, 'Select Session')
        while not os.path.exists(os.path.join(self.sessDir, 'runAnalyzed.mat')):
            QMessageBox.warning(self, '', 'Invalid session', QMessageBox.Ok, QMessageBox.Ok)
            self.sessDir = QFileDialog.getExistingDirectory(self, 'Select Session')
        self.setWindowTitle(os.path.split(self.sessDir)[-1]+'[*] - WhiskLabeler')

        self.createToolBar()

        layout = QGridLayout()
        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.mediaPlayer = QtCapture(os.path.join(self.sessDir, 'runWisk.mp4'))
        self.totFrames = int(self.mediaPlayer.getTotalFrames())
        layout.addWidget(self.mediaPlayer, 0, 0, 3, 3)

        self.labels, self.trials = self.loadTrials()
        if self.labels == 0 and self.trials == 0:
            retval = QMessageBox.warning(self, '', 'Invalid saved training data. Start over?', QMessageBox.Cancel | QMessageBox.Ok, QMessageBox.Ok)
            if retval == QMessageBox.Cancel:
                self.close()
            else:
                os.remove(os.path.join(self.sessDir, 'whiskerLabels.csv'))
                self.labels, self.trials = self.loadTrials()
        self.listTrials = [i for i in self.trials]
        self.minFrame = 0
        self.isPlaying = False
        self.quitCommit = False

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, self.totFrames)
        self.progressBar.setTextVisible(True)
        self.progressBar.setValue(0)
        self.progressBar.setFormat('Frame: 0/'+str(self.totFrames))
        layout.addWidget(self.progressBar, 3, 0, 1, -1)

        self.trialProgress = QProgressBar()
        self.trialProgress.setRange(0, 100)
        self.trialProgress.setTextVisible(True)
        self.trialProgress.setValue(0)
        self.trialProgress.setFormat('Trial: 0/0/100')
        layout.addWidget(self.trialProgress, 4, 0, 1, -1)

        listGroup = QGroupBox('Trials')
        listLayout = QVBoxLayout()
        self.trialList = QListWidget()
        self.trialList.setFocusPolicy(QtCore.Qt.NoFocus)
        self.trialList.itemClicked.connect(lambda item: self.seekTrial(int(item.text().split(':')[0])))
        listLayout.addWidget(self.trialList)
        listGroup.setLayout(listLayout)
        layout.addWidget(listGroup, 0, 3, 2, 1)

        metadataGroup = QGroupBox('Trial Data')
        metadataLayout = QVBoxLayout()
        self.trialLabel = QLabel('Trial: 0')
        self.touchLabel = QLabel('Touch: ')
        self.totalTrials = QLabel('Total Trials: '+str(len(self.listTrials)))
        self.unlabeledTrials = QLabel('Unlabeled Trials: ')
        metadataLayout.addWidget(self.trialLabel)
        metadataLayout.addWidget(self.touchLabel)
        metadataLayout.addWidget(self.totalTrials)
        metadataLayout.addWidget(self.unlabeledTrials)
        metadataGroup.setLayout(metadataLayout)
        layout.addWidget(metadataGroup, 2, 3)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(3, 0)

        wid.setLayout(layout)

        self.trialIdxs = self.populateList()
        self.updateUnlabeled()

        self.seekTrial(self.findNextUnlabeled())

    def findNextUnlabeled(self, minT=-1):
        for trial in self.listTrials:
            if trial<=minT:
                continue
            if self.labels[trial][0] == -1:
                return trial
        return self.listTrials[0]

    def updateUnlabeled(self):
        self.unlabeledTrials.setText('Unlabeled Trials: '+str(sum([self.labels[i][0]==-1 for i in self.labels])))

    def seekTrial(self, trial):
        if trial not in self.listTrials:
            trial = findNextUnlabeled(trial)
        self.currTrial = trial
        self.stopPlaying()
        self.mediaPlayer.setFrame(self.trials[self.currTrial][0])
        self.minFrame = self.trials[self.currTrial][0]
        self.maxFrame = self.trials[self.currTrial][1]
        self.trialProgress.setRange(self.minFrame, self.maxFrame)
        self.trialProgress.setValue(self.minFrame)
        self.trialProgress.setFormat('Trial: '+str(self.minFrame)+'/'+str(self.minFrame)+'/'+str(self.maxFrame))
        self.trialList.setCurrentItem(self.trialList.item(self.labels[self.currTrial][1]))
        self.trialLabel.setText('Trial: '+str(self.currTrial))
        self.touchLabel.setText('Touch: '+('' if self.labels[self.currTrial][0]==-1 else str(self.labels[self.currTrial][0])))
        self.nextFrame()

    def labelTrial(self):
        self.labels[self.currTrial][0] = self.mediaPlayer.getFrame()-1
        self.trialList.item(self.labels[self.currTrial][1]).setBackground(QColor('#FFFF00'))
        self.trialList.item(self.labels[self.currTrial][1]).setText(str(self.currTrial)+': '+str(self.labels[self.currTrial][0]))
        self.touchLabel.setText('Touch: '+str(self.labels[self.currTrial][0]))
        self.updateUnlabeled()
        self.setWindowModified(True)
        self.reshow()

    def unlabelTrial(self):
        self.labels[self.currTrial][0] = -1
        self.trialList.item(self.labels[self.currTrial][1]).setBackground(QColor('#FFFFFF'))
        self.trialList.item(self.labels[self.currTrial][1]).setText(str(self.currTrial))
        self.touchLabel.setText('Touch: ')
        self.updateUnlabeled()
        self.setWindowModified(True)
        self.reshow()

    def populateList(self):
        trialIdxs = {}
        for i, trial in enumerate(self.labels):
            trialIdxs[i] = trial
            self.trialList.addItem(str(trial))
            if self.labels[trial][0] != -1:
                self.trialList.item(i).setBackground(QColor('#FFFF00'))
                self.trialList.item(i).setText(str(trial)+': '+str(self.labels[trial][0]))
        return trialIdxs

    def loadTrials(self):
        mat = scipy.io.loadmat(os.path.join(self.sessDir, 'runAnalyzed.mat'))
        obsOnTimes = np.squeeze(mat['obsOnTimes'])
        obsOnFrames = []
        clipLen = len(mat['frameTimeStampsWisk'])
        otherclipLen = len(mat['frameTimeStamps'])
        beforeAfter = True
        for i in range(len(mat['frameTimeStampsWisk'])):
            if np.isnan(mat['frameTimeStampsWisk'][i][0]):
                if beforeAfter:
                    mat['frameTimeStampsWisk'][i][0]=-1
                else:
                    mat['frameTimeStampsWisk'][i][0]=INF
            else:
                beforeAfter = False
        beforeAfter = True
        for i in range(len(mat['frameTimeStamps'])):
            if np.isnan(mat['frameTimeStamps'][i][0]):
                if beforeAfter:
                    mat['frameTimeStamps'][i][0] = -1
                else:
                    mat['frameTimeStampsWisk'][i][0]=INF
            else:
                beforeAfter = False
        firstIdx = bisect.bisect(np.squeeze(mat['frameTimeStamps']), -1)
        if firstIdx == clipLen:
            firstIdx = 0
        otherfirstIdx = bisect.bisect(np.squeeze(mat['frameTimeStampsWisk']), -1)
        if otherfirstIdx == otherclipLen:
            otherfirstIdx = 0

        lastIdx = -1
        otherlastIdx = -1

        while mat['frameTimeStamps'][lastIdx][0]==INF:
            lastIdx-=1
        while mat['frameTimeStampsWisk'][otherlastIdx][0]==INF:
            otherlastIdx-=1
            
        otherFirstTime = mat['frameTimeStamps'][firstIdx][0]
        firstTime = mat['frameTimeStampsWisk'][otherfirstIdx][0]
        for i, obsOnTime in enumerate(obsOnTimes):
            if not (obsOnTime>min(mat['frameTimeStampsWisk'][otherlastIdx][0], mat['frameTimeStamps'][lastIdx][0]) or obsOnTime<max(firstTime, otherFirstTime)):
                obsOnFrames.append((i, bisect.bisect_left(np.squeeze(mat['frameTimeStampsWisk']), obsOnTime)))
            else:
                obsOnFrames.append((i, -1))

        obsOffTimes = np.squeeze(mat['obsOffTimes'])
        obsOffFrames = []
        for i, obsOffTime in enumerate(obsOffTimes):
            if not (obsOffTime>min(mat['frameTimeStampsWisk'][otherlastIdx][0], mat['frameTimeStamps'][lastIdx][0]) or obsOffTime<max(firstTime, otherFirstTime)):
                obsOffFrames.append((i, bisect.bisect_left(np.squeeze(mat['frameTimeStampsWisk']), obsOffTime)))
            else:
                obsOffFrames.append((i, -1))

        if len(obsOnTimes) != len(obsOffTimes):
            raise ValueError('obsOnTimes and obsOffTimes have different lengths! Please check no trials have been skipped')

        temptrialFrames = list(zip(obsOnFrames, obsOffFrames))
        trialFrames = {}
        for temp in temptrialFrames:
            if temp[0][0] != temp[1][0]:
                raise ValueError('some trial got shifted somewhere, exiting')
            trialFrames[temp[0][0]] = (temp[0][1], temp[1][1])

        labels = {}
        if os.path.exists(os.path.join(self.sessDir, 'whiskerLabels.csv')):
            with open(os.path.join(self.sessDir, 'whiskerLabels.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if int(row[1])>self.totFrames:
                        print('out of bounds', int(row[1]))
                        return 0, 0
                    labels[int(row[0])] = [int(row[1]), i]
            if len(labels) != len(trialFrames):
                print('size mismatch')
                return 0, 0
        else:
            labels = {i: [-1, j] for j, i in enumerate(trialFrames)}

        return labels, trialFrames

    def createToolBar(self):
        toolBar = self.addToolBar('Toolbar')

        saveButton = QAction('&Save', self)
        saveButton.setShortcut('Ctrl+s')
        saveButton.triggered.connect(self.save)
        toolBar.addAction(saveButton)

        exitButton = QAction('&Exit', self)
        exitButton.triggered.connect(self.confirm)
        toolBar.addAction(exitButton)

        helpButton = QAction('&Help', self)
        helpButton.triggered.connect(self.help)
        toolBar.addAction(helpButton)

    def nextFrame(self):
        if self.mediaPlayer.getFrame()>self.maxFrame:
            self.stopPlaying()
            return
        self.progressBar.setValue(self.mediaPlayer.getFrame())
        self.progressBar.setFormat('Frame: '+str(int(self.mediaPlayer.getFrame()))+'/'+str(self.totFrames))
        self.trialProgress.setValue(self.mediaPlayer.getFrame())
        self.trialProgress.setFormat('Trial: '+str(self.minFrame)+'/'+str(int(self.mediaPlayer.getFrame()))+'/'+str(self.maxFrame))
        self.mediaPlayer.nextFrameSlot(self.mediaPlayer.getFrame()==self.labels[self.currTrial][0])

    def prevFrame(self):
        self.stopPlaying()
        if self.mediaPlayer.getFrame()<self.minFrame+2:
            return
        self.mediaPlayer.setFrame(self.mediaPlayer.getFrame()-2)
        self.nextFrame()

    def reshow(self):
        if self.mediaPlayer.getFrame() == 0:
            return
        self.mediaPlayer.setFrame(self.mediaPlayer.getFrame()-1)
        self.mediaPlayer.nextFrameSlot(self.mediaPlayer.getFrame()==self.labels[self.currTrial][0])

    def startPlaying(self):
        self.isPlaying = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.timer.start(1000./self.mediaPlayer.getFPS())

    def stopPlaying(self):
        self.isPlaying = False
        self.timer.stop()

    def togglePlaying(self):
        if self.isPlaying:
            self.stopPlaying()
        else:
            self.startPlaying()

    def save(self):
        with open(os.path.join(self.sessDir, 'whiskerLabels.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([[trial, self.labels[trial][0]] for trial in self.labels])
        self.setWindowModified(False)
        print("saved")

    def confirm(self):
        if not self.isWindowModified() or self.quitCommit:
            return True
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Save before quitting?')
        msg.setWindowTitle('')
        msg.setStandardButtons(QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes)
        msg.setEscapeButton(QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        retval = msg.exec_()

        if retval!=QMessageBox.Cancel:
            if retval==QMessageBox.Yes:
                self.save()
            self.quitCommit = True
            self.close()
            return True
        return False

    def closeEvent(self, event):
        if self.confirm():
            event.accept()
        else:
            event.ignore()

    def viewLabeled(self):
        if self.labels[self.currTrial][0] == -1:
            return
        self.stopPlaying()
        self.mediaPlayer.setFrame(self.labels[self.currTrial][0])
        self.nextFrame()

    def seekInTrial(self, key, zeroKey):
        key-=zeroKey
        key/=10
        self.mediaPlayer.setFrame(int(self.minFrame+key*(self.maxFrame-self.minFrame)))
        self.nextFrame()

    def help(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Help')
        msg.setText('Keyboard Shortcuts:')
        msg.setInformativeText("""Right arrow/L: Forward one frame
Left arrow/H: Back one frame
Up arrow: Move up one trial
Down arrow: Move down one trial
Space: Toggle Play/Pause
B: Back to beginning of trial
N: Next unlabeled trial
T: Label current frame
U: Unlabel trial
V: View labeled frame
0-9: Seek through trial
Ctrl+S: Save""")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.exec_()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Right or key == QtCore.Qt.Key_L:
            self.stopPlaying()
            self.nextFrame()
        elif key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_H:
            self.prevFrame()
        elif key == QtCore.Qt.Key_Space:
            self.togglePlaying()
        elif key == QtCore.Qt.Key_B:
            self.seekTrial(self.currTrial)
        elif key == QtCore.Qt.Key_N:
            self.seekTrial(self.findNextUnlabeled(self.currTrial))
        elif key == QtCore.Qt.Key_T:
            self.labelTrial()
        elif key == QtCore.Qt.Key_U:
            self.unlabelTrial()
        elif key == QtCore.Qt.Key_V:
            self.viewLabeled()
        elif key>=QtCore.Qt.Key_0 and key<=QtCore.Qt.Key_9:
            self.seekInTrial(key, QtCore.Qt.Key_0)
        elif key==QtCore.Qt.Key_Up:
            self.seekTrial(self.listTrials[(self.labels[self.currTrial][1]-1)%len(self.labels)])
        elif key==QtCore.Qt.Key_Down:
            self.seekTrial(self.listTrials[(self.labels[self.currTrial][1]+1)%len(self.labels)])

class AppContext(ApplicationContext):           # 1. Subclass ApplicationContext
    def run(self):                              # 2. Implement run()
        ex = MouseTrack()
        ex.show()
        return self.app.exec_()                 # 3. End run() with this line

if __name__ == '__main__':
    appctxt = AppContext()                      # 4. Instantiate the subclass
    exit_code = appctxt.run()                   # 5. Invoke run()
    sys.exit(exit_code)
