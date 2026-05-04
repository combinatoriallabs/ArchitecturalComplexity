global LOG_LEVEL 
LOG_LEVEL = 1 #0: INFO, 1: NOTICE, 2: WARNING, 3: ERROR
global PRINT_LEVEL
PRINT_LEVEL = 0 #0: INFO, 1: NOTICE, 2: WARNING, 3: ERROR
global INFO
global NOTICE 
global WARNING 
global ERROR
global SUPRESS
INFO = 0
NOTICE = 1
WARNING = 2
ERROR = 3
SUPRESS = -1

MAX_SIZE_GB = 1.0

GLOG = None #Init a logger to this variable anywhere that includes Logger stuff to enable printf

import os
import datetime
import inspect

def printf(strMessage: str, iLevel: int = 0) -> None:
    frame = inspect.stack()[1]
    caller = inspect.getframeinfo(frame[0])
    if GLOG is not None:
        GLOG.Log(strMessage, caller.filename.split("/")[-1], str(caller.lineno), iLevel)
    elif iLevel >= PRINT_LEVEL:
        print(strMessage)
    return
    
class Logger():
    def __init__(self, strLogPath: str = "./app.log"):
        '''
        Simple logging class. By default creates an app.log file in PWD. Pass another path to change location. 
        All Arch submodule classes will automatically send messages to a global instance of this class called GLOG if it exists.
        You can use the included printf() function to acheive the same functionality. 
        '''
        self.strLogPath = strLogPath
        if self.strLogPath[-4:] != ".log":
            self.strLogPath += ".log"
        
        if not os.path.exists(self.strLogPath): open(self.strLogPath, "a").close()

        if (os.path.getsize(self.strLogPath) / 1e9) > MAX_SIZE_GB:
            print("Truncating log file {}".format(self.strLogPath))
            with open(self.strLogPath, "r") as f:
                vecLines = f.readlines()
            idx = int(0.1 * len(vecLines))
            vecLines = vecLines[idx:]
            with open(self.strLogPath, "w") as f:
                f.writelines(vecLines)

        self.FILE = open(self.strLogPath, "a")
            
        self.mapLevelToStr = [
            "[INFO]",
            "[NOTICE]",
            "[WARNING]",
            "[ERROR]",
        ]
            
        self.FILE.write("---------------------------------------\n")
        self.FILE.write("Start of logging session: " + datetime.datetime.now().strftime("%Y %m/%d %H:%M:%S") + "\n")
            
        return
    
    def Log(self, strMessage: str, strFile: str, strLine: str, iLevel: int = 0) -> None:
        if iLevel >= PRINT_LEVEL:
            print(strMessage)
        
        strMessage = self.mapLevelToStr[iLevel] + " " + strFile + ":" + strLine + ": " + strMessage
        
        if iLevel >= LOG_LEVEL:
            self.FILE.write(strMessage + "\n")
            
        return
            
    def __del__(self):
        self.FILE.write("End of session\n")
        self.FILE.close()
        
        return