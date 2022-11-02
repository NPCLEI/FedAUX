import time
from threading import Thread

begain_write_num_line = 10000
"""
    定义一开始输出多少行才开始写入.log文件,避免频繁写入log文件
"""
inited = False
global_title = "npc report"

class NPCLogTitleContext:
    def __init__(self,context_title = "",showEndTime = True) -> None:
        global global_title
        self.past_title = global_title
        self.context_title = context_title
        self.showEndTime = showEndTime
        if showEndTime:
            self.startTime = time.time() # log user-training start time

    def __enter__(self):
        global global_title
        NPCBlank()
        global_title = self.context_title
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global global_title
        # if self.showEndTime:
        #     NPCLog()
        NPCBlank()
        global_title = self.past_title


def curTime():
    return time.asctime()

def logger_wirter_func():
    global logger_file,logger_file_fp,global_print2file_control,global_logger_queue,begain_write_num_line
    while True:
        time.sleep(0.5)
        if len(global_logger_queue) > begain_write_num_line:
            break
    with open(logger_file,"a+") as logger_file_fp:
        while True:
            if len(global_logger_queue) > 0:
                logger_file_fp.writelines(global_logger_queue)
                global_logger_queue = []
                logger_file_fp.flush()
            time.sleep(0.001)

def NPCReport(*inputs,idf = "",end = "\n"):

    print("%s[%s npc report]"%(idf,curTime()))
    for ipt in inputs:
        print(ipt,end="")

    print(end=end)

def NPCLine(char = '-',num = 120):
    NPCLog(char * num,title="")

def NPCBlank(line_num = 1):
    for _ in range(line_num):
        NPCLog(title="")

def NPCLog(*inputs,idf = "",end = "\n",title = True,flush = False):
    ###################if you want to remove this package,you can just relpace 'NPCLog' to 'print'
    if not inited:
        print(*inputs,end = end)
        return

    global global_print2file_control,logger_file_fp,global_logger_queue,global_title_control,global_title

    if title and global_title_control:
        print("%s[%s %s] "%(idf,curTime(),global_title),end="",flush=flush)
        global_logger_queue.append("%s[%s npc report] "%(idf,curTime()))
    for ipt in inputs:
        print(ipt,end="",flush=flush)
        global_logger_queue.append(str(ipt))

    print(end=end,flush=True)
    global_logger_queue.append(end)


def initLogger(envir_path,record = True):
    
    global logger_file,logger_file_fp,global_print2file_control,global_logger_queue,logger_wirter_thread,global_title_control,global_title,inited
    global_print2file_control = record
    if not record:
        return
    logger_file = "%s/Log/logger_%s.log"%(envir_path,curTime().replace(":","_"))
    # logger_file = "./Log/logger_Thu Jun 30 12_42_52 2022.log"
    global_logger_queue = []
    logger_wirter_thread = Thread(target=logger_wirter_func,daemon=True)
    logger_wirter_thread.start()
    global_title_control = True
    NPCLog(title="")
    NPCLog(title="")
    NPCLog(" <&> logger:NPC at your service.",title=False)
    NPCLog(title="")
    inited = True

if __name__ == "__main__":
    
    initLogger()
    NPCLog("logger:NPC at your service.")
    NPCLog("logger:NPC at your service.")
    NPCLog("logger:NPC at your service.")
    time.sleep(1)