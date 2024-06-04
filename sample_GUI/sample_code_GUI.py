from AIGamePlatform import Othello

app=Othello() # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw

from promise import Promise
from threading import Thread
from PyQt5.QtWidgets import QApplication
import GUI
gui=None

@app.competition(competition_id='[competition ID]')
def _callback_(board, color): # 函數名稱可以自訂，board是當前盤面，color代表黑子或白子
    def promise_solver(resolve, reject):
        def show_GUI():
            global gui
            if gui is None:
                QApp=QApplication([])
                gui=GUI.ReversiUI(board, color, resolve)
                QApp.exec_()
            else:
                gui.renew_board(board, color, resolve)
        Thread(target=show_GUI).start()
    promise = Promise(promise_solver)
    action=promise.get()
    return action # 回傳要落子的座標

