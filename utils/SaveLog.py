"""
@Author: 幻想
@Date: 2022/04/23 11:20
"""
import os


class SaveLog:
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path

    def savelog(self, text):
        self.f = open(self.path, 'a')
        self.f.write(text + '\n')
        self.f.flush()
        self.f.close()
        # print("Save Success!")
