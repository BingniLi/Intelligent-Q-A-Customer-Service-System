import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import do_load
from operations.search import do_search, do_get_answer
from operations.count import do_count
from operations.drop import do_drop
from logs import LOGGER
from encode import SentenceModel
from starlette.staticfiles import StaticFiles
import csv

MODEL = SentenceModel()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

def read_csv(filename):
    column1 = []
    column2 = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # 假设文件至少有两列数据
                column1.append(row[0])
                column2.append(row[1])

    return column1, column2

# 示例用法
filename = '/root/lbn/data/test.csv'  # 替换为你的 CSV 文件路径
q_list, t_list = read_csv(filename)

def calculate_accuracy(question, answers, table_name = None):
    correct_count = 0
    total_count = len(question)

    for question, answer in zip(question, answers):
        _, predicted_answer, _ = do_get_answer(table_name, question, MODEL, MILVUS_CLI, MYSQL_CLI)
        if predicted_answer == answer:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    return accuracy


accuracy = calculate_accuracy(q_list, t_list)
print("Accuracy:%.*f"%(2,accuracy),'%')

