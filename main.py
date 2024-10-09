from vanna.openai import OpenAI_Chat
from openai import OpenAI
from fastapi import FastAPI
from vanna.vannadb import VannaDB_VectorStore
import os

MY_VANNA_MODEL = os.environ["MY_VANNA_MODEL"]
vanna_api_key = os.environ["vanna_api_key"]
api_key = os.environ["api_key"]



class MyVanna(VannaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        VannaDB_VectorStore.__init__(self, vanna_model=MY_VANNA_MODEL, vanna_api_key=vanna_api_key, config=config)
        OpenAI_Chat.__init__(self, client=OpenAI(base_url="https://api.groq.com/openai/v1",api_key=api_key), config=config) # Make sure to put your AzureOpenAI client here

vn = MyVanna(config={'model': "llama3-8b-8192"})

vn.connect_to_sqlite('BallByBall.db')

def sql(question):
    return vn.generate_sql(question,allow_llm_to_see_data=True)

def df(sql_q):
    return vn.run_sql(sql_q)    

app = FastAPI()

@app.get("/question")
def sql_q(query: str):
    r = sql(query)
    d = df(r).to_string()
    return [d,r]