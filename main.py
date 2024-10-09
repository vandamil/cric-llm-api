from vanna.openai import OpenAI_Chat
from openai import OpenAI
from fastapi import FastAPI
from vanna.vannadb import VannaDB_VectorStore

class MyVanna(VannaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        MY_VANNA_MODEL = 'sel_1'
        VannaDB_VectorStore.__init__(self, vanna_model=MY_VANNA_MODEL, vanna_api_key='390bc303830b4105b5a27b0e9517434a', config=config)
        OpenAI_Chat.__init__(self, client=OpenAI(base_url="https://api.groq.com/openai/v1",api_key='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'), config=config) # Make sure to put your AzureOpenAI client here

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