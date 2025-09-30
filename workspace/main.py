import openai
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from datetime import datetime
import pytz

API_KEY = ""
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "x-ai/grok-4-fast:free"
    
## 掛け算をする関数
@tool
def mutiply_numbers(a: int, b: int) -> int:
  """2つの数字を掛け算して返す"""
  return a * b

## 足し算をする関数
@tool
def add_numbers(a: int, b: int) -> int:
  """2つの数字を足し算して返す"""
  return a + b

## タイムゾーンを指定するとその地域の現在時刻を取得する関数
@tool
def get_current_time(timezone: str) -> str:
    """指定されたタイムゾーンの現在時刻を返す"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"タイムゾーンエラー: {e}"

def execute():
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # GPTのテスト
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "こんにちは！"}],
    )
    print(response.choices[0].message.content)

def execute2():
    model = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
    )

    response = model.invoke("こんにちは！")
    print(response.content)

## エージェント実行関数
def run_agent(user_input: str) -> str:
    # ツールを定義
    tools = [mutiply_numbers, add_numbers, get_current_time]

    # エージェントを作成
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=API_KEY, openai_api_base=BASE_URL)
    system_prompt = "ユーザーの入力に対して、適切なツールを使って答えてください"
    agent = create_react_agent(llm, tools, prompt=system_prompt)

    # ユーザーの入力を実行
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)

    return response

if __name__ == "__main__":
    # response = run_agent("12と7を掛け算して")
    response = run_agent("米国の現在時刻を教えて")
    # print(response)