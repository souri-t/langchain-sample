import openai
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
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


tools_node = ToolNode([mutiply_numbers, add_numbers, get_current_time])

## ツールが必要かどうかを判断する関数
def decide_next(state: MessagesState) -> str:
    # 直近のメッセージを取得
    last_message = state["messages"][-1]

    # 直近のメッセージにtool_callsがあればツールノードへ、なければendへ
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools_edge"
    return "end"


## LLMノードを実行する関数
def call_llm(state: MessagesState) -> str:
    system_prompt = "ユーザーの入力に応じて、mutiply_numbersとadd_numbersから適切なツールを使って答えてください"
    
    # 現在のメッセージ履歴を取得し、システムプロンプトが一つもなければ新規追加する（システムプロンプトが入るのは最初の一回のみ）
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages

    # LLMを呼び出し
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=API_KEY, openai_api_base=BASE_URL)
    response = llm.bind_tools([mutiply_numbers, add_numbers]).invoke(messages)

    return {"messages": messages + [response]}


## エージェント実行関数
def build() -> any:
    # Graph構築
    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools_node", tools_node)
    
    # エントリーポイントと繋がるエッジを設定
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", decide_next, {"tools_edge": "tools_node", "end": "__end__"})
    graph.add_edge("tools_node", "llm")
    app = graph.compile()
    
    return app


def run_agent(user_input: str, app: any) -> str:
    response = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)
    return response

# app.get_graph()の画像を保存する関数
def save_structure(app = None):
    if app:
        with open("graph_structure.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    user_input = "12と7を掛け算して、その後に16を足して"
    app = build()
    response = run_agent(user_input, app)
    save_structure(app)
    