from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Explicitly set the backend before importing pyplot
import matplotlib.pyplot as plt

import seaborn as sns
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    df: pd.DataFrame  # Holds the dataset throughout the graph

llm = ChatOpenAI(
    model='gpt-4o'
)

@tool
def calculate_statistics(df, col_name: Union[str, list[str]], group_by: Union[None, str, list[str]] = None) -> dict:
    """This function calculates statistics on a column denoted by col_name of a dataset denoted by df"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    col_name = [col_name] if type(col_name) == str else col_name
    if group_by:
        if type(group_by) == str:
            grouped = df.groupby(group_by)[col_name]
            stats = {
            'mean': grouped.mean().round(5).to_dict(),
            'median': grouped.median().round(5).to_dict(),
            'max': grouped.max().round(5).to_dict(),
            'min': grouped.min().round(5).to_dict(),
            'stddev': grouped.std().round(5).to_dict()}
        
        elif type(group_by) == list:
            grouped = df.groupby(group_by)[col_name]
            stats = {
            'mean': grouped.mean().round(5).reset_index().to_dict(orient='records'),
            'median': grouped.median().round(5).reset_index().to_dict(orient='records'),
            'max': grouped.max().round(5).reset_index().to_dict(orient='records'),
            'min': grouped.min().round(5).reset_index().to_dict(orient='records'),
            'stddev': grouped.std().round(5).reset_index().to_dict(orient='records')}

    else:  
        stats = {
            'mean': df[col_name].mean().round(5).to_dict(),
            'median': df[col_name].median().round(5).to_dict(),
            'max': df[col_name].max().round(5).to_dict(),
            'min': df[col_name].min().round(5).to_dict(),
            'stddev': df[col_name].std().round(5).to_dict()}
    
    return stats

@tool
def generate_plot(df, x_column: str, y_column: Union[str, list[str]], plot: str, group_by: Union[None, str, list[str]] = None, title: Union[None, str] = None):
    """This function plots the data using pandas .plot() function and columns provided as parameters using the plot type deemed most logical for the scenarios by the LLM"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if x_column not in df.columns:
        return f"❌ x_column '{x_column}' not found in the dataset."

    if isinstance(y_column, str):
        if y_column not in df.columns:
            return f"❌ y_column '{y_column}' not found in the dataset."
    elif isinstance(y_column, list):
        invalid_cols = [col for col in y_column if col not in df.columns]
        if invalid_cols:
            return f"❌ y_column(s) {invalid_cols} not found in the dataset."
    else:
        return "❌ y_column must be a string or list of strings."
    
    plot = plot.lower().strip()
    plot_type_mapping = {
        # Line plot
        "line": "lineplot",
        "lineplot": "lineplot",
        "linplot": "lineplot",
        "lin": "lineplot",

        # Bar plot
        "bar": "barplot",
        "barplot": "barplot",
        "barchart": "barplot",

        # Scatter plot
        "scatter": "scatterplot",
        "scatterplot": "scatterplot",

        # Histogram
        "hist": "histplot",
        "histogram": "histplot",
        "histo": "histplot",

        # Box plot
        "box": "boxplot",
        "boxplot": "boxplot",
        "boxchart": "boxplot",

        # # Heatmap
        # "heatmap": "heatmap",
        # "correlation matrix": "heatmap"
    }

    func_map = {
        'lineplot': lambda: sns.lineplot(data=df, x=x_column, y=y_column),
        'barplot': lambda: sns.barplot(data=df, x=x_column, y=y_column, estimator="mean"),
        'scatterplot': lambda: sns.scatterplot(data=df, x=x_column, y=y_column),
        'boxplot': lambda: sns.boxplot(data=df, x=x_column, y=y_column),
        'histplot': lambda: sns.histplot(data=df, x=x_column, bins=10, kde=True)
    }

    if plot not in plot_type_mapping:
        return 'The required plot is not supported yet!'
    
    plot = plot_type_mapping[plot]

    if group_by:
        if plot == 'scatterplot':
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=group_by)
            plt.title(title)
            plt.show()
            # plt.savefig(f'{title}.png')
            return
        elif plot == 'lineplot':
            sns.lineplot(data=df, x=x_column, y=y_column, hue=group_by)
            plt.title(title)
            plt.show()
            # plt.savefig(f'{title}.png')
            return
        else:
            return 'Required Plot not supported with grouping!'

    
    func_map[plot]()
    plt.title(title)
    plt.show()
    # plt.savefig(f'{title}.png')
    return

tools = [calculate_statistics, generate_plot]

llm = llm.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content='''You are a Data Insights Agent that helps analyze tabular data (typically CSV or DataFrame).
        You are equipped with two powerful tools:
        1. `calculate_statistics`: Use this to compute summary statistics (mean, median, min, max, stddev), optionally grouped by one or more columns.
        2. `generate_plot`: Use this to generate visualizations such as bar plots, line plots, scatter plots, histograms, and box plots.
        📌 Your responsibilities:
        - Decide which tool to use based on the user request.
        - Always ask for required details: column names, groupings, or plot type if not clearly provided.
        - Validate input where necessary (e.g., ensure column names exist in the dataset).
        - Never answer analytical questions directly — always use the tools provided.
        - If the user asks for an insight or analysis (e.g., “Which region performs best?”), use `calculate_statistics` to compare values.
        - If a task is required to be done for multiple columns/ rows you can run the tool multiple times for individual columns since the tool inputs sometimes require individual columns
        ❌ Do not attempt to analyze without data or use knowledge beyond what's in the provided dataset.
        ✅ If you're unsure what the user wants, politely ask clarifying questions.
        You are focused, analytical, and precise — like a data analyst powered by Python tools.'''
    )

    # Add a preview of the dataset (first 5 rows only) as a SystemMessage
    df_preview = SystemMessage(content="📊 Data Preview:\n" + state["df"].head().to_markdown())
    print("Available columns:", state['df'].columns)
    # Combine everything and pass to the LLM
    response = llm.invoke([system_prompt, df_preview] + state["messages"])
    return {'messages': [response], 'df': state['df']}


def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node('Model',model_call)
graph.add_node('Tools',ToolNode(tools))
graph.add_conditional_edges(
    'Model',
    should_continue,
    {
        'continue':'Tools',
        'end': END
    }
)
graph.add_edge('Tools','Model')
graph.add_edge(START,'Model')
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def print_conversation(conversation):
    for msg in conversation["messages"]:
        if isinstance(msg, AIMessage) and msg.content.strip():  # Only print non-empty AI responses
            print(f"\n🤖 AI: {msg.content}\n")

user_input = input("🧑 Human: ")
df = pd.read_csv(r'folder_path/retail_data.csv')

while user_input != "exit":
    # response = app.invoke({"messages": [HumanMessage(content=user_input)], 'df': df})
    # print_conversation(response)
    print_stream(app.stream({"messages": [HumanMessage(content=user_input)], 'df':df}, stream_mode="values"))
    user_input = input("🧑 Human: ")

plt.close('all')
