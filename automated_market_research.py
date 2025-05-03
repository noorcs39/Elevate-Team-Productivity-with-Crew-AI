from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_ollama import OllamaLLM

# LLM setup (Ollama must be running)
llm = OllamaLLM(model="llama3")

# Tools
search_tool = SerperDevTool()

# --- AGENTS ---

# 1. Data Gatherer
data_gatherer = Agent(
    role="Market Data Gatherer",
    goal="Collect accurate and recent market data for a product category",
    backstory="An expert in web research and data collection, focused on gathering actionable market intelligence.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 2. Analysis Agent
analyst = Agent(
    role="Market Analyst",
    goal="Analyze collected market data to extract trends, competitor strategies, and customer demographics",
    backstory="A strategic thinker with strong analytical skills and experience in evaluating market research data.",
    tools=[],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 3. Report Generator
report_generator = Agent(
    role="Research Report Generator",
    goal="Create a concise and insightful market research report",
    backstory="A skilled technical writer who converts analysis into clear and informative reports.",
    tools=[],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASKS ---

# Task 1: Gather data
task_gather = Task(
    description="Research recent trends, top competitors, pricing, and customer demographics for the wearable tech industry.",
    expected_output="A structured summary of trends, competitors, pricing models, and customer profiles.",
    agent=data_gatherer
)

# Task 2: Analyze data
task_analyze = Task(
    description="Analyze the gathered data to identify market opportunities, threats, and key differentiators.",
    expected_output="An analysis highlighting market gaps, opportunities, and competitive landscape.",
    agent=analyst
)

# Task 3: Generate report
task_report = Task(
    description="Write a 300-word market research report summarizing the findings and analysis for a business audience.",
    expected_output="A well-formatted market research report with clear insights.",
    agent=report_generator
)

# --- CREW SETUP ---
crew = Crew(
    agents=[data_gatherer, analyst, report_generator],
    tasks=[task_gather, task_analyze, task_report],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
result = crew.kickoff()
print("\n📊 Final Market Research Report:\n", result)
