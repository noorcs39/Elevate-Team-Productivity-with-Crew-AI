from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_ollama import OllamaLLM

# LLM setup (Ollama must be running)
llm = OllamaLLM(model="llama3")

# Tool setup
search_tool = SerperDevTool()

# 1. Researcher
researcher = Agent(
    role="Senior Research Analyst",
    goal="Track cutting-edge AI developments",
    backstory="An expert in researching AI trends and technologies.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

# 2. Writer
writer = Agent(
    role="Content Writer",
    goal="Create engaging newsletter content based on AI research",
    backstory="A skilled writer who turns complex research into accessible and compelling content.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llm,
    memory=None
)

# 3. Personalizer (NEW)
personalizer = Agent(
    role="Content Personalizer",
    goal="Craft personalized introductions tailored to audience segments",
    backstory="An expert in analyzing audience data and writing personalized introductions that increase engagement.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

# 4. Editor
editor = Agent(
    role="Newsletter Editor",
    goal="Polish and format newsletter with integrated personalized content",
    backstory="An editor with mastery of language, formatting, and integrating personalized touches for a compelling final product.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llm,
    memory=None
)

# Task 1: Research
research_task = Task(
    description="Research and summarize four trending topics in AI.",
    agent=researcher,
    expected_output="A list of 4 AI trends with brief explanations."
)

# Task 2: Write the article
writing_task = Task(
    description="Write a 500-word newsletter article on the 4 AI trends provided.",
    agent=writer,
    expected_output="A newsletter draft with 4 headers and a paragraph under each."
)

# Task 3: Personalize
personalization_task = Task(
    description="Analyze reader data and create a personalized introduction for the newsletter.",
    agent=personalizer,
    expected_output="A compelling introduction tailored to a specific audience segment (e.g., AI professionals)."
)

# Task 4: Edit and Finalize
editing_task = Task(
    description=(
        "Edit the newsletter content for grammar, clarity, and flow. Integrate the personalized intro at the top. "
        "Final format should include: personalized intro, 4 headers with 1 paragraph each, and polished grammar."
    ),
    agent=editor,
    expected_output="A finalized newsletter with a personalized opening, 4 headers, and well-formatted content."
)

# Create the Crew
crew = Crew(
    agents=[researcher, writer, personalizer, editor],
    tasks=[research_task, writing_task, personalization_task, editing_task],
    process=Process.sequential,
    verbose=True
)

# Run it
result = crew.kickoff()
print("\n✅ Final Personalized Newsletter:\n", result)
