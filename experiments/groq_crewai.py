from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew


# Load environment variable
load_dotenv()

app = FastAPI()

@app.get("/run-crew")
def run_crew():
    query = "What is artificial intelligence?"
    
    # Groq llm model
    llm = ChatGroq(model="groq/gemma2-9b-it")

    # Define Agent
    researcher = Agent(
        role="Researcher",
        goal="Find information about AI",
        backstory="An expert researcher who can gather knowledge from multiple sources.",
        llm=llm
    )

    # Define Task
    task = Task(
        description=query,
        agent=researcher,
        expected_output="A short summary of findings."
    )
    
    # Run Crew
    try:
        crew = Crew(agents=[researcher], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"result": str(result)}
    except Exception as e:
        print(e)
        return {"Error": str(e)}
