from crewai import Crew, Process
from .utils.logger import logger


from .agents import (
    build_text_parser_agent, build_ats_writer_agent, build_evaluator_agent, build_refiner_agent
)

from .tasks import (
    parse_resume_task, rewrite_for_ats_task, refine_bullets_task, evaluate_ats_task
)

def parse_texts(raw_resume_text: str):
    logger.info("Start parsing resume texts")
    agent = build_text_parser_agent()
    task = parse_resume_task(agent, raw_resume_text)
    
    # Build and run crew for parsing
    parse_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute parsing
    parse_result = parse_crew.kickoff()
    return str(parse_result).strip()    
    
def ats_writer(cleaned_text: str, job_title: str, job_description: str):
    logger.info("Start rewrting resume ATS")
    
    agent = build_ats_writer_agent()
    task = rewrite_for_ats_task(agent, cleaned_text, job_title, job_description)
    
    rewrite_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute rewriting
    rewrite_result = rewrite_crew.kickoff()
    return str(rewrite_result).strip()

def refine_bullets(rewritten: str):
    logger.info("Start refine bullets")
    agent = build_refiner_agent()
    # Create refine task with rewritten resume
    task = refine_bullets_task(agent, rewritten)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute refining
    refine_result = crew.kickoff()
    return str(refine_result).strip()

def evaluate_ats(final_resume: str, job_title: str, job_description: str):
    logger.info("Start evaluating ATS")
    # Create evaluation task with final resume
    evaluator = build_evaluator_agent()
    task = evaluate_ats_task(evaluator, final_resume, job_title, job_description)
    
    crew = Crew(
        agents=[evaluator],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute evaluation
    result = crew.kickoff()
    return str(result).strip()
    
    
def run_pipeline(raw_resume_text: str, job_title: str, job_description: str):
    cleaned_text = parse_texts(raw_resume_text)
    rewritten_text = ats_writer(cleaned_text, job_title, job_description)
    final_resume = refine_bullets(rewritten_text)
    evaluation = evaluate_ats(final_resume, job_title, job_description)
    return cleaned_text, rewritten_text, final_resume, evaluation
