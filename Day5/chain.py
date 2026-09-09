import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

Job_description = """
we are hiring a backend python developer

Requirement:
- Strong Python
- FastAPI or Django
- PostgreSQL
- AWS
- REST APIs
- 2+ years of experience
"""

resume = """
Profile
Experienced Software Developer with expertise in design, installation, testing and maintenance of software systems. Equipped with a diverse and promising skill-set. Proficient in various platforms, languages, and embedded systems. Experienced with cutting-edge development tools and procedures. Able to effectively self-manage during independent projects, as well as collaborate as part of a productive team.

Work Experience
09/2013 - 09/2021, Software Developer, Accrue Partners, New York

Collaborated with the product team to understand requirements and business specifications around portfolio management, analytics and risk.
Coded software updates and alterations based on detailed design specifications.
Solved complex problems using the latest in cloud, mobile, and web technologies.
Developed and presented findings and solutions to audiences including senior executives and stakeholders.
06/2011 - 08/2013, Junior Software Developer, CyberCoders, New York

Addressed and fixed complex bugs.
Implemented and updated application modules under the direction of Senior Software Developers.
Worked at an independent level, while also serving as an effective and enthusiastic collaborator.
Performed automated testing tasks and developed complex features routinely.
Education
08/2010 - 08/2012, Master of Science in Computer Science, The Massachusetts Institute of Technology., Cambridge, MA

09/2006 - 05/2010, Bachelor of Computer Science, Dartmouth, Hanover, NH

Skills
Advanced Analytical Thinking
Programming
Software Logic
postgreSQL
AWS
Software Troubleshooting
Knowledgable in User Interface/ User Experience
Adaptability
"""


def ask_llm(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def step1_res_extract():
    system_prompt = """You are a professional HR assistant. Extract the skills from the resume provided.
Only return the skills in a list format without any other text.
"""
    user_prompt = """Extract the skills from the following resume:
{resume}
"""
    return ask_llm(system_prompt, user_prompt.format(resume=resume))


def step2_job_extract():
    system_prompt = """You are a professional HR assistant. Extract the skills from the job description provided.
Only return the skills in a list format without any other text.
"""
    user_prompt = """Extract the skills from the job description:
{job_description}
"""
    return ask_llm(system_prompt, user_prompt.format(job_description=Job_description))


def step3_match(candidate_skills, job_description):
    system_prompt = """You are a professional HR assistant. Match the candidate skills with the job description provided.
Only return the matched skills in a list format without any other text. Produce a score between 1 and 100 and also create the verdict for the candidate, including strengths and areas to improve.
"""
    user_prompt = """Match the candidate skills with the following job description:
{job_description}
candidate skills:
{candidate_skills}
"""
    return ask_llm(system_prompt, user_prompt.format(job_description=job_description, candidate_skills=candidate_skills))


candidate = step1_res_extract()
job_match = step2_job_extract()
score = step3_match(candidate, job_match)
print(score)

