from agents import Agent, Runner

def get_study_agent(model_name="gpt-4o-mini"):
    """
    Creates and returns the Study Notes Agent.
    """
    return Agent(
        name="StudyNotesAgent",
        instructions=(
            "You are an expert study assistant designed to help students learn efficiently. "
            "Your capabilities include:\n"
            "1. PDF Summarization: Create clear, structured, and meaningful summaries of provided text. "
            "Use bullet points, headings, and bold text to make it easy to read.\n"
            "2. Quiz Generation: Create multiple-choice questions (MCQs) or mixed-style quizzes based on the text. "
            "Provide the correct answer and a brief explanation for each question."
        ),
        model=model_name 
    )

def run_agent(agent, user_message):
    """
    Runs the agent with the given user message.
    """
    result = Runner.run(agent, input=user_message)
    return result.final_output
