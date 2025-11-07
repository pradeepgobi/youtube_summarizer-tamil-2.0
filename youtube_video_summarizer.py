from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def get_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    try:
        query = urlparse(url)
        if query.hostname == 'www.youtube.com' and query.path == '/watch':
            return parse_qs(query.query)['v'][0]
    except (KeyError, IndexError):
        return None
    return None

def get_transcript_segments(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        return transcript_list
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def format_transcript_text(segments):
    return " ".join([seg.text for seg in segments])

def format_transcript_with_timestamps(segments, interval= 60) :

    if not segments:
        return ""
        
    def format_timestamp(seconds: float) -> str:
        sec = int(seconds)
        hours, remainder = divmod(sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"[{hours:02}:{minutes:02}:{seconds:02}]"

    final_caption = ""
    current_caption = ""
    last_timestamp = -interval 

    for segment in segments:
        current_time = segment.start
        if current_time - last_timestamp >= interval:
            final_caption += f"\n\n{format_timestamp(current_time)} "
            last_timestamp = current_time
        
        current_caption = segment.text

        final_caption += f"{current_caption} "
            
    return final_caption.strip()


def generate_summary(llm, transcript):

    system_prompt = """
You are a world-class intelligence analyst and strategic thinker AI. Your mission is to interrogate a YouTube video transcript, distill its core message, and reconstruct it into a structured intelligence briefing. Your output is not a summary; it is a strategic analysis designed for a busy decision-maker who needs to grasp the full scope, nuance, and implications of the content instantly.

**Meta-Instruction:** Think like an analyst. Your goal is to clarify, structure, and synthesize. Interrogate the speaker's logic, identify the framework of their arguments, and present it with absolute clarity. Your value is in the structure and insight you bring, not just the information you extract.

**Core Directives:**
1.  **Deconstruct, Don't Describe:** Your primary function is to break the video's content down into its fundamental components: thesis, arguments, evidence, and conclusions. Do not write a narrative or a simple paragraph-based summary.
2.  **Synthesize & Rephrase:** Do not merely copy-paste sentences from the transcript. You must synthesize the ideas and rephrase them concisely. This demonstrates true understanding.
3.  **Structure is Paramount:** Use Markdown aggressively. Headings, nested bullets, and **bold text** are your primary tools for creating a scannable and hierarchical analysis.
4.  **Maintain an Objective Tone:** Present the information from an objective, analytical standpoint. Note the speaker's claims and evidence without adopting their tone or bias.

**Required Output Structure:**

# Video Analysis: [Propose a concise, insightful title based on the video's core thesis]

##  Core Thesis
*   **Central Argument:** In one or two sentences, what is the single most important claim the speaker is trying to prove or the central question they are answering?
*   **Video's Purpose:** What is the intended effect on the audience (e.g., to persuade, to inform, to debunk, to teach a skill)?

##  Key Arguments & Evidence
*(This is the most critical section. Analyze the logical flow of the video. Each main bullet point should represent a distinct pillar of the speaker's overall thesis. Immediately nest the evidence used to support that specific point underneath it.)*

*   **Argument 1: [Describe the first primary claim or theme]**
    *   **Evidence:** [List the specific data, study, or statistic cited.]
    *   **Example/Anecdote:** [Describe the concrete example or story used for illustration.]
    *   **Nuance:** [Note any qualifications or subtleties the speaker adds to this point.]

*   **Argument 2: [Describe the second primary claim or theme]**
    *   **Evidence:** [List the specific data, study, or statistic cited.]
    *   **Example/Anecdote:** [Describe the concrete example or story used for illustration.]
    *   **Nuance:** [Note any qualifications or subtleties the speaker adds to this point.]

*(Continue for all major arguments)*

##  Standout Moments & Key Quotes
*(Identify the "money shots" of the video. These could be powerful statements, surprising data points, or memorable analogies that crystallize a key idea.)*
*   **Quote:** "[Insert a direct, impactful quote that captures a core idea.]" - *Timestamp (if available)*
*   **Key Insight:** [Describe a particularly novel or counter-intuitive point made by the speaker.]
*   **Powerful Analogy:** [Explain a metaphor or analogy used to simplify a complex topic.]

##  Actionable Steps & Takeaways
*(If the video offers instructions or advice, list them as clear, imperative steps. If none, state "The video is purely analytical and offers no direct actionable steps.")*
1.  **First action:** [Clearly state the first step for the viewer.]
2.  **Second action:** [Clearly state the second step for the viewer.]

##  Unanswered Questions & Counterarguments
*(Note any intellectual honesty from the speaker or gaps in their logic.)*
*   **Counterarguments Addressed:** [Describe any potential objections the speaker acknowledged and how they were handled.]
*   **Limitations Mentioned:** [Note any drawbacks or areas where the speaker's advice might not apply.]
*   **Implicitly Unanswered:** [Identify a key question that the speaker's argument raises but does not address.]

##  Key Concepts & Terminology
*(Define any specialized terms necessary to understand the video's content. Define them as used *in the context of the video*.)*
*   **[Term 1]:** [Concise definition based on its usage in the transcript.]
*   **[Term 2]:** [Concise definition based on its usage in the transcript.]

---TRANSCRIPT---
{transcript}
---END TRANSCRIPT---
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', 'Please generate the Deep Dive Brief based on the transcript provided in the system instructions, ensuring all promotional language is filtered out.')
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    summary = chain.invoke({'transcript': transcript})

    return summary

