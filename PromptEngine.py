import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


# Loading the model, has to be done first when starting streamlit to run the whole thing, IMP!!!!!!!!


import llm
# tokenizer, model = llm.model_init()


class KeywordEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_keyword(self, prompt):
        self.user_prompt = prompt
        self.system_prompt = """
        You are an intelligent assistant that extracts potential keywords for searching relevant news articles and internet content.

        The input text may include multiple unrelated topics. Your task is to extract **separate keyword phrases for each distinct topic** in the paragraph.

        Focus on named entities, events, and phrases someone might search on Google News or in academic/research databases. Do not include filler words or full sentences. Keep keywords short, relevant, and grouped by topic. Do not add any extra sentences under any circumstances.

        Format:
        Keywords 1: <comma-separated list>
        Keywords 2: <comma-separated list>
        ...
        If there's only one topic, just give a single line of keywords.
        """

        # maybe change prompt to have json style output like below.
        # below is the json file with the prompt 

        # """
        # You are an intelligent assistant that extracts potential keywords for searching relevant news articles and internet content.

        # The input text may contain one or more unrelated topics. Your task is to identify each topic and return a list of relevant keyword phrases **grouped by topic**.

        # Return the result in **valid JSON format**. Do not include any extra explanation or commentary—only the JSON object.

        # Focus on:
        # - Real-world entities, events, or research-related phrases
        # - Short, relevant search phrases (no full sentences)
        # - Group keywords logically per topic

        # Format:
        # {
        # "Topic 1": ["keyword1", "keyword2", ...],
        # "Topic 2": ["keyword1", "keyword2", ...]
        # }
        # If there's only one topic, just use a single key like:
        # {
        # "Topic": ["keyword1", "keyword2", ...]
        # }
        # """

        self.generated_text = llm.generate_response(llm_model=self.model, 
                                                    llm_tokenizer=self.tokenizer, 
                                                    prompt_user=self.user_prompt, 
                                                    prompt_system=self.system_prompt, 
                                                    temperature=0.3, 
                                                    max_tokens=200)
        
        print(self.generated_text)
        return self.generated_text


################################################################
####################    Script Engine     ######################
################################################################

class ScriptEngine:

    """Has 3 parameter model and tokenizer which is mandatory, followed by verbosity(verbosity == True means the article is very accurate, less story like, verbosity == False means the article is more story like) and max_tokens(max sequence length)"""



    def __init__(self, model, tokenizer, verbosity=False, max_tokens=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.verbosity = verbosity
        self.max_tokens = max_tokens


    def extract_keyword(self, prompt):

        self.user_prompt = prompt

        if self.verbosity == True: # case when the news accuracy is important
            self.system_prompt = """using the given json file and its hierarchy and timeline, create a news article with all the relevant information."""
            self.temperature = 0.1
        else: # case when news relevance to a story is more important
            self.system_prompt = """using the given json file and its hierarchy and timeline, create a story."""
            self.temperature = 0.7


        self.generated_text = llm.generate_response(llm_model=self.model,
                                                    llm_tokenizer=self.tokenizer, 
                                                    prompt_user=self.user_prompt, 
                                                    prompt_system=self.system_prompt, 
                                                    max_tokens=self.max_tokens)
        print(self.generated_text)
        return self.generated_text



###############################################################
#######################  Sample Code   ########################
###############################################################


# forward pass for the keyword engine, the keyword extracting pipeline
# keyword_engine = KeywordEngine(model=model, tokenizer=tokenizer)
# keyword_engine.extract_keyword(prompt="A random man went to mars and decided to set up base and start a family. There, a human was born. Scientists report a total loss of life on earth.", ) 

# # forward pass for the Script engine, the script writing code extracting pipeline
# script_engine = ScriptEngine(model=model, tokenizer=tokenizer, verbosity=True)
# script_engine.extract_keyword(prompt="""{
#   "conflict_hierarchy": {
#     "root_cause": {
#       "event": "October 7, 2023 Hamas Attack",
#       "date": "2023-10-07",
#       "description": "Hamas launched unprecedented surprise attack on Israel, killing ~1,200 people",
#       "source": "Iranian-backed Hamas militant group",
#       "consequences": ["Gaza War begins", "Regional tensions escalate"]
#     },
#     "escalation_phases": [
#       {
#         "phase": "Gaza War Phase",
#         "period": "October 2023 - April 2024",
#         "key_events": [
#           {
#             "event": "Israeli Military Response in Gaza",
#             "date": "October 2023",
#             "description": "Israel launches devastating military campaign in Gaza Strip",
#             "casualties": "Over 52,000 total deaths (50,810 Palestinian, 1,706 Israeli as of April 2025)"
#           },
#           {
#             "event": "Iran-backed Proxy Forces Mobilize",
#             "date": "Late 2023",
#             "description": "Iranian proxy forces ramp up strikes in protest of Israeli Gaza operations",
#             "affected_groups": ["Hezbollah", "Houthis", "Iraqi militias"]
#           }
#         ]
#       },
#       {
#         "phase": "Direct Iran-Israel Escalation",
#         "period": "April 2024 - June 2025",
#         "key_events": [
#           {
#             "event": "Iran's Direct Attack on Israel",
#             "date": "April 2024",
#             "description": "Iran launches 300+ missiles and drones directly at Israel",
#             "result": "Most intercepted by Israeli defenses"
#           },
#           {
#             "event": "Assassination Campaign",
#             "date": "Mid-Late 2024",
#             "description": "Israeli strikes kill high-profile figures including Hamas leaders Ismail Haniyeh and Yahya Sinwar",
#             "impact": "Tit-for-tat escalation continues"
#           },
#           {
#             "event": "Trump Nuclear Ultimatum",
#             "date": "April 2025",
#             "description": "Trump sets two-month deadline for Iran nuclear deal",
#             "deadline": "June 12, 2025"
#           }
#         ]
#       },
#       {
#         "phase": "Full-Scale War",
#         "period": "June 2025 - Present",
#         "key_events": [
#           {
#             "event": "Israeli Surprise Attack on Iran",
#             "date": "2025-06-13",
#             "description": "Israel launches surprise attacks on key Iranian military and nuclear facilities",
#             "trigger": "Day after Trump's nuclear ultimatum deadline expired",
#             "immediate_result": "Assassination of prominent Iranian military leaders"
#           },
#           {
#             "event": "US Military Intervention",
#             "date": "2025-06-22",
#             "description": "US Air Force and Navy attack three nuclear facilities in Iran",
#             "authorization": "President Trump orders bombing of Iranian nuclear sites"
#           },
#           {
#             "event": "Ceasefire Agreement",
#             "date": "2025-06-23",
#             "description": "Iran agrees to US-proposed ceasefire mediated by Qatar",
#             "current_status": "Temporary halt to hostilities"
#           }
#         ]
#       }
#     ],
#     "key_actors": {
#       "primary": {
#         "Israel": "Direct military action against Iran and Gaza",
#         "Iran": "Supports proxy groups, launches direct attacks",
#         "United States": "Military intervention, diplomatic mediation"
#       },
#       "proxies_and_allies": {
#         "Hamas": "Iranian-backed, initiated October 7 attack",
#         "Hezbollah": "Iranian proxy in Lebanon",
#         "Qatar": "Ceasefire mediator",
#         "UN": "Calls for peace, expresses alarm"
#       }
#     },
#     "escalation_pattern": {
#       "initial_trigger": "Hamas attack on Israel (Iranian proxy action)",
#       "escalation_mechanism": "Tit-for-tat strikes and assassinations",
#       "threshold_crossed": "Direct state-to-state warfare (June 13, 2025)",
#       "intervention_point": "US military involvement",
#       "current_status": "Ceasefire negotiations"
#     },
#     "timeline_summary": {
#       "duration": "21 months (October 2023 - July 2025)",
#       "escalation_speed": "Gradual proxy war → Rapid direct confrontation",
#       "casualties": "50,000+ in Gaza phase, unknown in Iran-Israel phase",
#       "geographic_scope": "Gaza → Regional → Iran mainland"
#     }
#   }
# }""")