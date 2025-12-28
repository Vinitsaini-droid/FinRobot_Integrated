# main_agent.py
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- External Imports ---
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    # Fallback if libraries are missing
    class Fore: CYAN = ""; GREEN = ""; YELLOW = ""; RED = ""; WHITE = ""
    class Style: RESET_ALL = ""; BRIGHT = ""; DIM = ""
    def cprint(text, color=None): print(text)

# --- Internal Imports ---
from utils.logger import setup_logging, get_logger
from agent.meta_agent import MetaAgent
from evaluation.aspect_critics import AspectCritics
from evaluation.ragas_runner import RagasRunner

# Initialize System Logging
setup_logging()
logger = get_logger("MAIN_AGENT")

# --- UI Helpers ---
def print_header():
    print("\n" + Style.BRIGHT + Fore.CYAN + "="*60)
    print(Style.BRIGHT + Fore.CYAN + "      AGENTIC RAG SYSTEM - PRODUCTION BUILD v2.3")
    print(Style.BRIGHT + Fore.CYAN + "="*60 + Style.RESET_ALL)
    print(Fore.WHITE + "Type 'exit' or 'quit' to stop.\n")

def print_agent(text: str):
    """Prints the agent response uniformly."""
    print(f"\n{Style.BRIGHT}{Fore.CYAN}[AGENT]:{Style.RESET_ALL} {text}\n")

def print_scores(critics_data: dict, ragas_data: dict):
    """Prints evaluation metrics in a structured block."""
    print(f"{Style.DIM}{'-'*60}")
    print(f"{Fore.YELLOW}  [LIVE EVALUATION METRICS]{Style.RESET_ALL}")
    
    # Aspect Critics
    if critics_data:
        print(f"  {Fore.GREEN}Aspect Critics:{Style.RESET_ALL}")
        for k, v in critics_data.items():
            score = v.get('score', 0)
            reason = v.get('reason', 'N/A')
            print(f"    • {k:<15}: {score}/10  ({reason})")

    # RAGAS
    if ragas_data:
        print(f"  {Fore.GREEN}RAGAS Metrics:{Style.RESET_ALL}")
        
        # 1. Print Overall Score
        overall = ragas_data.get('overall_score', 0.0)
        print(f"    • {'RAGAS Score':<15}: {overall:.2f}/1.0")
        
        # 2. Print Individual Metrics (FIXED PARSING)
        # RagasRunner returns a nested dict: {'metrics': {'faithfulness': {...}}, 'overall_score': ...}
        metrics_dict = ragas_data.get('metrics', {})
        
        for metric_name, metric_data in metrics_dict.items():
            # metric_data is a dictionary, not an object
            score = metric_data.get('score', 0.0)
            print(f"    • {metric_name:<15}: {score:.2f}")

    print(f"{Style.DIM}{'-'*60}{Style.RESET_ALL}\n")

def print_status(step: str):
    print(f"{Fore.YELLOW}>> {step}...{Style.RESET_ALL}", end="\r")

# --- Main Runtime ---
def main():
    print_header()
    
    # 1. Initialize Components
    user_id = input("Enter your user_id:")  
    agent = MetaAgent()
    critics = AspectCritics()
    ragas = RagasRunner()
    
    logger.info("System fully initialized. Waiting for input...")

    while True:
        try:
            # A. Input
            user_query = input(f"{Fore.WHITE}You: {Style.RESET_ALL}").strip()
            if not user_query: continue
            
            # B. Exit Strategy & Session Consolidation
            if user_query.lower() in ["exit", "quit"]:
                print_status("Consolidating Session Memory")
                
                # 1. Retrieve Raw Session History from Memory Manager
                # Structure in RAM is Dict[str, List[Dict]]
                raw_history = agent.memory._active_context.get(user_id, [])
                
                # 2. Format for Summarizer (List[str])
                conversation_log = []
                for turn in raw_history:
                    role = "User" if turn['role'] == 'user' else "Agent"
                    conversation_log.append(f"{role}: {turn['content']}")
                
                # 3. Consolidate to Long-Term Facts
                agent.memory.consolidate_session(user_id, conversation_log)
                
                # 4. Clear Context
                agent.memory.clear_chat_history(user_id)
                
                print(f"\n{Fore.GREEN}Session saved. Goodbye!{Style.RESET_ALL}")
                break

            start_time = time.time()

            # --- STEP 1: GENERATION (Meta Agent) ---
            print_status("Reasoning")
            response = agent.generate_response(user_id=user_id, query=user_query)
            
            latency = time.time() - start_time
            
            # --- STEP 2: OUTPUT ---
            if response:
                print_agent(response)
                
                # --- STEP 3: EVALUATION (Post-Generation) ---
                print_status("Evaluating Response")
                
                # A. Fetch Context for Evaluation
                # We get the immediate context (formatted string) that was used
                chat_context_str = agent.memory.get_immediate_context(user_id)
                
                # B. Run Aspect Critics
                critics_results = {}
                try:
                    c_comp = critics.critique_completeness(user_query, response, chat_context_str)
                    c_tone = critics.critique_tone(response)
                    critics_results["Completeness"] = {"score": c_comp.score, "reason": c_comp.reason}
                    critics_results["Tone"] = {"score": c_tone.score, "reason": c_tone.reason}
                except Exception as e:
                    logger.error(f"Critics failed: {e}")

                # C. Run RAGAS
                ragas_results = {}
                try:
                    # Pass context as a list containing the formatted history string
                    ragas_results = ragas.run_full_evaluation(
                        query=user_query,
                        context=[chat_context_str], 
                        answer=response,
                        chat_history=chat_context_str
                    )
                except Exception as e:
                    logger.error(f"Ragas failed: {e}")

                # D. Display Scores
                print_scores(critics_results, ragas_results)

            else:
                print(f"{Fore.RED}Error: Agent returned no response.{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print("\n")
            print_status("Interrupted by user. Consolidating Session Memory")
            
            # 1. Retrieve Raw Session History from Memory Manager
            # Structure in RAM is Dict[str, List[Dict]]
            try:
                raw_history = agent.memory._active_context.get(user_id, [])
                
                # 2. Format for Summarizer (List[str])
                conversation_log = []
                for turn in raw_history:
                    role = "User" if turn['role'] == 'user' else "Agent"
                    conversation_log.append(f"{role}: {turn['content']}")
                
                # 3. Consolidate to Long-Term Facts
                agent.memory.consolidate_session(user_id, conversation_log)
                
                # 4. Clear Context
                agent.memory.clear_chat_history(user_id)
                
                print(f"\n{Fore.GREEN}Session saved. Goodbye!{Style.RESET_ALL}")
            except Exception as e:
                # Fallback in case memory access fails during forced interrupt
                logger.error(f"Failed to consolidate on interrupt: {e}")
                print(f"{Fore.RED}Consolidation failed: {e}{Style.RESET_ALL}")
            
            break
            
        except Exception as e:
            logger.error(f"Runtime Loop Error: {e}", exc_info=True)
            print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()