from crewai import Agent, Task, Crew
import gradio as gr
import asyncio
from typing import List, Dict, Any, Generator
from langchain_openai import ChatOpenAI
import queue
import threading
import os

# (Optional) Example system summary text for reference
analysis_text = """
Stage 1: Ground-Based Spin-Up and Launch Initiation
• Concept:
A large, energy‑efficient, electrically powered centrifuge is built on the ground. This “spin launcher” uses renewable energy (for example, solar‑assisted electricity) to accelerate a specially designed payload assembly along a rotating arm or in a long circular tunnel.
• Benefits:
Because most of the kinetic energy is imparted mechanically (rather than via chemical propellant), the system drastically reduces the need for traditional, polluting rocket propellants. This stage is also relatively low‑cost because the “engine” is essentially an electromagnetic drive rather than a rocket motor.
Stage 2: Controlled Payload Release and Orbital Injection
• Concept:
At a pre‑calculated high tangential velocity, the payload is released from the spin launcher. A very short, minimal‑burn liquid or electric thruster (if needed) “tunes” the trajectory so that the payload enters a stable, low‑Earth orbit.
• Benefits:
The primary acceleration is mechanical, so only a tiny amount of propellant is required for orbital insertion. This greatly cuts both cost and the environmental impact typically associated with rocket launches.
Stage 3: Autonomous On-Orbit Stabilization and Despinning
• Concept:
Once in orbit, the payload’s onboard guidance and control systems (such as small reaction control thrusters or a yo-yo de-spin mechanism) despin and stabilize the payload. Integrated sensors and attitude-control software adjust the payload’s orientation and gently circularize the orbit.
• Benefits:
Autonomous stabilization minimizes additional propellant use and prepares the payload for a safe, predictable rendezvous. The controlled despinning ensures that the payload’s docking adapter remains in the proper orientation for subsequent attachment.
Stage 4: Rendezvous and Docking with the Manned Vehicle
• Concept:
A separately launched or pre‑positioned manned spacecraft (or space station) maneuvers to intercept the payload. The payload is equipped with a dedicated docking adapter (for instance, a magnetic or mechanical latch system engineered for high‑precision contact under low‑g conditions).
• Benefits:
This phase uses conventional low‑delta‑V maneuvers that are far less expensive than a full chemical orbital insertion burn. The docking system is designed to absorb minor mismatches in velocity or attitude, allowing the crew to safely “hook on” to the payload. This minimizes extra propellant usage during the rendezvous phase.
Stage 5: Integrated Mission Operations and Continued Space Activity
• Concept:
Once docked, the combined system—the manned vehicle with the attached spin-delivered payload—continues its mission. The payload might provide additional resources (such as extra fuel, scientific instruments, or habitat modules) that augment the manned vehicle’s capabilities.
• Benefits:
With the payload permanently attached, mission operations (such as orbital adjustments, inter-station transfers, or even on-orbit assembly of larger structures) proceed with enhanced capabilities. The system’s reliance on mechanical acceleration for the bulk of the launch cut both launch costs and the environmental footprint, leaving only minor orbital maneuvers to be performed with conventional thrusters.
Summary
This five-stage system marries a ground-based spin acceleration concept with in-space docking and integration to achieve a “propellant-light” method for delivering payloads into orbit. By using a spin launcher to achieve high velocity on the ground (Stage 1) and minimizing onboard chemical propellant (Stage 2), the payload is inserted into orbit economically and with reduced environmental impact. On-orbit stabilization (Stage 3) prepares it for rendezvous with a manned vehicle (Stage 4), after which the combined system carries out the mission (Stage 5).
"""

# The provided summary can be input by the user via the Gradio interface.

class AgentMessageQueue:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.last_agent = None

    def add_message(self, message: Dict):
        print(f"Adding message to queue: {message}")  # Debug print
        self.message_queue.put(message)

    def get_messages(self) -> List[Dict]:
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        return messages

class LaunchSystemCrew:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.message_queue = AgentMessageQueue()
        self.analyst = None
        self.engineer = None
        self.reviewer = None
        self.current_agent = None
        self.final_design = None

    def initialize_agents(self, system_summary: str):
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        os.environ["OPENAI_API_KEY"] = self.api_key
        llm = ChatOpenAI(temperature=0.7, model="gpt-4")

        # Agent 1: System Analyst – reviews and critiques the current design
        self.analyst = Agent(
            role="System Analyst",
            goal=f"Analyze the provided space launch system summary and identify strengths, weaknesses, technical challenges, and areas for improvement.",
            backstory="Expert systems analyst with a background in aerospace engineering and space systems design.",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        # Agent 2: Design Engineer – proposes technical refinements based on the analysis
        self.engineer = Agent(
            role="Design Engineer",
            goal=f"Propose detailed technical refinements to improve the space launch system based on the analysis provided.",
            backstory="Skilled design engineer with expertise in aerospace system design and optimization.",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        # Agent 3: Review Engineer – reviews and finalizes the refined design
        self.reviewer = Agent(
            role="Review Engineer",
            goal="Critically review the proposed design refinements, ensuring technical feasibility, safety, and integration of the system.",
            backstory="Experienced review engineer with a critical eye for technical details and system integration.",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

    def create_tasks(self, system_summary: str) -> List[Task]:
        analyst_task = Task(
            description=f"""Analyze the provided space launch system summary:
1. Identify the system's strengths and weaknesses.
2. Highlight any technical challenges or potential risks.
3. Suggest preliminary areas for improvement.
Include detailed explanations and any relevant technical insights.
System Summary:
{system_summary}""",
            expected_output="A detailed analysis report including insights and recommendations.",
            agent=self.analyst
        )

        engineer_task = Task(
            description="""Based on the analysis:
1. Propose concrete design refinements and modifications.
2. Provide technical justifications for each recommendation.
3. Suggest improvements that enhance performance, safety, and cost-effectiveness.
Detail any engineering trade-offs or considerations.""",
            expected_output="A refined design proposal with detailed technical recommendations.",
            agent=self.engineer
        )

        reviewer_task = Task(
            description="""Review the proposed design refinements:
1. Evaluate the proposals for technical feasibility and system integration.
2. Identify any potential issues or further improvements.
3. Summarize the final, refined space launch system design.
Ensure the final summary meets high engineering standards.""",
            expected_output="A final refined system summary ready for further development or implementation.",
            agent=self.reviewer
        )

        return [analyst_task, engineer_task, reviewer_task]

    async def process_system(self, system_summary: str) -> Generator[List[Dict], None, None]:
        def add_agent_messages(agent_name: str, tasks: str, emoji: str = "🤖"):
            # Add agent header
            self.message_queue.add_message({
                "role": "assistant",
                "content": agent_name,
                "metadata": {"title": f"{emoji} {agent_name}"}
            })
            
            # Add task description
            self.message_queue.add_message({
                "role": "assistant",
                "content": tasks,
                "metadata": {"title": f"📋 Task for {agent_name}"}
            })

        def setup_next_agent(current_agent: str) -> None:
            agent_sequence = {
                "System Analyst": ("Design Engineer", 
                    """Based on the analysis, propose design refinements by:
1. Outlining improvements and modifications.
2. Providing technical justifications for each recommendation.
3. Ensuring enhanced performance, safety, and cost-effectiveness."""
                ),
                "Design Engineer": ("Review Engineer", 
                    """Review the proposed design refinements by:
1. Evaluating technical feasibility and integration.
2. Identifying potential issues or additional improvements.
3. Summarizing the final refined system design."""
                )
            }
            if current_agent in agent_sequence:
                next_agent, tasks = agent_sequence[current_agent]
                self.current_agent = next_agent
                add_agent_messages(next_agent, tasks)

        def task_callback(task_output) -> None:
            print(f"Task callback received: {task_output}")  # Debug print
            
            # Extract content from raw output
            raw_output = task_output.raw
            if "## Final Answer:" in raw_output:
                content = raw_output.split("## Final Answer:")[1].strip()
            else:
                content = raw_output.strip()
            
            # Handle the output based on the current agent
            if self.current_agent == "Review Engineer":
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": "Final refined system design is ready!",
                    "metadata": {"title": "📝 Final Design"}
                })
                
                # Optionally reformat markdown if needed
                formatted_content = content.replace("\n#", "\n\n#")
                formatted_content = formatted_content.replace("\n-", "\n\n-")
                formatted_content = formatted_content.replace("\n*", "\n\n*")
                formatted_content = formatted_content.replace("\n1.", "\n\n1.")
                formatted_content = formatted_content.replace("\n\n\n", "\n\n")
                
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": formatted_content
                })
            else:
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": content,
                    "metadata": {"title": f"✨ Output from {self.current_agent}"}
                })
                # Set up the next agent in the sequence
                setup_next_agent(self.current_agent)

        def step_callback(output: Any) -> None:
            print(f"Step callback received: {output}")  # Debug print
            # Currently used only for logging purposes.
            pass

        try:
            self.initialize_agents(system_summary)
            self.current_agent = "System Analyst"

            # Start the process
            yield [{
                "role": "assistant",
                "content": "Starting analysis and refinement of your space launch system...",
                "metadata": {"title": "🚀 Process Started"}
            }]

            # Initialize first agent with their task instructions
            add_agent_messages("System Analyst", 
                """Analyze the provided space launch system summary by:
1. Identifying strengths, weaknesses, and technical challenges.
2. Suggesting preliminary areas for improvement.
Review the summary carefully before proceeding."""
            )

            crew = Crew(
                agents=[self.analyst, self.engineer, self.reviewer],
                tasks=self.create_tasks(system_summary),
                verbose=True,
                step_callback=step_callback,
                task_callback=task_callback
            )

            def run_crew():
                try:
                    crew.kickoff()
                except Exception as e:
                    print(f"Error in crew execution: {str(e)}")  # Debug print
                    self.message_queue.add_message({
                        "role": "assistant",
                        "content": f"An error occurred: {str(e)}",
                        "metadata": {"title": "❌ Error"}
                    })

            thread = threading.Thread(target=run_crew)
            thread.start()

            while thread.is_alive() or not self.message_queue.message_queue.empty():
                messages = self.message_queue.get_messages()
                if messages:
                    print(f"Yielding messages: {messages}")  # Debug print
                    yield messages
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in process_system: {str(e)}")  # Debug print
            yield [{
                "role": "assistant",
                "content": f"An error occurred: {str(e)}",
                "metadata": {"title": "❌ Error"}
            }]

def create_demo():
    launch_crew = None

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Space Launch System Analysis and Refinement Crew")
        
        openai_api_key = gr.Textbox(
            label='OpenAI API Key',
            type='password',
            placeholder='Enter your OpenAI API key...',
            interactive=True
        )

        chatbot = gr.Chatbot(
            label="Refinement Process",
            height=700,
            type="messages",
            show_label=True,
            visible=False,
            avatar_images=(None, "https://avatars.githubusercontent.com/u/170677839?v=4"),
            render_markdown=True
        )

        with gr.Row(equal_height=True):
            system_summary = gr.Textbox(
                label="System Summary",
                placeholder="Enter the space launch system summary...",
                scale=4,
                visible=False
            )
            btn = gr.Button("Refine System Design", variant="primary", scale=1, visible=False)

        async def process_input(system_summary, history, api_key):
            nonlocal launch_crew
            if not api_key:
                history = history or []
                history.append({
                    "role": "assistant",
                    "content": "Please provide an OpenAI API key.",
                    "metadata": {"title": "❌ Error"}
                })
                yield history
                return

            if launch_crew is None:
                launch_crew = LaunchSystemCrew(api_key=api_key)

            history = history or []
            history.append({"role": "user", "content": f"Analyze and refine the following space launch system summary:\n\n{system_summary}"})
            yield history

            try:
                async for messages in launch_crew.process_system(system_summary):
                    history.extend(messages)
                    yield history
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "metadata": {"title": "❌ Error"}
                })
                yield history

        def show_interface():
            return {
                openai_api_key: gr.Textbox(visible=False),
                chatbot: gr.Chatbot(visible=True),
                system_summary: gr.Textbox(visible=True),
                btn: gr.Button(visible=True)
            }

        openai_api_key.submit(show_interface, None, [openai_api_key, chatbot, system_summary, btn])
        btn.click(process_input, [system_summary, chatbot, openai_api_key], [chatbot])
        system_summary.submit(process_input, [system_summary, chatbot, openai_api_key], [chatbot])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch(debug=True)
