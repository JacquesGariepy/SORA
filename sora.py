import os
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union
from scipy.stats import ttest_ind, f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import torch
import torch.nn as nn
import torch.optim as optim

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.memory import ConversationBufferMemory

import autogen
from crewai import Agent, Task, Crew, Process

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

class SORAStudy:
    def __init__(self, num_agents: int, num_scenarios: int, epochs: int = 100):
        self.num_agents = num_agents
        self.num_scenarios = num_scenarios
        self.epochs = epochs
        self.agents = []
        self.autogen_agents = []
        self.scenarios = []
        self.interaction_data = pd.DataFrame()
        self.metrics = {}
        self.neural_net = None
        self.llm = OpenAI(temperature=0.7)
        self.loss_history = []

    def setup(self):
        logger.info("Setting up the SORA study environment...")
        self.create_agents()
        self.create_scenarios()
        self.setup_neural_network()
        self.setup_langchain()
        self.setup_autogen_agents()

    def create_agents(self):
        logger.info(f"Creating {self.num_agents} agents...")
        roles = ["Analyst", "Decision Maker", "Executor", "Monitor", "Innovator"]
        for i in range(self.num_agents):
            role = roles[i % len(roles)]
            agent = Agent(
                name=f"Agent_{i}",
                role=role,
                goal=f"Perform tasks as a {role} in various complex scenarios",
                backstory=f"An AI agent specialized in {role} tasks with evolving capabilities and high agentivity",
                allow_delegation=True,
                verbose=True,
                llm=self.llm
            )
            self.agents.append(agent)

    def create_scenarios(self):
        logger.info(f"Creating {self.num_scenarios} complex scenarios...")
        scenario_types = [
            "Crisis Management", "Market Analysis", "Scientific Research",
            "Urban Planning", "Environmental Conservation", "Healthcare Innovation"
        ]
        for i in range(self.num_scenarios):
            self.scenarios.append({
                "id": i,
                "type": random.choice(scenario_types),
                "complexity": np.random.uniform(0, 1),
                "urgency": np.random.uniform(0, 1),
                "ethical_implications": np.random.uniform(0, 1)
            })

    def setup_neural_network(self):
        logger.info("Setting up neural network for agent decision making...")
        input_size = 5  # scenario features + agent role encoding
        hidden_size = 64
        output_size = len(self.scenarios[0]) - 1  # exclude 'id'
        self.neural_net = NeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def setup_langchain(self):
        logger.info("Setting up LangChain components for metacognition...")
        self.prompt_template = CustomPromptTemplate(
            template="You are an AI agent with high agentivity, specialized in {agent_role}. "
                     "Given the following complex scenario: {scenario_description}, "
                     "how would you approach it? Consider the complexity, urgency, "
                     "ethical implications, and potential for collaborative problem-solving.\n{agent_scratchpad}",
            tools=self.create_tools(),
            input_variables=["agent_role", "scenario_description", "agent_scratchpad"]
        )
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.create_tools()]
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")

    def setup_autogen_agents(self):
        logger.info("Setting up AutoGen agents for diverse behavior...")
        for agent in self.agents:
            autogen_agent = autogen.ConversableAgent(
                name=agent.name,
                system_message=f"You are an AI agent with high agentivity, specialized in {agent.role} tasks. "
                                "Your goal is to exhibit diverse and adaptive behavior in complex scenarios.",
                llm_config={"config_list": [{"model": "gpt-3.5-turbo"}]}
            )
            self.autogen_agents.append(autogen_agent)

    def create_tools(self):
        return [
            Tool(
                name="Analyze",
                func=lambda x: "Analysis complete: " + x,
                description="Use for in-depth analysis of complex situations"
            ),
            Tool(
                name="Decide",
                func=lambda x: "Decision made: " + x,
                description="Use to make strategic decisions considering multiple factors"
            ),
            Tool(
                name="Execute",
                func=lambda x: "Action executed: " + x,
                description="Use to implement planned actions in the scenario"
            ),
            Tool(
                name="Monitor",
                func=lambda x: "Monitoring results: " + x,
                description="Use to track and evaluate outcomes of actions"
            ),
            Tool(
                name="Innovate",
                func=lambda x: "Innovation proposed: " + x,
                description="Use to generate creative solutions to complex problems"
            )
        ]

    def output_parser(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    def run_simulation(self):
        logger.info("Running SORA simulation...")
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            for scenario in self.scenarios:
                tasks = [
                    Task(
                        description=f"Analyze {scenario['type']} scenario with complexity {scenario['complexity']:.2f}",
                        agent=self.agents[0]
                    ),
                    Task(
                        description=f"Make decision for {scenario['type']} scenario with urgency {scenario['urgency']:.2f}",
                        agent=self.agents[1]
                    ),
                    Task(
                        description=f"Execute plan for {scenario['type']} scenario",
                        agent=self.agents[2]
                    ),
                    Task(
                        description=f"Monitor outcomes of {scenario['type']} scenario",
                        agent=self.agents[3]
                    ),
                    Task(
                        description=f"Propose innovations for {scenario['type']} scenario with ethical implications {scenario['ethical_implications']:.2f}",
                        agent=self.agents[4]
                    )
                ]
                
                crew = Crew(
                    agents=self.agents,
                    tasks=tasks,
                    process=Process.sequential
                )
                crew_result = crew.kickoff()
                
                # AutoGen multi-agent conversation for diverse behavior
                autogen_manager = autogen.GroupChatManager(
                    groupchat=autogen.GroupChat(agents=self.autogen_agents, messages=[]),
                    llm_config={"config_list": [{"model": "gpt-3.5-turbo"}]}
                )
                autogen_result = autogen_manager.run(
                    f"Discuss the results and implications of the {scenario['type']} scenario. "
                    f"Consider ethical implications, adaptability, and potential for emergent behavior. "
                    f"Crew results: {crew_result}"
                )
                
                self.record_interactions(scenario, crew_result + "\n" + autogen_result, epoch)
            self.train_neural_network(epoch)

    def record_interactions(self, scenario: Dict, result: str, epoch: int):
        for agent in self.agents:
            interaction_data = {
                "epoch": epoch,
                "scenario_id": scenario["id"],
                "scenario_type": scenario["type"],
                "scenario_complexity": scenario["complexity"],
                "scenario_urgency": scenario["urgency"],
                "scenario_ethical_implications": scenario["ethical_implications"],
                "agent_name": agent.name,
                "agent_role": agent.role,
                "interaction_content": result
            }
            self.interaction_data = self.interaction_data.append(interaction_data, ignore_index=True)

    def train_neural_network(self, epoch: int):
        logger.info(f"Training neural network for adaptive decision making - Epoch {epoch + 1}")
        epoch_data = self.interaction_data[self.interaction_data['epoch'] == epoch]
        
        inputs = epoch_data[['scenario_complexity', 'scenario_urgency', 'scenario_ethical_implications']].values
        targets = pd.get_dummies(epoch_data['scenario_type']).values

        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)

        self.optimizer.zero_grad()
        outputs = self.neural_net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        logger.info(f"Epoch {epoch + 1} Loss: {loss.item()}")

    def measure_behavioral_diversity(self) -> float:
        logger.info("Measuring behavioral diversity...")
        unique_interactions = self.interaction_data.groupby(['agent_name', 'scenario_type']).size().reset_index(name='count')
        total_possible = len(self.agents) * len(set(self.interaction_data['scenario_type']))
        return len(unique_interactions) / total_possible

    def measure_metacognition(self) -> float:
        logger.info("Measuring metacognition...")
        self.interaction_data['metacognition_score'] = self.interaction_data['interaction_content'].apply(
            lambda x: len([w for w in x.lower().split() if w in ['think', 'consider', 'reflect', 'evaluate', 'assess', 'analyze', 'reason']])
        )
        return self.interaction_data['metacognition_score'].mean()

    def measure_adaptability(self) -> float:
        logger.info("Measuring adaptability...")
        agent_performance = self.interaction_data.groupby('agent_name')['scenario_complexity'].agg(['mean', 'std'])
        return 1 - (agent_performance['std'] / agent_performance['mean']).mean()

    def measure_transparency(self) -> float:
        logger.info("Measuring transparency...")
        self.interaction_data['explanation_length'] = self.interaction_data['interaction_content'].str.len()
        max_length = self.interaction_data['explanation_length'].max()
        return (self.interaction_data['explanation_length'] / max_length).mean()

    def measure_social_complexity(self) -> float:
        logger.info("Measuring social complexity...")
        G = nx.from_pandas_edgelist(self.interaction_data, 'agent_name', 'scenario_id')
        return nx.average_clustering(G)

    def calculate_metrics(self):
        logger.info("Calculating SORA metrics...")
        self.metrics['behavioral_diversity'] = self.measure_behavioral_diversity()
        self.metrics['metacognition'] = self.measure_metacognition()
        self.metrics['adaptability'] = self.measure_adaptability()
        self.metrics['transparency'] = self.measure_transparency()
        self.metrics['social_complexity'] = self.measure_social_complexity()

    def perform_statistical_analysis(self):
        logger.info("Performing statistical analysis...")
        scenario_types = self.interaction_data['scenario_type'].unique()
        for i in range(len(scenario_types)):
            for j in range(i+1, len(scenario_types)):
                type1 = scenario_types[i]
                type2 = scenario_types[j]
                group1 = self.interaction_data[self.interaction_data['scenario_type'] == type1]['scenario_complexity']
                group2 = self.interaction_data[self.interaction_data['scenario_type'] == type2]['scenario_complexity']
                t_stat, p_value = ttest_ind(group1, group2)
                logger.info(f"T-test results for {type1} vs {type2}: t-statistic = {t_stat}, p-value = {p_value}")

        scenario_groups = [group for _, group in self.interaction_data.groupby('scenario_type')['scenario_complexity']]
        f_statistic, p_value = f_oneway(*scenario_groups)
        logger.info(f"ANOVA results: F-statistic = {f_statistic}, p-value = {p_value}")

        tukey_results = pairwise_tukeyhsd(self.interaction_data['scenario_complexity'], 
                                          self.interaction_data['scenario_type'])
        logger.info("Tukey's test results:")
        logger.info(tukey_results)

    def visualize_results(self):
        logger.info("Visualizing SORA study results...")
        # Agent Interaction Network
        G = nx.from_pandas_edgelist(self.interaction_data, 'agent_name', 'scenario_id')
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        plt.title("Agent Interaction Network in SORA Study")
        plt.savefig("sora_agent_interaction_network.png")
        plt.close()

        # Agentivity Metrics
        metrics_df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', data=metrics_df)
        plt.title("SORA Agentivity Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("sora_agentivity_metrics.png")
        plt.close()

        # PCA of Agent Performance
        agent_performance = self.interaction_data.groupby('agent_name').agg({
            'scenario_complexity': 'mean',
            'scenario_urgency': 'mean',
            'scenario_ethical_implications': 'mean',
            'metacognition_score': 'mean'
        })
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        scaled_data = scaler.fit_transform(agent_performance)
        pca_result = pca.fit_transform(scaled_data)

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA of Agent Performance in SORA Study')
        for i, agent in enumerate(agent_performance.index):
            plt.annotate(agent, (pca_result[i, 0], pca_result[i, 1]))
        plt.tight_layout()
        plt.savefig("sora_agent_performance_pca.png")
        plt.close()

        # Neural Network Training Loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss in SORA Study')
        plt.savefig("sora_neural_network_loss.png")
        plt.close()

    def cluster_analysis(self):
        logger.info("Performing cluster analysis on agent behavior...")
        agent_behavior = self.interaction_data.groupby('agent_name').agg({
            'scenario_complexity': 'mean',
            'scenario_urgency': 'mean',
            'scenario_ethical_implications': 'mean',
            'metacognition_score': 'mean'
        })
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(agent_behavior)
        
        inertias = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), inertias, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k in SORA Agent Clustering')
        plt.savefig("sora_elbow_curve.png")
        plt.close()
        
        optimal_k = 3  # This should be determined by analyzing the elbow curve
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Cluster Analysis of Agent Behavior in SORA Study')
        plt.colorbar(scatter)
        for i, agent in enumerate(agent_behavior.index):
            plt.annotate(agent, (pca_result[i, 0], pca_result[i, 1]))
        plt.tight_layout()
        plt.savefig("sora_agent_behavior_clusters.png")
        plt.close()
        
        return cluster_labels

    def analyze_ethical_implications(self):
        logger.info("Analyzing ethical implications of agent decisions...")
        ethical_scores = self.interaction_data.groupby('agent_name')['scenario_ethical_implications'].mean()
        
        plt.figure(figsize=(12, 6))
        ethical_scores.sort_values().plot(kind='bar')
        plt.title('Average Ethical Implication Scores by Agent in SORA Study')
        plt.xlabel('Agent')
        plt.ylabel('Average Ethical Implication Score')
        plt.tight_layout()
        plt.savefig("sora_ethical_implications_by_agent.png")
        plt.close()
        
        correlation_matrix = self.interaction_data[['scenario_complexity', 'scenario_urgency', 
                                                    'scenario_ethical_implications', 'metacognition_score']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Scenario Factors and Agent Performance in SORA Study')
        plt.tight_layout()
        plt.savefig("sora_correlation_matrix.png")
        plt.close()

    def generate_report(self):
        logger.info("Generating comprehensive SORA study report...")
        report = f"""
        SORA Study Report: Evaluation and Amplification of Agentivity in Multi-Agent AI Systems
        =======================================================================================

        1. Overview
        -----------
        Number of Agents: {self.num_agents}
        Number of Scenarios: {self.num_scenarios}
        Epochs: {self.epochs}

        2. Key Metrics
        --------------
        {pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value']).to_string(index=False)}

        3. Statistical Analysis
        -----------------------
        [Include summary of t-tests, ANOVA, and Tukey's test results]

        4. Cluster Analysis
        -------------------
        [Include summary of cluster analysis results]

        5. Ethical Implications
        -----------------------
        [Include summary of ethical implication analysis]

        6. Neural Network Performance
        -----------------------------
        Final Loss: {self.loss_history[-1]}

        7. CrewAI, AutoGen, and LangChain Integration
        ---------------------------------------------
        The study utilized CrewAI for task allocation and workflow management across {len(self.agents)} specialized agents.
        AutoGen facilitated multi-agent discussions, enhancing behavioral diversity and adaptability.
        LangChain was employed for metacognitive processes and complex reasoning tasks.

        8. Hypotheses Evaluation
        ------------------------
        H1 (Behavioral Diversity): {self.metrics['behavioral_diversity']:.2f}
        H2 (Metacognition): {self.metrics['metacognition']:.2f}
        H3 (Adaptability): {self.metrics['adaptability']:.2f}
        H4 (Transparency): {self.metrics['transparency']:.2f}
        H5 (Social Complexity): {self.metrics['social_complexity']:.2f}

        9. Conclusions and Future Work
        ------------------------------
        This SORA study demonstrates the potential of multi-agent systems with high agentivity in addressing complex scenarios.
        The integration of CrewAI, AutoGen, and LangChain has shown promising results in enhancing behavioral diversity,
        metacognition, adaptability, transparency, and social complexity of AI systems.

        Future research directions:
        - Explore the emergence of more complex social structures in AI ecosystems
        - Develop adaptive ethical frameworks for highly agentive AI systems
        - Investigate the potential for human-AI symbiosis in problem-solving
        - Enhance the scalability and robustness of multi-agent systems in real-world applications

        [Additional detailed analysis and insights to be added based on specific study outcomes]
        """
        with open("sora_study_report.txt", "w") as f:
            f.write(report)

    def run_study(self):
        self.setup()
        self.run_simulation()
        self.calculate_metrics()
        self.perform_statistical_analysis()
        self.visualize_results()
        cluster_labels = self.cluster_analysis()
        self.analyze_ethical_implications()
        self.generate_report()
        logger.info("SORA study completed. Results, visualizations, and report generated.")

if __name__ == "__main__":
    study = SORAStudy(num_agents=20, num_scenarios=50, epochs=100)
    study.run_study()
